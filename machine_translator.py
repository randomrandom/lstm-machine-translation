import os
import sys
import numpy as np
import random
import string
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import collections
import urllib
import zipfile

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print 'Found and verified', filename
  else:
    print statinfo.st_size
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return f.read(name)
  f.close()
  
text = read_data(filename)
print "Data size", len(text)

# Create a small validation set
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print train_size, train_text[:64]
print valid_size, valid_text[:64]


vocabulary_size = 25000
unk_sign = 'UNK'
eos_sign = '.'
eos_index = 0
go_sign = '#'
go_index = 1

def build_words_dataset(text): 
  words = text.split()
  
  count = [(unk_sign, -1)]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 3))
  
  index = 0
  dictionary = dict()
  x_dictionary = dict()
  
  # adding go sign
  dictionary[go_sign] = len(dictionary)
  x_dictionary[go_sign] = len(x_dictionary)
  go_index = len(x_dictionary)
  
  # adding eos sign
  dictionary[eos_sign] = len(dictionary)
  x_dictionary[eos_sign] = len(x_dictionary)
  eos_index = len(x_dictionary)
  
  # adding 
  for word in count:
    if word not in dictionary:
      dictionary[word[0]] = len(dictionary)
      reversed_word = word[0][::-1]
      if unk_sign == word[0]: reversed_word = unk_sign
      x_dictionary[reversed_word] = len(x_dictionary)
    
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  x_reverse_dictionary = dict(zip(x_dictionary.values(), x_dictionary.keys()))
  return dictionary, reverse_dictionary, x_dictionary, x_reverse_dictionary

dictionary, reverse_dictionary, x_dictionary, x_reverse_dictionary = \
  build_words_dataset(train_text + valid_text) # we don't use text because there might be bad word split
  
def word_to_id(word, dictionary=dictionary):
  
  if word in dictionary:
    return dictionary[word]
  else:
    return dictionary[unk_sign]
	
def embeddings_to_ids(final_embeddings, embeds):
  bigram_ids = []
  for i in xrange(embeds.shape[0]):
      nominator = np.dot(final_embeddings, embeds[i])
      denominator = la.norm(embeds[i])
      cosims = nominator / denominator
      bigram_ids.append(np.argmax(cosims))
  return bigram_ids
      
def probs_to_ids(probabilities):
  return [c for c in np.argmax(probabilities, 1)]

def prob_to_char_id(probability):
  return np.argmax(probability)

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution, bottom_start=0):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in xrange(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction, bottom_start=0):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[vocabulary_size], dtype=np.float)
  p[sample_distribution(prediction[0], bottom_start)] = 1.0
  return p

def get_best_prediction(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[vocabulary_size], dtype=np.float)
  p[np.argmax(prediction, 1)] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]

# print train_batches._words_count
print len(train_text)
print len(dictionary)
print len(reverse_dictionary)
# print train_batches._cursor
reverse_dictionary[24999]

batch_size=64
num_unrollings=4

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):  
    self._words_text = text.split()
    self._words_count = len(self._words_text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._words_count / batch_size
    self._segment_size = segment
    self._cursor = [ offset * segment for offset in xrange(batch_size)]
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in xrange(self._batch_size):
      batch[b, word_to_id(self._words_text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._words_count
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in xrange(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches
  
  @staticmethod
  def create_input_sequence(batches, batch_size, input1_size, input2_size):
    all_inputs = list()
    input_size = input1_size + input2_size

    # setup inputs
    for i in xrange(input1_size):
      data = probs_to_ids(batches[i])
      all_inputs.append(data)
      
    all_inputs = all_inputs[::-1]
    all_inputs.append([dictionary[go_sign]] * batch_size)

    translation_input_start = input1_size + 1

    for i in xrange(translation_input_start, input_size, 1):
      data = probs_to_ids(batches[i-translation_input_start])
      reversed_data = list()
      for word_id in data:
        word = reverse_dictionary[word_id]
        reversed_word = word[::-1]
        reverse_word_id = word_to_id(reversed_word, x_dictionary)
        reversed_data.append(reverse_word_id)
      data = reversed_data
      all_inputs.append(data)

    return all_inputs
  
  @staticmethod
  def create_label_sequence(all_inputs, batch_size, input1_size, input2_size):
    # setup outputs
    all_labels = list()
    input_size = input1_size + input2_size

    outputs_end_without_eos = input_size - 1
    for i in xrange(input1_size, outputs_end_without_eos, 1):  
      data = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
      ids = all_inputs[i+1]
      for j in xrange(len(ids)):
        data[j, 0] = ids[j]
      all_labels.append(data)

    # add the eos sign
    data = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    ids = [dictionary[eos_sign]] * batch_size
    for j in xrange(len(ids)):
        data[j, 0] = ids[j]
    all_labels.append(data)

    return all_labels

  @staticmethod
  def label_ids_to_probs(all_labels, vocabulary_size):
    probabilities = list()
    for label in all_labels:
      p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
      p[0, label] = 1.0
      probabilities.append(p)
    return probabilities
    
def words(probabilities, dictionary=reverse_dictionary):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character representation."""
  return [dictionary[c] for c in np.argmax(probabilities, 1)]

def batches2sentence(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [' '.join(x) for x in zip(s, words(b))]
  return s

def sequences2sentence(sequences, input1_size):
  if input1_size > 0: # we have training inputs
    s = [''] * np.array(sequences).shape[0]
  else:
    s = [''] * np.array(sequences).shape[1]
    
  for idx, b in enumerate(sequences):
    if input1_size == 0: # we have labels
      b = np.concatenate(b)
    if idx < input1_size:
      converted_words = [reverse_dictionary[id] for id in b]
    else:
      converted_words = [x_reverse_dictionary[id] for id in b]
    s = [' '.join(x) for x in zip(s, converted_words)]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, num_unrollings)

#print batches2sentence(train_batches.next())
#print batches2sentence(train_batches.next())
t_batches = train_batches.next()
print batches2sentence(t_batches)
train_inputs = BatchGenerator.create_input_sequence(t_batches, batch_size, num_unrollings, num_unrollings + 1)
print sequences2sentence(train_inputs, num_unrollings)
t_labels = BatchGenerator.create_label_sequence(train_inputs, batch_size, num_unrollings, num_unrollings + 1)
print sequences2sentence(t_labels, 0)
print "Validation:"
#print batches2sentence(valid_batches.next())
#print batches2sentence(valid_batches.next())
v_batch = valid_batches.next()
print batches2sentence(v_batch)
valid_inputs = BatchGenerator.create_input_sequence(v_batch, 1, num_unrollings, num_unrollings + 1)
print sequences2sentence(valid_inputs, num_unrollings)
v_labels = BatchGenerator.create_label_sequence(valid_inputs, 1, num_unrollings, num_unrollings + 1)
print sequences2sentence(v_labels, 0)

num_nodes = 64
embedding_size = 128
num_steps = 12000
number_of_layers = 4
num_sampled = 64 # Number of negative examples to sample.
sentence_length = num_unrollings

train_input_size = 2*sentence_length + 1
label_input_size = sentence_length + 1

train1_input_size = sentence_length
train2_input_size = train_input_size - train1_input_size

graph = tf.Graph()
with graph.as_default():
  
  # Dropout
  keep_prob = tf.placeholder(tf.float32) 
  
  # Parameters:    
  # Definition of the LSTM cells
  lstm = rnn_cell.BasicLSTMCell(num_nodes)
  if keep_prob < 1:
      lstm = rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
  stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)
  
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes * (2*number_of_layers)]), trainable=False)
  
  # Embedding variables
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  x_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))

  # Input data.
  train_data = list()
  train_labels = list()
  
  # Define input & label variables
  for x in xrange(train_input_size):
    train_data.append(tf.placeholder(tf.int32, shape=[batch_size]))
    
  for x in xrange(label_input_size):  
    train_labels.append(tf.placeholder(tf.int32, shape=[batch_size, 1]))
  
  # Convert the input variables into embeddings
  encoded_inputs = list()
  
  # Encoding the input sequence
  for i in xrange(train1_input_size):
    words_batch = train_data[i]
    embed = tf.nn.embedding_lookup(embeddings, words_batch)
    encoded_inputs.append(embed)

  # Encoding the output sequence
  for i in xrange(train1_input_size, train_input_size):
    words_batch = train_data[i]
    embed = tf.nn.embedding_lookup(x_embeddings, words_batch)
    encoded_inputs.append(embed)
  train_inputs = encoded_inputs

  # Unrolled LSTM loop.
  outputs = list()
  state = saved_state
  output = saved_output

  # we want the following mapping: A B C D # -> W X Y Z .
  with tf.variable_scope("LSTM-encoder") as scope:
    # input sequence
    for i in xrange(train1_input_size):
      if i > 0: scope.reuse_variables()
      output, state = stacked_lstm(train_inputs[i], state)
      
  with tf.variable_scope("LSTM-decoder") as scope:
    for i in xrange(train1_input_size, train_input_size):
      if i > train1_input_size: scope.reuse_variables()
      output, state = stacked_lstm(train_inputs[i], state)
      outputs.append(output)
    
  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    
    # input transformation
    all_inputs = tf.concat(0, outputs)
    w_t = tf.transpose(w)
    # output transformation
    all_labels = tf.concat(0, train_labels)
    
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
      tf.nn.sampled_softmax_loss(w_t, b, all_inputs, all_labels, num_sampled, vocabulary_size))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, num_steps / 2, 0.1, staircase=False)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
   
  # Sampling and validation eval
  sample_inputs = list()
  for i in xrange(sentence_length):
    sample_inputs.append(tf.placeholder(tf.int32, shape=[1]))
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes * (2*number_of_layers)]), trainable=False)
  saved_translation_output = tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
  saved_translation_state = tf.Variable(tf.zeros([1, num_nodes * (2*number_of_layers)]), trainable=False)
  translation_input = tf.placeholder(tf.int32, shape=[1], name="translation_input")
  
  reset_sample_state = tf.group(
    saved_translation_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes * (2*number_of_layers)])),
    saved_translation_state.assign(tf.zeros([1, num_nodes * (2*number_of_layers)])))
  
  sample_state = saved_sample_state
  with tf.variable_scope("LSTM-encoder", reuse=True) as scope:
    for sample_input in sample_inputs:
      sample_embed = tf.nn.embedding_lookup(embeddings, sample_input)
      sample_output, sample_state = stacked_lstm(sample_embed, sample_state)
      
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state),
                                  saved_translation_state.assign(sample_state)]):
      sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(saved_sample_output, w, b))
  
  with tf.variable_scope("LSTM-decoder", reuse=True) as scope:
    translation_embed = tf.nn.embedding_lookup(x_embeddings, translation_input)
    translation_output, translation_state = stacked_lstm(translation_embed, saved_translation_state)

    with tf.control_dependencies([saved_translation_output.assign(translation_output),
                                  saved_translation_state.assign(translation_state)]):
        sample_translation_prediction = tf.nn.softmax(tf.nn.xw_plus_b(saved_translation_output, w, b))        
		
summary_frequency = 100
sample_words_count = 39

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print 'Initialized'
  mean_loss = 0
  for step in xrange(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    
    all_inputs = BatchGenerator.create_input_sequence(batches, batch_size,
                                                      sentence_length, sentence_length + 1)
    all_labels = BatchGenerator.create_label_sequence(all_inputs, batch_size,
                                                      sentence_length, sentence_length + 1)
    
    # setup inputs
    for i, sequence in enumerate(all_inputs):
      feed_dict[train_data[i]] = sequence
    
    # setup outputs
    for i, sequence in enumerate(all_labels):
      feed_dict[train_labels[i]] = sequence
    
    # setup dropout
    feed_dict[keep_prob] = 0.8
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print 'Average loss at step', step, ':', mean_loss, 'learning rate:', lr
      mean_loss = 0
      labels = np.concatenate(all_labels)
      labels = BatchGenerator.label_ids_to_probs(labels, vocabulary_size)
      labels = np.concatenate(labels)
      print 'Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels)))

      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      last_sequence = ''
      last_labels = ''
      last_prediction = ''
      min_perplexity = sys.maxint
      valid_iterations = valid_size / 7
      for _ in xrange(valid_iterations):
        reset_sample_state.run()
        
        # feeding the input sequence to be translated
        v_batch = valid_batches.next()
        v_input_ids = BatchGenerator.create_input_sequence(v_batch, 1, sentence_length, sentence_length + 1)
        sample_dict = dict()
        for i, v_input in enumerate(v_input_ids[:sentence_length]):
          sample_dict[sample_inputs[i]] = v_input
        sample_dict[keep_prob] = 1.0
        v_predictions = list()
        prob_predictions = list()
        prediction = sample_prediction.eval(sample_dict)
        
        # starting the translation by inputing the "GO" sign
        sentence = ''
        go_symbol = dictionary[go_sign]
        feed = [go_symbol]
        for _ in xrange(sentence_length + 1):
          prediction = sample_translation_prediction.eval({translation_input: feed, keep_prob: 1.0})
          feed = sample(prediction)
          sentence += ' ' + words([feed], x_reverse_dictionary)[0]
          feed = probs_to_ids([feed])
          v_predictions.append(feed)
          prob_predictions.append(prediction)
            
        # convert labels into probabilities so we can measure perplexity
        v_label_ids = BatchGenerator.create_label_sequence(v_input_ids, 1, sentence_length, sentence_length + 1)
        v_labels = BatchGenerator.label_ids_to_probs(v_label_ids, vocabulary_size)
        v_labels = np.concatenate(v_labels)
        prob_predictions = np.concatenate(prob_predictions)
        
        # get the best case as a display sample
        perplexity = logprob(prob_predictions, v_labels)
        if perplexity < min_perplexity:
          min_perplexity = perplexity
        
        last_sequence = sequences2sentence(v_input_ids, sentence_length)
        last_labels = sequences2sentence(v_label_ids, 0)
        last_prediction = sentence
        valid_logprob = valid_logprob + perplexity
      print 'Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_iterations))
      print last_sequence
      print last_labels
      print last_prediction
