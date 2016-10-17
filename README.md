# lstm-machine-translation
Implementation of Machine Translation via TensorFlow

## Overview

This code implements the "Sequence to Sequence Learning with Neural Networks" published by Google (https://arxiv.org/abs/1409.3215). The model is suitable for machine translation and is considered to be among the recent state-of-the art models used by Google for Machine Translation.

There are many improvements that can be added in data the data preprocessing part that can further lower the training error and accuracy, however the current model should be a good working prototype to begin with.

For convenience we don't use real translated data, but in this case have decided to translate a sentence to a new sentence with reversed words and word order (e.g. "how are you" -> "uoy era woh"). You can reference the original paper why this mimics quite well the complexity of learning how to translate a spoken language.

The model uses embedding vectors to store the words understandings.
"#" is used as a "GO" symbol.
"." is used as a "\<EOS>" symbol.

## Further development

- Download real dataset.
- Add buckets and padding in order to support sentences with different lengths

## Possible tunings

At the moment the model is using 4 unrollings, followed by a "GO" symbol.
The number of layers, max number of nodes and embedding size are easily tunable to test different configurations.

## Results

The current results were achieved on 12,000 steps of batch training with 4 unrollings. Experimentations with bigger epochs size might be interesting:

```
Average loss at step 11900 : 1.37952561259 learning rate: 0.103912
Minibatch perplexity: 37.87
Validation set perplexity: 41.58
[' defined self by label # lebal yb fles denifed']
```

## Requirements
    python 2.7
    tensorflow 0.6.0

Tensorflow can be installed using

    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
    pip install --ignore-installed --upgrade $TF_BINARY_URL
