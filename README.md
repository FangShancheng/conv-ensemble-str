## Attention and Language Ensemble for Scene Text Recognition with Convolutional Sequence Modeling


### Requirements

- Tensorflow-gpu r1.3.0
- Python 2.7
- CUDA 8.0

### Introduction

![architecture](https://github.com/FangShancheng/conv-ensemble-str/raw/master/figures/architecture.png)

The proposed architecture consists of an encoder that extracts abstractive image features, and a decoder that generates character sequences. The encoder is a residual convolutional network, and the decoder is also based on the deep convolutional network.

Concretely, the first layer of the encoder is a base convolutional layer which converts images to feature maps in specific distribution. Then a deep encoder is built by stacking residual blocks. Also, there is a similar structure in the decoder. After embedding input symbols, the stacked decoder blocks predict the output sequences. Inside each decoder block, an attention module and a language
module are designed equally as an ensemble. The attention module focuses on interest region from encoder feature maps, whose main operations are scaled dot-product. The language module, based on gated convolutional layers, aims to model the language sequences in character level. In our work, visual cues and linguistic rules are of the same importance. Based on this point, attention focusing and language modeling with the same input are regarded as an ensemble to boost prediction jointly. In addition, we use batch normalization in the encoder and layer normalization in the decoder to keep variances stable across the main nodes of the networks.
 
### Usage

1. Prepare training dataset.

2. Train the model.

    > ./train.sh

3. Evaluate the model continuously.

    > ./eval_continous.sh

5. Evaluate the model one pass.

    > ./eval_one_pass.sh


Note: Our work is based on the previous work [conv_seq2seq](https://github.com/tobyyouup/conv_seq2seq).
Thanks the authors for sharing the code.
