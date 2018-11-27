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


### Demo

A simple demo `demo.py` is provided to recognize text from an image file using pretrained model. The pretrained model can be found [here](https://www.dropbox.com/s/81j7zcr23vqd8zq/model.ckpt.tar.gz?dl=0).

```
python demo.py --path=data/demo.jpg --checkpoint=PATH_TO_PRETRAINED_MODEL
```


### Training

1. Prepare training dataset.

    Prepare training datasets into tfrecord format. You can customize your datasets based our tfrecord tool under `tools/make_tfrecord_datasets.py`.

2. Start training.

    - Train from scratch.
    ```
    ./train.sh
    ```

    - Or use the pretrained model by adding an additional flag: `--checkpoint=--checkpoint=PATH_TO_PRETRAINED_MODEL`

3. Evaluate the model continuously during training.

    ```
    ./eval_continous.sh
    ```


### Citation

Please cite our paper if our work is helpful to you.

```
@inproceedings{fang2018attention,
  title={Attention and Language Ensemble for Scene Text Recognition with Convolutional Sequence Modeling},
  author={Fang, Shancheng and Xie, Hongtao and Zha, Zheng-Jun and Sun, Nannan and Tan, Jianlong and Zhang, Yongdong},
  booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
  pages={248--256},
  year={2018},
  organization={ACM}
}
```


Note: Our work is based on the previous work [conv_seq2seq](https://github.com/tobyyouup/conv_seq2seq).
Thanks the authors for sharing the code.
