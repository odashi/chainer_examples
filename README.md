Chainer example code for NLP
============================

*These samples are based on Chainer 1.4 or earlier.*

This repository contains some neural network examples
for natural language processing (NLP)
using **Chainer** framework.

[Chainer Official](http://chainer.org/ "Chainer official") ([GitHub](https://github.com/pfnet/chainer "Github"))

Contents
--------

* **Machine Translation**
    * `mt_s2s_encdec.py` - Using encoder-decoder style recurrent neural network
    * `mt_s2s_attention.py` - Using attentional neural network

* **Word Segmentation (Tokenization)**
    * `seg_ffnn.py` - Using feedforward neural network
    * `seg_rnn.py` - Using recurrent neural network

* **Language Model**
    * `lm_rnn.py` - Using recurrent neural network (RNNLM)

Note that this repos does not include `my_settings.py`
which is used only for specifying machine-dependent import paths.

Contact
-------

If you find an issue or have some questions, please contact Yusuke Oda:
* @odashi_t on Twitter (faster than other methods)
* yus.takara (at) gmail.com

