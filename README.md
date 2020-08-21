# Monocular Neural Image-based Rendering with Continuous View Control

Unofficial implementation of '[**Monocular Neural Image-based Rendering with Continuous View Control**](https://arxiv.org/abs/1901.01880)' using **mxnet gluon**.
This model can generate novel views of objects from only one view, with fine-grained control over the virtual viewpoints. This repository contain some code from the original implementation and some parts of the encoder-decoder changed. <br>

Paper: https://arxiv.org/abs/1901.01880

Official implementation: https://github.com/xuchen-ethz/continuous_view_synthesis

## Reproducing Results

### Installation

Install all packages with pip, run 
```bash
pip install -r requirements.txt
```
mxnet 2.0.0 can be found here https://dist.mxnet.io/python/all

### Training

To train the model run the following command 

```
python train.py
```
