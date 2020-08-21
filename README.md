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

-- First download the dataset (I only implemented the code for 'car' or 'chair' dataset) from
[Google Drive](https://drive.google.com/drive/folders/1YbgU-JOXYsGi7yTrYb1F3niXj6nZp4Li?usp=sharing) <br>
-- Create new folder and extract the dataset to the new folder <br>
-- Train the model with 
```
python train.py --dataset_path 'new_folder_path'
```
<br>

If the new folder is 'C:\Datasets\continuous_view_synthesis_dataset\' the command is:
```
python train.py --dataset_path 'C:\Datasets\continuous_view_synthesis_dataset\'
```