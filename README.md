
# My VRDL HW3

Use detectron2 model zoo's pretrained model to do

model from detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl

## Requirements

To install requirements

follow the step from  https://detectron2.readthedocs.io/en/latest/tutorials/install.html to install detectron2

## My environment
GUP: GTX3060

CUDA: 11.3
cudnn: 8.2.1

## Training

To train the model(s) in the paper, run this command:

```train
python train_custom.py
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/file/d/1BHcU5-P-zK5tOsZ2gaqF7VlCr8e0yumR/view?usp=sharing) 


## Inference
to reproduce submission file

```Inference
python eval_custom_final_v10.py
```



