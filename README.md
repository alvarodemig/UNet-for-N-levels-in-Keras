# UNet for N levels in Keras

Implementation os the Convolutional Nerual Network proposed in the paper: **"U-Net: Convolutional Networks for Biomedical Image Segmentation"**, which can be found here: https://arxiv.org/pdf/1505.04597.pdf


Model Structure
----------------------------------------------------------------
![UNet model](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

The code for the model presents some different approaches compared to the original model:
  - Padding: same. It means that the size of the image after each convolution will not change.
  - Upsampling instead of Deconvolution (UpSampling2D + Conv2D instead of Conv2DTranspose)

This repository extends and generalize this structure including the next options:
  - Levels: total number of levels that the UNet will have (default: 5)
  - Initial channels: we can choose how many channels we will have in the 1st level (default: 32)
  - Channels rate: how the number of filters will be modified per level (default: 2)
  - Activation function (default: ReLu)
  - Batch Normalization: we can include this step (default: 0, as in the original paper)
  - Dropout: we can include a dropout layer after convolutions, values from 0 to 1. (default: 0, as in the original paper)

-------------------------------------------------------------------------------------
Programming Language: Python
Libraries: Keras
