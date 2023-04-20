# TF-EfficientLEAF
## TensorFlow Implementation of EfficientLEAF: A Faster LEarnable Audio Frontend of Questionable Use
## About
This repository aims to reproduce the Efficient LEAF front-end model using TensorFlow 
and Keras towards enabling learnable audio frontends in TensorFlow without using Gin 
and Lingvo dependencies limiting the usability and compatibility of the original LEAF library. The 
code heavily reproduces the original code featured in the EUSIPCO EfficientLEAF: 
A Faster Learnable Audio Frontend of Questionable Use published by Jan Schlüter 
and Gerald Gutenbrunner (https://arxiv.org/abs/2207.05508).  The original GitHub 
repo can be found at: https://github.com/CPJKU/EfficientLEAF.

Thank you to Jan and Gerald for answering my questions regarding their 
implementation and their support of making Efficient LEAF available 
to the TensorFlow community.

## Tested with:
* Python 3.9
* Tensorflow 2.10.x (Windows) and Tensorflow 2.12.x (Linux)

## Using:
* Nvidia CUDA >=11.2.2
* Nvidia CUDNN >=8.1.0.77

## Dependencies:
* Tensorflow 2
* Huggingface Datasets
* Librosa
* Soundfile

## Example eLEAF output

![Example eLEAF Output](https://github.com/amrosado/TF-EfficientLEAF/blob/initial_commit/example_output/eLEAF/eleaf_example_10.png?raw=true)

## Example Automatic Speech Recognition Transformer Output

