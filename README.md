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

![Example eLEAF Output](https://github.com/amrosado/TF-EfficientLEAF/blob/aaron/initial_commit/example_output/eLEAF/eleaf_example_10.png?raw=true)

## Example Automatic Speech Recognition Transformer Output
```
target:     <forgive me i hardly know what i am saying a thousand times forgive me madame was right quite right this brutal exile has completely turned my brain>
prediction: <for get me i hardly no on what i am saying the thousand time me madame was right quite right as brittle excile as complictly turned my brain>

target:     <there cannot be a doubt he received you kindly for in fact you returned without his permission>
prediction: <their cannot be a doubt he received you kindly for in fact you returned without his permission>

target:     <oh mademoiselle why have i not a devoted sister or a true friend such as yourself>
prediction: <oh met must elie have i not a divoted sister or a true or a true friend the such as yourself>

target:     <what already here they said to her>
prediction: <what already here may said to her>
```
