# Blind Spot Detection using Machine Learning
A lightweight camera-based solution for checking a blind spot programmatically using TensorFlow and Python on a Raspberry Pi. 

## Repository Contents
This repository contains multiple elements of the project. These elements include:
- Jupyter Notebook going into how the model was made
- The trained model
- Python program that uses the model on two cameras

## Model Info
Using transfer learning on MobileNetV2, an accuracy of ~98% was reached for blind spot detection with an average prediction time of 0.09s on the Raspberry Pi 4 without any machine learning accelerators. Given an ML accelerator such as the [Google Coral USB Accelerator](https://coral.ai/products/accelerator/), it would likely reach prediction times of 0.0026s (2.6ms, [source](https://coral.ai/docs/edgetpu/benchmarks/)).

## Demo Video
Want to skip straight into the details? Check out [this video](https://youtu.be/gVqHdGIRrTY) demoing the machine learning algorithm.

[![Demo Video](https://i.imgur.com/ZLRfkQ5.png)](https://youtu.be/gVqHdGIRrTY)

## Copyright
© 2021 Aritro Saha. All rights reserved.

Modification and redistribution of the source code is not allowed.
