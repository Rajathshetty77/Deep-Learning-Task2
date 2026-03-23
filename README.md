# Deep Learning Project: CNN, Sequence Models and GAN

## Overview

This project demonstrates the implementation of different deep learning models using PyTorch. It includes:

- CNN for image classification using Fashion-MNIST
- RNN, LSTM and GRU for sequence-based learning
- GAN for generating synthetic images

The aim of this project is to understand how these models work and observe their behavior during training.

---

## Project Structure

deep-learning-project/
│
├── main.py
├── requirements.txt
├── README.md
│
├── final_outputs/
│   └── gan_samples/

---

## How to Run

1. Install required libraries:

pip install -r requirements.txt

2. Run the program:

python main.py

---

## Description of Models

### CNN (Fashion-MNIST)
- The CNN model is trained on grayscale clothing images.
- It uses convolution and pooling layers for feature extraction.
- The model learns to classify images into different clothing categories.

### Sequence Models (RNN, LSTM, GRU)
- Synthetic sequence data is used for training.
- RNN, LSTM, and GRU are implemented and compared.
- LSTM and GRU generally provide more stable results.

### GAN (Image Generation)
- A GAN is trained on Fashion-MNIST images.
- The generator creates images from random noise.
- The discriminator evaluates whether images are real or fake.
- Generated images improve gradually over training epochs.

---

## Results

- CNN shows consistent learning with decreasing loss.
- LSTM and GRU perform better than basic RNN.
- GAN outputs evolve from noise to structured images.

---

## Outputs

Generated images can be found in:

final_outputs/gan_samples/

These images show the progress of GAN training across epochs.

---

## Observations

- CNN is effective for image classification tasks.
- LSTM and GRU handle sequence data better than RNN.
- GAN training is sensitive and requires careful tuning.

---

## Conclusion

This project helped in understanding different deep learning architectures and their applications. It also provided practical experience in training and analyzing model performance.

---

## Author

Rajath D Shetty
