# Calibration in Convolutional Neural Networks (CNNs)

This project explores techniques to improve the calibration of Convolutional Neural Networks (CNNs), ensuring that the predicted probabilities align more closely with actual outcomes. Proper calibration is crucial for applications where reliable probability estimates are essential.
Overview

Neural networks often produce overconfident predictions, leading to miscalibration. This project investigates methods to assess and enhance the calibration of CNNs, focusing on:

    Expected Calibration Error (ECE): A metric that quantifies the discrepancy between predicted confidence and actual accuracy.

    Temperature Scaling: A post-processing technique that adjusts the logits to improve calibration without affecting the model's accuracy.

## Dataset

The experiments utilize the CIFAR-10 dataset, which comprises 60,000 32x32 color images across 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.
Model Architecture

A Convolutional Neural Network (CNN) is implemented using PyTorch. The architecture includes:

    Multiple convolutional layers with ReLU activations and max-pooling.

    Fully connected layers leading to the final output layer with softmax activation for classification.

## Training

The model is trained on the CIFAR-10 training set using cross-entropy loss and the Adam optimizer. After training, the model's calibration is evaluated using the ECE metric.
Calibration Techniques

    Temperature Scaling: This method involves introducing a temperature parameter to scale the logits before applying the softmax function. The temperature parameter is optimized on a validation set to minimize the negative log-likelihood, thereby improving calibration.

## Results

The application of temperature scaling demonstrates a reduction in the Expected Calibration Error (ECE), indicating improved alignment between predicted probabilities and actual outcomes.
