# Machine Learning - Apple Disease Detection

This project implements a deep learning model using transfer learning with MobileNetV2 to detect diseases in apple fruits. The model is trained on a dataset sourced from Kaggle and utilizes MLflow for experiment tracking and model management.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [MLflow Tracking](#mlflow-tracking)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Project Overview

The goal of this project is to build a robust image classification model that can accurately identify different diseases affecting apples based on images. This can be valuable for agricultural applications, enabling early detection and management of diseases.

## Features

- **Transfer Learning:** Leverages the power of a pre-trained convolutional neural network (MobileNetV2) to benefit from learned features on a large dataset (ImageNet).
- **Data Augmentation:** Techniques like rotation, zooming, and flipping are applied to the training data to increase the size and diversity of the dataset, improving the model's ability to generalize.
- **MLflow Integration:** Tracks various aspects of the machine learning lifecycle, including parameters, metrics, and artifacts (model weights, plots, reports). This facilitates experiment comparison and reproducibility.
- **Callbacks:** Utilizes Keras callbacks such as ModelCheckpoint (to save the best model), EarlyStopping (to prevent overfitting), and ReduceLROnPlateau (to adjust the learning rate during training).

## Dataset

The dataset used in this project is available on Kaggle. It contains images of apple fruits categorized by their disease status.

**(Link to the dataset on Kaggle: https://www.kaggle.com/datasets/ateebnoone/fruits-dataset-for-fruit-disease-classification)**

Ensure you have downloaded the dataset and organized it into appropriate directories for training, validation, and testing. The notebook assumes a directory structure like:
