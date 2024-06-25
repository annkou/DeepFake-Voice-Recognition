# DeepFake Voice Recognition

This repository contains the code for the DeepFake Voice Recognition project, which aims to build models capable of distinguishing between original and deepfake audio.


The initial dataset includes both original audio clips and deepfake audio clips. To download the dataset, as well as the trained models showcased in the deep learning notebooks, visit [this Google Drive](https://drive.google.com/drive/folders/1HnDNgGu_MmIQTSAAsbpc2xE2jWBMkHXr?usp=drive_link) directory.



The source code for the deep learning component is located in the `dl` directory. This directory contains all the necessary functions and classes to prepare the dataset, train and evaluate the models, and define the model architectures. The actual training and evaluation are performed in the notebooks in the root directory.

To get started, visit the [Introduction](./Introduction.ipynb) in order to get familiar with our thought process and dataset.

Before exploring the other notebooks, please refer to the [deep-learning-prep](./deep-learning-prep.ipynb) notebook. This is where the initial exploratory data analysis (EDA) and preparation of the audio files are performed to extract the necessary data for modeling and training.

Two approaches where used to tackle this problem: deep learning and traditional machine learning, as a baseline comparison.

Each of these approaches can further be explored in the notebooks bellow:
* [deep-learning-modeling](./deep-learning-modelling.ipynb)
* [deep-learning-modeling-aug](./deep-learning-modelling-aug.ipynb)
* [machine-learning-classic](./machine-learning-classic.ipynb)
* [machine-learning-cleaned](./machine-learning-cleaned.ipynb)
* [machine-learning-augmented](./machine-learning-augmented.ipynb)