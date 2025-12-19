### **Project 1: Face Recognition**

Face recognition is one of the most used applications in machine vision, and today deep learning models have achieved very high accuracy in this application which consists of two stages: face classification, i.e. classification of images based on ID, and face verification, i.e. determining whether two images are of the same person's face or not. Considering this, in this project, we first design a classifier for the face images in the dataset. Then, using this classification and the embeddings that this classification provides, we will implement the face verification task.

The model considered for the task of classifying images in this project is resnet 50 and Norm 2 is used to calculate the distance between the input embedding vector and the stored vectors.
