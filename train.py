"""
Workshop Project: Training a Simple CNN

Goal:
- Load a CNN model
- Load images and labels from disk
- Create training batches
- Train the model using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import json

import cnn  # Contains the CNN architecture and helper functions


# --------------------------------------------------
# 1. Model setup
# --------------------------------------------------

# Create the model and move it to the correct device (CPU/GPU)
model = cnn.SimpleCNN().to(cnn.device)

# OPTIONAL:
# Load previously trained weights
# Todo: uncomment and explain when loading pretrained models is useful
# model.load_state_dict(torch.load("simple_cnn_weights.pth"))


# --------------------------------------------------
# 2. Training configuration
# --------------------------------------------------

# Define loss function

# .......

# Define optimizer

# ....... 

# ΠΕΙΡΑΜΑΤΙΣΤΕΙΤΕ ΜΕ ΔΙΑΦΟΡΟΥΣ ΤΥΠΟΥ OPTIMIZERS ΚΑΙ ΡΥΘΜΟΥΣ ΜΑΘΗΣΗΣ, ΚΑΙ LOSS FUNCTIONS

# Number of images in the dataset
num_pictures = 5000

# Batch size

# δοκιμάστε διαφορετικά μεγέθη batch

# Path to training data
train_set_path = "PATH_TO_TRAIN_SET"


# --------------------------------------------------
# 3. Dataset loading and batching
# --------------------------------------------------

# This list will store all training batches
batches = []

# Todo:
# Loop over the dataset in steps of batch_size
# For each batch:
#   - Load images from disk using OpenCV
#   - Convert images to tensors
#   - Load labels from JSON files
#   - Convert labels to tensors
#   - Store (images, labels) as one batch

# ΧΡΗΣΙΜΟΠΟΙΕΙΣΤΕ ΤΑ helper functions cnn.image_to_tensor ΚΑΙ cnn.label_to_tensor

# HINT:
# for i in range(0, num_pictures, batch_size):
#     ...


# --------------------------------------------------
# 4. Training loop
# --------------------------------------------------

num_epochs = 30

for epoch in range(num_epochs):

    running_loss = 0.0

    # todo:
    # Loop over all batches
    # For each batch:
    #   1. Combine image tensors into one batch tensor
    #   2. Combine label tensors into one label tensor
    #   3. Move tensors to device
    #   4. .....
    #   5. ......
    # ΨΑΞΤΕ ΣΤΟ internet ΠΩΣ ΝΑ ΓΡΑΨΕΤΕ ΕΝΑ TRAINING LOOP ΣΕ PYTORCH

    # Print average loss for the epoch
    # todo: compute average loss
    print(f"Epoch {epoch + 1} completed")


# --------------------------------------------------
# 5. Saving the model
# --------------------------------------------------

# Save trained model weights
# todo: explain why we save only state_dict
torch.save(model.state_dict(), "simple_cnn_weights.pth")