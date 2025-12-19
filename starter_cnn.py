import torch
import torch.nn as nn
import torch.nn.functional as F


# AYTO EINAI TO TEMPLATE TOY CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Todo: ορίστε εδώ τα convolution layers
        # self.conv1 = ...
        #...

        # Todo: pooling, flattening, fully connected layer
        #...


    def forward(self, x):
        # Todo: γράψτε την εμπρός διάδοση
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # ...
        return x


# Μετατροπή εικόνας σε tensor
def image_to_tensor(image):
    # Todo: μετατροπή εικόνας σε tensor (C,H,W)
    tensor = torch.tensor(image, dtype=torch.float32)
    # tensor = ...
    return tensor


# Μετατροπή ετικετών σε αριθμητικά tensors
def label_to_tensor(label):
    # TODO: ορισμός mapping ετικετών -> αριθμών
    name_to_index = {
        # "label_name": index,
    }
    # index = name_to_index[label]
    # return torch.tensor(index)
    pass


# Επιλογή συσκευής
device = "cuda" if torch.cuda.is_available() else "cpu"