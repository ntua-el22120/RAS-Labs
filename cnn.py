import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)

        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)



        self.flatten = nn.Flatten()
        self.maxPool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)  # flattened size = 16,384

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128→124→62
        x = F.relu(self.conv2(x)) # 62→58
        x = F.relu(self.conv3(x))  # 58→54
        x = F.relu(self.conv4(x))  # 54→52
        x = F.relu(self.conv5(x))  # 52→50
        x = self.maxPool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def image_to_tensor(image):
    # Convert to float tensor, shape (C, H, W)
    tensor = torch.tensor(image, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1) / 255.0
    # add batch dimension, shape (B, C, H, W)
    tensor = tensor.unsqueeze(0)
    return tensor

def label_to_tensor(label):
    name_to_index = {
        "nothing": 0,
        "stop": 1,
        "priority": 2,
        "roundabout": 3
    }
    index = name_to_index[label]
    tensor = torch.tensor(index, dtype=torch.long)
    tensor = tensor.unsqueeze(0)
    return tensor



device = "cuda" if torch.cuda.is_available() else "cpu"