import torch
import torch.nn as nn

from torchvision import models

class ResNet50Classifier(nn.Module):
    def __init__(self):
        super(ResNet50Classifier, self).__init__()
        # Load a pre-trained ResNet-50 model
        self.base_model = models.resnet50()
        # Replace the final fully connected layer with a custom classification layer
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)

    def forward(self, x):
        return self.base_model(x)
