import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision, timm

class MyModel(nn.Module):
    """Some Information about MyModel"""
    def __init__(self, model_name, num_classes, is_timm=True):
        super().__init__()

        if is_timm:
            self.model = timm.create_model(model_name, pretrained=True)
            n_features = self.model.num_features
            self.mask_classifier = timm.models.layers.ClassifierHead(n_features, num_classes)

    def forward(self, x):
        features = self.model.forward_features(x)
        x = self.mask_classifier(features)
        y = self.gender_classifier(features)
        z = self.age_classifier(features)

        return x, y, z