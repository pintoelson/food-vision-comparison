from torch import nn
import torchvision
import torch

class Food101_V0(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        print
        self.linear_layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
class Food101_V1(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape,
            out_channels = hidden_units,
            kernel_size = 3,
            stride = 1,
            padding = 1),
            nn.ReLU(),

            nn.Conv2d(in_channels = hidden_units,
            out_channels = hidden_units,
            kernel_size = 3,
            stride = 1,
            padding = 1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 4),
        )

        

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units,
            out_channels = hidden_units,
            kernel_size = 3,
            stride = 1, 
            padding = 1),
            nn.ReLU(),

            nn.Conv2d(in_channels = hidden_units,
            out_channels = hidden_units,
            kernel_size = 3,
            stride = 1,
            padding = 1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units * 28 * 28,
                    out_features = output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

#######################################FoodV2#######################################

weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT # .DEFAULT = best available weights 
model = torchvision.models.efficientnet_v2_s(weights=weights)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(    
    nn.Dropout(0.5),
    torch.nn.Linear(in_features=1280, 
                    out_features=1280, # same number of output units as our number of classes
                    bias=True),
    nn.Dropout(0.5),
    torch.nn.Linear(in_features=1280, 
                    out_features=101, # same number of output units as our number of classes
                    bias=True))

Food101_V2 = model



#######################################FoodV3#######################################

weights = torchvision.models.ResNet101_Weights.DEFAULT # .DEFAULT = best available weights 
model = torchvision.models.resnet101(weights=weights)

model.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=2048, 
                    out_features=2048, # same number of output units as our number of classes
                    bias=True),
    nn.Dropout(0.2),
    nn.ReLU(),
    torch.nn.Linear(in_features=2048, 
                    out_features=2048, # same number of output units as our number of classes
                    bias=True),
    nn.Dropout(0.2),
    nn.ReLU(),
    torch.nn.Linear(in_features=2048, 
                    out_features=101, # same number of output units as our number of classes
                    bias=True))

Food101_V3 = model