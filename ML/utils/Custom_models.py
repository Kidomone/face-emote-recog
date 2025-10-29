import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def Resnet_Custom(output_shape=7, load_path=None, to_hidden=False):
    if load_path is not None:
        model = torch.load(load_path, weights_only=False)
        return model
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # changing input shape for gray scale
    pretrained_conv1 = model.conv1.weight
    new_conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    new_conv1.weight.data = pretrained_conv1.mean(dim=1, keepdim=True)
    model.conv1 = new_conv1

    # change output shape for classes
    model.fc = nn.Linear(model.fc.in_features, output_shape)
    
    # if to_hidden:
    #     model.fc = nn.Sequential(
    #         nn.Linear(model.fc.in_features, 512),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(0.3),
    #         nn.Linear(512, output_shape)
    #     )

    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.conv1.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model