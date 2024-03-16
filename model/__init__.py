from .alexnet import *
from .alexnet_minimal import *


def get_model(
    name='alexnet',
    num_classes=10,
    activation='relu'
):
    if name == 'alexnet':
        return AlexNet(num_classes=num_classes, activation=activation)
    elif name =='alexnet_minimal':
        return AlexNet_Minimal(num_classes=num_classes, activation=activation)
    else:
        raise ValueError("Only support options: alexnet")