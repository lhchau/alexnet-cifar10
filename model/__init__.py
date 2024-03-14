from .alexnet import *


def get_model(
    name='alexnet',
    num_classes=10,
    activation='relu'
):
    if name == 'alexnet':
        return AlexNet(num_classes=num_classes, activation=activation)
    else:
        raise ValueError("Only support options: alexnet")