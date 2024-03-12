from .alexnet import *

def get_model(
    name='alexnet',
    num_classes=10
):
    if name == 'alexnet':
        return AlexNet(num_classes=num_classes)
    else:
        raise ValueError("Only support options: alexnet")