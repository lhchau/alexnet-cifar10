from .alexnet import *
from .minimal_alexnet import *
from .chaunet import ChauNet
from .vgg import *


def get_model(
    model_name='alexnet',
    num_classes=10,
    activation='relu',
    dropout=True
):
    if model_name == 'alexnet':
        return AlexNet(num_classes=num_classes, activation=activation, dropout=dropout)
    elif model_name == 'minimal_alexnet':
        return Minimal_Alexnet(num_classes=num_classes, activation=activation, dropout=dropout)
    elif model_name == 'chaunet':
        return ChauNet(num_classes=num_classes, activation=activation, dropout=dropout)
    elif model_name == 'vgg11':
        return vgg11_bn(num_classes=num_classes, activation=activation)
    elif model_name == 'vgg13':
        return vgg13_bn(num_classes=num_classes, activation=activation)
    elif model_name == 'vgg16':
        return vgg16_bn(num_classes=num_classes, activation=activation)
    elif model_name == 'vgg19':
        return vgg19_bn(num_classes=num_classes, activation=activation)
    elif model_name == 'vgg23':
        return vgg23_bn(num_classes=num_classes, activation=activation)
    else:
        raise ValueError("Only support options: alexnet")