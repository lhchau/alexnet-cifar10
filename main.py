import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

from tqdm import tqdm

from utils.train_one_epoch import *
from dataloader.get_dataloader import *
from model.alexnet import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
#### 0. SETUP CONFIGURATION
################################
best_acc = 0

cfg = {
    'batch_size': 128,
    'epochs': 200,
    'learning_rate': 1e-3,
    'lambda_l2': 0.0
}

################################
#### 1. BUILD THE DATASET
################################
train_dataloader, val_dataloader, test_dataloader, classes = get_dataloader(
    data='cifar10', 
    data_augmentation='basic', 
    batch_size=128, 
    num_workers=4
)

################################
#### 2. BUILD THE NEURAL NETWORK
################################
model = AlexNet(
    num_classes=len(classes)
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), 
    lr=cfg['learning_rate']
)

################################
#### 3.b Training 
################################
if __name__ == '__main__':
    for epoch in range(cfg['epochs']):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_one_epoch(
            dataloader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=device, 
            batch_size=cfg['batch_size']
        )
        
        best_acc = validation_one_epoch(
            epoch=epoch, 
            dataloader=val_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            device=device,
            current_time=current_time,
            best_acc=best_acc
        )
    test_one_epoch(
        dataloader=test_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        device=device,
        current_time=current_time
    )

    ################################
    #### 3.d Testing on each class
    ################################
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))