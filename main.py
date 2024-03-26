import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import yaml
import argparse
import wandb
import pprint

from utils import *
from dataloader import *
from model import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
#### 0. SETUP CONFIGURATION
################################
best_acc = 0

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--experiment', type=str, help='path to YAML config file')
args = parser.parse_args()

yaml_filepath = os.path.join(".", "config", f"{args.experiment}.yaml")
with open(yaml_filepath, "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.Loader)
    pprint.pprint(cfg)

EPOCHS = cfg['trainer']['epochs']

wandb.init(
    project=cfg['dataset']['data_name'], 
    name="config_vgg", 
)
log_dict = {}
test_dict = {}

################################
#### 1. BUILD THE DATASET
################################
train_dataloader, val_dataloader, test_dataloader, classes = get_dataloader(**cfg['dataset'])
try:
    num_classes = len(classes)
except:
    num_classes = classes
################################
#### 2. BUILD THE NEURAL NETWORK
################################
model = get_model(
    **cfg['model'],
    num_classes=num_classes,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), 
    **cfg['optimizer']
)

################################
#### 3.b Training 
################################
if __name__ == '__main__':
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_one_epoch(
            dataloader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=device, 
            log_dict=log_dict
        )
        
        best_acc = validation_one_epoch(
            epoch=epoch, 
            dataloader=val_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            device=device,
            current_time=current_time,
            best_acc=best_acc,
            log_dict=log_dict
        )
        
        wandb.log(log_dict)
        
    test_one_epoch(
        dataloader=test_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        device=device,
        current_time=current_time,
        log_dict=test_dict
    )
    
    wandb.log(test_dict)

    ################################
    #### 3.c Testing on each class
    ################################
    if isinstance(classes, list):
        class_correct = list(0. for _ in range(num_classes))
        class_total = list(0. for _ in range(num_classes))
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
        table_data = [{classes[i]: 100 * class_correct[i]} for i in range(num_classes)]
        table = wandb.Table(data=table_data, columns=["Class", "Accuracy"])
        wandb.log({"Test Accuracy per Class": table})
