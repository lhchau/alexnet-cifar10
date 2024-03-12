import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),            # Tranh overfitting
            
            nn.Linear(9216, 4096),
            nn.ReLU(inplace = True),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.feature_extraction(x)  # (6x6x256) 
        x = self.flatten(x)             # 9216
        logit = self.classifier(x)      # num_classes, ^y

        return logit