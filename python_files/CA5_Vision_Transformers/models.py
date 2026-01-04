# models.py - Model definitions

import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Modify the final layer for Re-ID (classification)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

class MHSA(nn.Module):
    """ Multi-Head Self-Attention for 2D Images """
    def __init__(self, n_dims, width, height, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        
        # 1. Projections
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        # 2. Content-Content Attention
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        
        # 3. Positional Embeddings (Relative) -> Simplified for this assignment
        # In a full BotNet, you add relative position encodings here. 
        # For simplicity, we calculate basic attention:
        energy = content_content 
        
        attention = self.softmax(energy) # Save this for visualization later!
        self.last_attention_map = attention # Hook for Q1.5

        # 4. Aggregation
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out

class BotNet50(nn.Module):
    def __init__(self, num_classes, resolution=(256, 128)):
        super(BotNet50, self).__init__()
        # Load backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Extract initial layers
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        # Replace Layer 4 Convolutions with MHSA blocks
        # Note: In a real BotNet, you replace the 3x3 conv inside the Bottleneck with MHSA.
        # Ideally, iterate through resnet.layer4 and replace conv2 with MHSA.
        self.layer4 = resnet.layer4 
        
        # Example: Replacing the final spatial processing with a global MHSA before pooling
        # (Simplified version for assignment feasibility)
        # H, W at stage 4 for 256x128 input is usually 16x8
        self.mhsa = MHSA(2048, width=resolution[1]//32, height=resolution[0]//32)
        
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 
        
        # Apply Attention
        x = self.mhsa(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x