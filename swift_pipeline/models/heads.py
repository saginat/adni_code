import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Classification head for SwiFT encoder output"""

    def __init__(self, num_classes=2, num_features=288):
        super().__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.norm = nn.LayerNorm(num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(num_features, num_outputs)

    def forward(self, x):
        # x -> (b, C, D, H, W, T)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class RegressionHead(nn.Module):
    """Regression head for SwiFT encoder output"""

    def __init__(self, num_features=288):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(num_features, 1)

    def forward(self, x):
        # x -> (b, C, D, H, W, T)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class ContrastiveHead(nn.Module):
    """Contrastive learning head for SwiFT pretraining"""

    def __init__(self, num_features=288, embedding_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, embedding_dim),
        )

    def forward(self, x):
        # x -> (b, C, D, H, W, T)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.projection(x)
        return x


class ContrastiveHeadSwiFT(nn.Module):
    """
    Exact SwiFT contrastive head matching emb_mlp.py
    
    Architecture:
    - AdaptiveAvgPool1d to pool temporal dimension
    - Linear(num_features, embedding_dim, bias=False)
    - BatchNorm1d(embedding_dim)
    - L2 normalize output
    
    Reference: SwiFT/project/module/models/emb_mlp.py
    """

    def __init__(self, num_features=288, embedding_dim=128):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        # Temporal pooling
        self.temp_avg = nn.AdaptiveAvgPool1d(1)
        
        # Projection: Linear (no bias) + BatchNorm
        self.fc1 = nn.Linear(num_features, embedding_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        """
        Forward pass matching SwiFT's emb_mlp exactly.
        
        Args:
            x: Encoder output of shape (B, C, D, H, W, T)
               For 96x96x96 with 4 stages: (B, 288, 2, 2, 2, 20)
               
        Returns:
            L2-normalized embeddings of shape (B, embedding_dim)
        """
        # x -> (B, C, D, H, W, T)
        # Flatten spatial dims: (B, C, D*H*W*T)
        x = x.flatten(start_dim=2).transpose(1, 2)  # (B, L, C) where L = D*H*W*T
        
        # Global temporal average pooling
        x = self.temp_avg(x.transpose(1, 2))  # (B, C, 1)
        x = x.flatten(1)  # (B, C)
        
        # Project and normalize (matching SwiFT exactly)
        x = self.fc1(x)  # (B, embedding_dim)
        x = self.bn1(x)  # (B, embedding_dim)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        
        return x

