import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import copy
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR

class EMA:
    """Exponential Moving Average for target network update"""
    def __init__(self, beta=0.996):  # Increased from 0.99 for slower, more stable updates
        self.beta = beta
        
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, target_network, online_network):
    """Update target network parameters using EMA"""
    for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
        old_weight, new_weight = target_param.data, online_param.data
        target_param.data = ema_updater.update_average(old_weight, new_weight)

class ProjectionHead(nn.Module):
    """Improved MLP projection head with 3 layers"""
    def __init__(self, input_dim, hidden_dim=4096, output_dim=512):  # Increased output dimension from 256 to 512
        super().__init__()
        self.net = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # Second layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # Final layer
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class PredictionHead(nn.Module):
    """Prediction head for Consistency"""
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TensorAugmentation(nn.Module):
    """Enhanced augmentation for tensor inputs"""
    def __init__(self, image_size=224, stronger_aug=False):
        super().__init__()
        self.image_size = image_size
        self.stronger_aug = stronger_aug
        
    def forward(self, x):
        # Input is (C, H, W) tensor
        # Apply random crop with variable scale based on stronger_aug
        scale = (0.08, 1.0) if self.stronger_aug else (0.2, 1.0)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            x, scale=scale, ratio=(0.75, 1.33))
        x = TF.resize(TF.crop(x, i, j, h, w), [self.image_size, self.image_size])
        # Augmentation at this stage is limited to a random resized crop
        # (matching the paper); flip/color-jitter/grayscale are intentionally not applied here.
        return x

class OpenWorldEncoder(nn.Module):
    def __init__(self, model_path=None, image_size=224, backbone="resnet50", 
                 radius_threshold_l2=0.5, radius_weight_l2=0.1,
                 center_distance_threshold_l2=1.0, center_distance_weight_l2=0.1):
        super().__init__()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # L2 distance based regularization parameters
        self.radius_threshold_l2 = radius_threshold_l2  # Maximum allowed L2 distance within class
        self.radius_weight_l2 = radius_weight_l2  # Weight for radius L2 loss
        self.center_distance_threshold_l2 = center_distance_threshold_l2  # Minimum L2 distance between class centers
        self.center_distance_weight_l2 = center_distance_weight_l2  # Weight for center distance L2 loss
        
        # Initialize backbone model
        self.backbone_name = backbone
        
        if self.backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.backbone_name == "resnet101":
            self.backbone = models.resnet101(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.backbone_name == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(pretrained=True)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from 'resnet50', 'resnet101', 'efficientnet_b2'")
        
        # Output dimensions for projection and prediction
        self.projection_dim = 512  # Increased from 256
        
        # Create online encoder components with improved heads
        self.online_backbone = self.backbone
        self.online_projector = ProjectionHead(self.feature_dim, hidden_dim=4096, output_dim=self.projection_dim)
        self.online_predictor = PredictionHead(self.projection_dim, hidden_dim=1024, output_dim=self.projection_dim)
        
        # Create target encoder components (no gradient)
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Stop gradient for target network
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        # Moving average updater with higher momentum
        self.ema_updater = EMA(beta=0.996)
        
        # Move to device
        self.to(self.device)
        
        # Tensor augmentation for when input is already a tensor
        self.tensor_augmentation1 = TensorAugmentation(image_size, stronger_aug=False)
        self.tensor_augmentation2 = TensorAugmentation(image_size, stronger_aug=True)  # Stronger for second view
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                print(f"Loaded model from {model_path}")
                self.load_state_dict(state_dict['model_state_dict'], strict=False)
            else:
                print(f"Loaded model from {model_path}")
                self.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from {model_path}")
    
    def _compute_loss(self, online, target, normalize=True):
        """
        Classic Consistency loss function - mean squared error between normalized vectors
        
        Args:
            online: Online network predictions
            target: Target network projections  
            normalize: Whether to normalize the inputs (default: True)
        """
        if normalize:
            online = F.normalize(online, dim=1, p=2)
            target = F.normalize(target, dim=1, p=2)
        
        # Compute MSE loss between normalized vectors (classic Consistency loss)
        loss = 2 - 2 * (online * target).sum(dim=1)
        
        return loss.mean()
    
    def update_target_network(self):
        """Update target network with EMA"""
        update_moving_average(self.ema_updater, self.target_backbone, self.online_backbone)
        update_moving_average(self.ema_updater, self.target_projector, self.online_projector)
    
    def _compute_class_centers(self, features, labels):
        """
        Compute pseudo-centers for each class in the batch
        
        Args:
            features: Feature vectors [B, D]
            labels: Class labels [B]
            
        Returns:
            centers: Dictionary mapping class labels to their centers
            class_indices: Dictionary mapping class labels to indices of samples
        """
        unique_labels = torch.unique(labels)
        centers = {}
        class_indices = {}
        
        for label in unique_labels:
            mask = (labels == label)
            class_features = features[mask]
            centers[label.item()] = class_features.mean(dim=0)
            class_indices[label.item()] = torch.where(mask)[0]
            
        return centers, class_indices
    
    def _compute_radius_loss_l2(self, features, labels):
        """
        Compute radius regularization loss using L2 distance to enforce class compactness
        
        Args:
            features: Feature vectors [B, D]
            labels: Class labels [B]
            
        Returns:
            loss: Radius regularization L2 loss
        """
        # Compute class centers and indices
        centers, class_indices = self._compute_class_centers(features, labels)
        
        total_loss = 0.0
        total_samples = 0
        
        # Compute radius loss for each class
        for label, center in centers.items():
            class_features = features[class_indices[label]]
            
            # Compute L2 distances between class features and center
            distances = torch.norm(class_features - center.unsqueeze(0), dim=1, p=2)
            
            # Compute loss for samples exceeding the radius threshold
            radius_violations = F.relu(distances - self.radius_threshold_l2)
            class_loss = radius_violations.mean()
            
            # Weight the loss by class size
            total_loss += class_loss * len(class_indices[label])
            total_samples += len(class_indices[label])
        
        # Normalize by number of samples
        final_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
        
        # Add a small epsilon to prevent numerical instability
        epsilon = 1e-6
        final_loss = final_loss + epsilon
        
        return final_loss
    
    def _compute_center_distance_loss_l2(self, features, labels):
        """
        Compute loss to ensure different class centers maintain minimum L2 distance
        
        Args:
            features: Feature vectors [B, D]
            labels: Class labels [B]
            
        Returns:
            loss: Center distance regularization L2 loss
        """
        # Compute class centers
        centers, _ = self._compute_class_centers(features, labels)
        
        # Convert centers dictionary to tensor for efficient computation
        center_labels = list(centers.keys())
        if len(center_labels) < 2:  # Need at least 2 classes for distance computation
            return torch.tensor(0.0, device=self.device)
            
        center_vectors = torch.stack([centers[label] for label in center_labels])
        
        # Compute pairwise L2 distances between all centers
        num_centers = center_vectors.size(0)
        distances = torch.zeros((num_centers, num_centers), device=self.device)
        
        for i in range(num_centers):
            for j in range(i + 1, num_centers):
                dist = torch.norm(center_vectors[i] - center_vectors[j], p=2)
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Create mask for upper triangle to avoid duplicate pairs
        mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
        
        # Get distances for valid pairs
        valid_distances = distances[mask]
        
        # Compute loss for pairs that are too close (distance < threshold)
        violations = F.relu(self.center_distance_threshold_l2 - valid_distances)
        loss = violations.mean()
        
        # Add a small epsilon to prevent numerical instability
        epsilon = 1e-6
        loss = loss + epsilon
        
        return loss

    def forward(self, x1, x2=None, labels=None):
        """
        Forward pass for training using the Consistency-style approach with class-paired images,
        plus L2 radius regularization (class compactness) and L2 center-distance
        regularization (class separation).

        Args:
            x1: First batch of images (shape [B, C, H, W])
            x2: Second batch of images from same class (shape [B, C, H, W]),
                if None, uses standard Consistency with augmentations of x1
            labels: Class labels for the batch [B]

        Returns:
            tuple: (total_loss, consistency_loss, radius_loss_l2, center_distance_loss_l2)
        """
        batch_size = x1.shape[0]
        
        if x2 is None:
            # Fallback to standard Consistency with two augmentations of the same image
            x1_aug = torch.stack([self.tensor_augmentation1(img) for img in x1])
            x2_aug = torch.stack([self.tensor_augmentation2(img) for img in x1])
        else:
            # Use class-paired approach with light augmentation on both images
            x1_aug = torch.stack([self.tensor_augmentation1(img) for img in x1])
            x2_aug = torch.stack([self.tensor_augmentation1(img) for img in x2])
        
        # Online network forward passes
        online_feat1 = self.online_backbone(x1_aug)
        online_proj1 = self.online_projector(online_feat1)
        online_pred1 = self.online_predictor(online_proj1)
        
        online_feat2 = self.online_backbone(x2_aug)
        online_proj2 = self.online_projector(online_feat2)
        online_pred2 = self.online_predictor(online_proj2)
        
        # Target network forward passes (no gradients)
        with torch.no_grad():
            target_feat1 = self.target_backbone(x1_aug)
            target_proj1 = self.target_projector(target_feat1)
            
            target_feat2 = self.target_backbone(x2_aug)
            target_proj2 = self.target_projector(target_feat2)
        
        # Consistency loss (symmetric)
        loss1 = self._compute_loss(online_pred1, target_proj2.detach(), normalize=True)
        loss2 = self._compute_loss(online_pred2, target_proj1.detach(), normalize=True)
        consistency_loss = (loss1 + loss2) / 2
        
        # Initialize additional losses
        radius_loss_l2 = torch.tensor(0.0, device=self.device)
        center_distance_loss_l2 = torch.tensor(0.0, device=self.device)
        
        # Compute regularization losses if labels are provided
        if labels is not None:
            # Use online features for regularization
            combined_features = torch.cat([online_proj1, online_proj2], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)
            
            # Compute L2-based losses
            radius_loss_l2 = self._compute_radius_loss_l2(combined_features, combined_labels)
            center_distance_loss_l2 = self._compute_center_distance_loss_l2(combined_features, combined_labels)
        
        # Combine all losses
        total_loss = (consistency_loss + 
                     self.radius_weight_l2 * radius_loss_l2 + 
                     self.center_distance_weight_l2 * center_distance_loss_l2)

        return total_loss, consistency_loss, radius_loss_l2, center_distance_loss_l2
    

    