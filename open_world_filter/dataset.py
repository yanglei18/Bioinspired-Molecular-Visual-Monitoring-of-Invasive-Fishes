import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from typing import List, Dict, Set
import math


inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to even larger size for more aggressive cropping
    # transforms.RandomCrop(224),  # Random crop to target size
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomRotation(degrees=20),  # Increased rotation range
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Increased color augmentation
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # More aggressive affine transformation
    # transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # Added perspective transformation
    # transforms.RandomGrayscale(p=0.1),  # Occasional grayscale
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def remove_background(img):
    """Remove background using the alpha/mask channel.

    For 4-channel images (e.g. mask-carrying webp), the 4th channel is used as a
    foreground mask; all other images are returned as plain RGB unchanged.
    """
    if img.mode == "RGBA" or len(img.getbands()) == 4:
        try:
            bands = img.split()
            if len(bands) == 4:
                mask = bands[3]
                rgb_masked = Image.new("RGB", img.size, (0, 0, 0))
                rgb_masked.paste(img.convert("RGB"), mask=mask)
                return rgb_masked
            return img.convert("RGB")
        except Exception as e:  # pragma: no cover - defensive, matches prior behavior
            print(f"Error processing image with mask: {e}")
            return img.convert("RGB")
    return img.convert("RGB")


class FishDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=224, return_paths=False, use_grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the fish images organized in class folders
            transform (callable, optional): Optional transform to be applied on samples
            image_size (int): Size to resize images to
            return_paths (bool): Whether to return image paths along with images
            use_grayscale (bool): Whether to convert images to grayscale
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.return_paths = return_paths
        self.use_grayscale = use_grayscale
        self.classes = []
        self.samples = []
        
        # If no transform provided, create a default one
        if self.transform is None:
            '''
            transforms_list = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
            # Add either grayscale conversion or color jitter based on flag
            if use_grayscale:
                transforms_list.append(transforms.Grayscale(num_output_channels=3))
            else:
                transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            '''
            transforms_list = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        
            self.transform = transforms.Compose(transforms_list)
        
        # Load all image paths and their labels
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.classes.append(class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx, class_name))
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]
        
        # Load the source image
        src_img = Image.open(img_path).convert('RGB')
        
        # Transform the source image
        if self.transform:
            src_tensor = self.transform(src_img)
        else:
            src_tensor = transforms.ToTensor()(src_img)
        

        # Return image and class index
        if self.return_paths:
            return src_tensor, class_idx, img_path
        else:
            return src_tensor, class_idx
    
    @staticmethod
    def get_datasets(train_dir, val_dir=None, image_size=224, return_paths=False, use_grayscale=False):
        """
        Creates train and validation datasets (for distributed training)
        
        Args:
            train_dir: Directory with training data
            val_dir: Directory with validation data, if None, returns None for val_dataset
            image_size: Size to resize images to
            return_paths: Whether to return image paths
            use_grayscale: Whether to convert images to grayscale
            
        Returns:
            train_dataset, val_dataset (None if val_dir is None)
        """
        # Create training dataset
        train_dataset = FishDataset(
            root_dir=train_dir,
            image_size=image_size,
            return_paths=return_paths,
            use_grayscale=use_grayscale
        )
        
        # Create validation dataset if val_dir provided
        val_dataset = None
        if val_dir and os.path.exists(val_dir):
            val_dataset = FishDataset(
                root_dir=val_dir,
                image_size=image_size,
                return_paths=return_paths,
                use_grayscale=use_grayscale
            )
        
        return train_dataset, val_dataset

class ClassPairFishDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=224, return_paths=False, use_grayscale=False, samples_per_class=None):
        """
        Dataset that returns pairs of different images from the same class.
        
        Args:
            root_dir (string): Directory with all the fish images organized in class folders
            transform (callable, optional): Optional transform to be applied on samples
            image_size (int): Size to resize images to
            return_paths (bool): Whether to return image paths along with images
            use_grayscale (bool): Whether to convert images to grayscale
            samples_per_class (int, optional): Maximum number of samples to use per class.
                                             If None, use all available samples.
                                             If specified, randomly select this many samples per class.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.return_paths = return_paths
        self.use_grayscale = use_grayscale
        self.samples_per_class = samples_per_class
        self.classes = []
        self.samples = []
        self.class_to_indices = defaultdict(list)
        
        # Create class mapping
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # If no transform provided, create a default one
        if self.transform is None:
            transforms_list = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
            
            # Add either grayscale conversion or color jitter based on flag
            if use_grayscale:
                transforms_list.append(transforms.Grayscale(num_output_channels=3))
            else:
                transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            
            transforms_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.transform = transforms.Compose(transforms_list)
        
        # First pass: collect all images per class
        class_images = defaultdict(list)
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.classes.append(class_name)
                self.class_to_idx[class_name] = class_idx
                self.idx_to_class[class_idx] = class_name
                
                # Collect all valid images for this class
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_path = os.path.join(class_dir, img_name)
                        class_images[class_idx].append((img_path, class_idx, class_name))
        
        # Second pass: sample images and build final dataset
        total_selected = 0
        for class_idx, images in class_images.items():
            if len(images) < 2:  # Skip classes with less than 2 images
                print(f"Warning: Skipping class {self.idx_to_class[class_idx]} with only {len(images)} images")
                continue
                
            # Randomly sample if samples_per_class is specified and less than available images
            if samples_per_class and len(images) > samples_per_class:
                selected_images = random.sample(images, samples_per_class)
                print(f"Class {self.idx_to_class[class_idx]}: Selected {samples_per_class} from {len(images)} images")
            else:
                selected_images = images
                if samples_per_class:
                    print(f"Class {self.idx_to_class[class_idx]}: Using all {len(images)} images (less than requested {samples_per_class})")
                
            # Add selected images to dataset
            for img_data in selected_images:
                self.samples.append(img_data)
                self.class_to_indices[class_idx].append(len(self.samples) - 1)
                total_selected += 1
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total classes: {len(self.class_to_indices)}")
        print(f"Total selected samples: {total_selected}")
        print(f"Average samples per class: {total_selected / len(self.class_to_indices):.1f}")
        print(f"Class distribution:")
        for class_idx in sorted(self.class_to_indices.keys()):
            print(f"  {self.idx_to_class[class_idx]}: {len(self.class_to_indices[class_idx])} samples")
        print("\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]
        
        # Find another image from the same class (but not the same image)
        same_class_indices = self.class_to_indices[class_idx]
        pair_candidates = [i for i in same_class_indices if i != idx]
        
        if not pair_candidates:
            # This shouldn't happen due to our filtering, but just in case
            pair_idx = idx  # Fall back to using the same image
        else:
            pair_idx = random.choice(pair_candidates)
        
        pair_img_path, _, _ = self.samples[pair_idx]
        
        # Load images without converting to RGB to keep the mask channel
        anchor_img = Image.open(img_path)
        pair_img = Image.open(pair_img_path)
        
        # Process each image to remove background using the mask
        anchor_img = self._remove_background(anchor_img)
        pair_img = self._remove_background(pair_img)
        
        # Transform the images
        if self.transform:
            anchor_tensor = self.transform(anchor_img)
            pair_tensor = self.transform(pair_img)
        else:
            anchor_tensor = transforms.ToTensor()(anchor_img)
            pair_tensor = transforms.ToTensor()(pair_img)

        # Convert class_idx to tensor for radius regularization
        class_idx_tensor = torch.tensor(class_idx, dtype=torch.long)

        # Return image pair and class index
        if self.return_paths:
            return anchor_tensor, pair_tensor, class_idx_tensor, img_path, pair_img_path
        else:
            return anchor_tensor, pair_tensor, class_idx_tensor
    
    def _remove_background(self, img):
        return remove_background(img)
    
    @staticmethod
    def get_datasets(train_dir, val_dir=None, image_size=224, return_paths=False,
                    use_grayscale=False, samples_per_class=None):
        """
        Creates train and validation datasets with class-paired samples (for distributed training)
        
        Args:
            train_dir: Directory with training data
            val_dir: Directory with validation data, if None, returns None for val_dataset
            image_size: Size to resize images to
            return_paths: Whether to return image paths
            use_grayscale: Whether to convert images to grayscale
            samples_per_class: Maximum number of samples to use per class
            
        Returns:
            train_dataset, val_dataset (None if val_dir is None)
        """
        # Create training dataset
        train_dataset = ClassPairFishDataset(
            root_dir=train_dir,
            image_size=image_size,
            return_paths=return_paths,
            use_grayscale=use_grayscale,
            samples_per_class=samples_per_class
        )
        
        # Create validation dataset if val_dir provided
        val_dataset = None
        if val_dir and os.path.exists(val_dir):
            val_dataset = ClassPairFishDataset(
                root_dir=val_dir,
                image_size=image_size,
                return_paths=return_paths,
                use_grayscale=use_grayscale,
                samples_per_class=samples_per_class  # Use same limit for validation
            )
        
        return train_dataset, val_dataset 

class InferenceDataset(Dataset):
    """Dataset for batch inference with improved data handling"""

    def __init__(self, data_dir, transform=inference_transform, max_samples_per_class=None):
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.samples = []
        self.true_labels = []
        self.class_counts = defaultdict(int)

        # Default transform if none provided
        if self.transform is None:
            self.transform = inference_transform

        # Find all images and their corresponding class labels
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                # Get all image files for this class
                img_files = [img_name for img_name in os.listdir(class_dir)
                             if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

                # Apply limit if specified
                if self.max_samples_per_class is not None:
                    img_files = img_files[:self.max_samples_per_class]

                # Add samples
                for img_name in img_files:
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_name))
                    self.true_labels.append(class_name)
                    self.class_counts[class_name] += 1

        self.classes = sorted(list(self.class_counts.keys()))
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")

        # Print class distribution
        for class_name, count in sorted(self.class_counts.items()):
            print(f"  {class_name}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]

        try:
            # Load image without converting to RGB first to preserve alpha channel
            img = Image.open(img_path)

            # Process background removal for 4-channel images (maintain consistent with training)
            img = self._remove_background(img)
            # os.makedirs("test_images", exist_ok=True)
            # img.save(f"test_images/{class_name}.png")
            if self.transform:
                img_tensor = self.transform(img)

            return {
                'image': img_tensor,
                'path': img_path,
                'true_class': class_name
            }
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return zeros as a placeholder
            return {
                'image': torch.zeros(3, 224, 224),
                'path': img_path,
                'true_class': class_name
            }

    def _remove_background(self, img):
        return remove_background(img)
