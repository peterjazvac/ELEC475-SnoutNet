import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Cannot identify image file {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))  # Skip to the next image
        label = eval(self.img_labels.iloc[idx, 1])  # Ground truth label

        # Get original image size
        original_size = image.size  # (width, height)

        image = self.transform(image)

        # Adjust label coordinates according to the new image size
        new_size = (227, 227)
        label = (label[0] * new_size[0] / original_size[0], label[1] * new_size[1] / original_size[1])

        if self.target_transform:
            label = self.target_transform(label)
        
        return image, torch.tensor(label, dtype=torch.float32)

# Initialize dataset and dataloader with resizing transformation
images_dir = '/Users/peterjazvac/Desktop/ELEC475Lab2/oxford-iiit-pet-noses/images-original/images'
train_labels_file = '/Users/peterjazvac/Desktop/ELEC475Lab2/oxford-iiit-pet-noses/train_noses.txt'
test_labels_file = '/Users/peterjazvac/Desktop/ELEC475Lab2/oxford-iiit-pet-noses/test_noses.txt'

batch_size = 16  # Reduce batch size to 16

# Apply resizing transformation
train_dataset = CustomImageDataset(train_labels_file, images_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

test_dataset = CustomImageDataset(test_labels_file, images_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Function to visualize some training images and their labels
def visualize_data(loader, num_images=5):
    for i, (images, labels) in enumerate(loader):
        if i >= num_images:
            break
        img = images[0].numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        img = np.clip(img, 0, 1)
        label = labels[0].numpy()

        # Debug statement to print the ground truth label
        print(f"Ground Truth: {label}")

        plt.imshow(img)
        plt.scatter([label[0]], [label[1]], c='g', label='Ground Truth')
        plt.legend()
        plt.show()

# Visualize some training images
visualize_data(test_loader, num_images=5)