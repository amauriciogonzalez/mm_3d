# Importing libraries
from torch.utils.data import DataLoader
from torchvision import transforms
from mm_shap_e import ObjaverseDataset, create_train_test_split, evaluate_objaverse_instance, MMShapE, evaluate_objaverse_instance_DF

import argparse
import importlib
import logging
import os
import sys
import time
import traceback

import torch.cuda

import matplotlib.pyplot as plt
from torch.utils.data import Subset





print("initializing dataset")
image_dir = "/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/Objaverse_Dataset/rendered_images/"
caption_csv = "/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/objaverse_csv.csv"
latent_code_dir = "/lustre/fs1/home/cap6411.student3/final_project/Cap3D/Cap3D_latentcodes/"
transform = transforms.ToTensor()

dataset = ObjaverseDataset(
  image_dir=image_dir,
  caption_csv=caption_csv,
  latent_code_dir=latent_code_dir,
  transform=transform
)

# specific_object_ids to do qualitative evaluation on
specific_object_ids = [
    'a354973dd4104388ad28d4c0da7c2f2b',
    '1e8b06bde8584f069cc0a26a27534da8',
    'f266dd57a0fb46f4a6877abdad2bbe86',
    '392dcf37195e43948cfbffe099082108',
    '4c5e1845990045e0a5ad238bafbe353c',
    '4bd4fc0e630b47d798314ecff829e0b9',
    '80c272f29b0f43189316ede9cb3f148d',
    'cc7bf5c7a8b647c7b8f10701c4cd620f',
    '29e245bc655942878879b93e658440e7',
    ]

# idxs_loader is a dataloader for the specific_object_ids

obj_ids = list(dataset.valid_object_ids)
idxs = [i for i, obj_id in enumerate(obj_ids) if obj_id in specific_object_ids]

idxs_subset = Subset(dataset, idxs[5:])
idxs_loader = DataLoader(idxs_subset, batch_size=1, shuffle=True, pin_memory=True)

print("initialized loaders")


print("Initialiing model")
model = MMShapE().to("cuda")
print("Evaluting model")
# Generate gifs for the specific_object_ids using DreamFusion model
if __name__ == '__main__':
  print(evaluate_objaverse_instance_DF(model, idxs_loader, evaluate=True))
