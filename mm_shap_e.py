import sys
import os

sys.path.insert(0, "./shap_e")

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh, gif_widget, decode_latent_mesh
from shap_e.util.collections import AttrDict
from shap_e.util.io import buffered_writer
from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
from shap_e.models.transmitter.base import Transmitter, VectorDecoder
import imageio
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import argparse
from rembg import remove
from sklearn.model_selection import train_test_split
import pandas as pd
import nrrd  # For reading .nrrd files
import trimesh # To export meshes
from skimage import measure
import nvidia_smi
import time
from typing import Union
from pprint import pprint
import matplotlib.pyplot as plt
import struct
from typing import BinaryIO, Optional


WEIGHTS_DIR = "./weights/"

MODEL_PATHS = {
    2: WEIGHTS_DIR+"mm_shap_e_crossmodal_attn.pth"
}


# ========================================= Model Layers for Multimodal Fusion =========================================

class CrossModalAttention(nn.Module):
    def __init__(self, latent_dim, reduced_dim=512, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.latent_dim = latent_dim
        self.reduced_dim = reduced_dim
        self.num_heads = num_heads
        
        # Dimensionality reduction
        self.reduce_dim = nn.Linear(latent_dim, reduced_dim)
        
        # Attention layers
        self.query = nn.Linear(reduced_dim, reduced_dim)
        self.key = nn.Linear(reduced_dim, reduced_dim)
        self.value = nn.Linear(reduced_dim, reduced_dim)
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=reduced_dim, num_heads=num_heads)
        
        # Project back to original latent dimension
        self.expand_dim = nn.Linear(reduced_dim, latent_dim)
        
    def forward(self, image_latents, text_latents):
        # image_latents and text_latents should have shape [batch_size, seq_len, latent_dim]

        #print("image latent shapes: ", image_latents.shape)
        #print("text latent shapes: ", text_latents.shape)
        
        # Reduce dimensionality
        image_latents_reduced = self.reduce_dim(image_latents)
        text_latents_reduced = self.reduce_dim(text_latents)

        #print("image latent reduced shapes: ", image_latents_reduced.shape)
        #print("text latent reduced shapes: ", text_latents_reduced.shape)
        
        # MultiheadAttention input requirements: [batch_size, seq_len, latent_dim]
        query = self.query(image_latents_reduced)
        key = self.key(text_latents_reduced)
        value = self.value(text_latents_reduced)

        #print("query: ", query.shape)
        #print("key: ", key.shape)
        #print("value: ", value.shape)
        
        # Cross-modal attention: image queries text
        attn_output, _ = self.multihead_attn(query, key, value)
        
        # Transpose back to [batch_size, seq_len, reduced_dim]
        attn_output = attn_output
        
        # Project back to original latent dimension
        attn_output_expanded = self.expand_dim(attn_output)
        
        return attn_output_expanded


# ========================================= Multimodal Shap-E Pipeline =========================================

class MMShapE(nn.Module):
    def __init__(self, fusion_mode=2, latent_dim=1048576, reduced_dim=512, num_heads=8, use_transmitter=True, output_path='./output'):
        super(MMShapE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fusion_mode = fusion_mode
        
        if use_transmitter:
            # Transmitter - the encoder and corresponding projection layers for converting encoder outputs into implicit neural representations.
            self.xm = load_model('transmitter', device=self.device)
        else:
            # decoder - just the final projection layer component of transmitter. This is a smaller checkpoint than transmitter since it does not
            # include parameters for encoding 3D assets. This is the minimum required model to convert diffusion outputs into implicit neural representations.
            self.xm = load_model('decoder', device=self.device)
        
        # image300M - the image-conditional latent diffusion model.
        self.image_model = load_model('image300M', device=self.device)
        
        # text300M - the text-conditional latent diffusion model.
        self.text_model = load_model('text300M', device=self.device)
        
        self.diffusion = diffusion_from_config(load_config('diffusion'))
        self.d_latent_dim = latent_dim
        self.d_reduced_dim = reduced_dim

        self._batch_size = 0        # Changes dynamically based on input
        self.text_guidance_scale = 15.0
        self.image_guidance_scale = 3.0

        self.output_path = output_path
        self.gif_path = f'{self.output_path}/gifs'
        self.ply_path = f'{self.output_path}/ply_meshes'
        self.obj_path = f'{self.output_path}/obj_meshes'
        self._create_directories()
        
        # Initialize cross-modal attention with reduced latent size
        self.cross_modal_attention = CrossModalAttention(latent_dim=self.d_latent_dim, reduced_dim=self.d_reduced_dim, num_heads=num_heads)
        
    def _create_directories(self):
        Path(f"{self.output_path}").mkdir(parents=True, exist_ok=True)
        Path(f"{self.gif_path}").mkdir(parents=True, exist_ok=True)
        Path(f"{self.ply_path}").mkdir(parents=True, exist_ok=True)
        Path(f"{self.obj_path}").mkdir(parents=True, exist_ok=True)
    
    def decode_and_display_latent_images_as_gif(self, latents, prompts=[], save_gif=False, display_gif = False, gif_fps=10, render_mode="nerf", size_of_renders=64):
        if len(prompts) == 0:
            file_names = [f"3d_representation_{i}" for i in range(len(latents))]
        else:
            file_names = [prompt.replace(" ", "_") for prompt in prompts]

        cameras = create_pan_cameras(size_of_renders, self.device)
        
        #print("Cameras: ", cameras)
        #print("Cameras shape: ", cameras.shape)
        
        for i, latent in enumerate(latents):
            images = decode_latent_images(self.xm, latent, cameras, rendering_mode=render_mode)

            #print(f"Images for latent {i}: ", images)
            #print(f"Number of images for latent {i}: ", len(images))

            # Save as a gif using imageio
            if save_gif:
                output_gif_path = f"{self.gif_path}/{file_names[i]}"
                gif_file_path = f"{output_gif_path}.gif"  # Save each latent as a separate gif
                imageio.mimsave(gif_file_path, images, fps=gif_fps)  # You can adjust the FPS
                print(f"Saved GIF: {gif_file_path}")

            if display_gif:
                display(gif_widget(images))

    def save_latents_as_meshes(self, latents, prompts=[], save_ply=False, save_obj=False):
        if len(prompts) == 0:
            file_names = [f"3d_representation_{i}" for i in range(len(latents))]
        else:
            file_names = [prompt.replace(" ", "_") for prompt in prompts]

        for i, latent in enumerate(latents):
            t = decode_latent_mesh(self.xm, latent).tri_mesh()
            if save_ply:
                with open(f'{self.ply_path}/{file_names[i]}.ply', 'wb') as f:
                    t.write_ply(f)
            if save_obj:
                with open(f'{self.obj_path}/{file_names[i]}.obj', 'w') as f:
                    t.write_obj(f)

    def decode_display_save(self, latents, prompts=[], save_gif=False, display_gif=False, gif_fps=10, save_ply=False, save_obj=False, render_mode="nerf", size_of_renders=64):
        # Prompts are used as file names
        self.decode_and_display_latent_images_as_gif(latents, prompts, render_mode, display_gif, gif_fps, render_mode, size_of_renders)
        self.save_latents_as_meshes(latents, prompts, save_ply, save_obj)

    def decode_latent_images_grad(
        self,
        xm: Union[Transmitter, VectorDecoder],
        latent: torch.Tensor,
        cameras: DifferentiableCameraBatch,
        rendering_mode: str = "stf",
    ):
        # Ensure that all operations inside this function are differentiable
        decoded = xm.renderer.render_views(
            AttrDict(cameras=cameras),
            params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
                latent[None]
            ),
            options=AttrDict(rendering_mode=rendering_mode, render_with_direction=False),
        )

        # Keep everything as PyTorch tensors.
        # Normalize the tensor values if needed (e.g., clamp between 0 and 1 or 0 and 255)
        images = decoded.channels.clamp(0, 1)  # Assuming we want values between 0 and 1

        # Ensure images remain tensors for further operations
        return images  # This returns a PyTorch tensor instead of PIL images

    def _create_orthographic_cameras(self, size: int, device: torch.device) -> DifferentiableCameraBatch:
        """
        Create 3 orthographic cameras aligned to the XY, YZ, and XZ planes for projection.
        """
        # Define camera positions and directions for XY, YZ, and XZ projections
        origins = [
            np.array([0, 0, 4]),  # XY view (camera looking along -Z)
            np.array([0, 4, 0]),  # XZ view (camera looking along -Y)
            np.array([4, 0, 0]),  # YZ view (camera looking along -X)
        ]
        zs = [
            np.array([0, 0, -1]),  # XY view (facing negative Z)
            np.array([0, -1, 0]),  # XZ view (facing negative Y)
            np.array([-1, 0, 0]),  # YZ view (facing negative X)
        ]
        xs = [
            np.array([1, 0, 0]),   # XY view (X-axis as is)
            np.array([1, 0, 0]),   # XZ view (X-axis as is)
            np.array([0, 1, 0]),   # YZ view (Y-axis as is)
        ]
        ys = [
            np.array([0, 1, 0]),   # XY view (Y-axis as is)
            np.array([0, 0, 1]),   # XZ view (Z-axis as is)
            np.array([0, 0, 1]),   # YZ view (Z-axis as is)
        ]

        return DifferentiableCameraBatch(
            shape=(1, len(xs)),
            flat_camera=DifferentiableProjectiveCamera(
                origin=torch.from_numpy(np.stack(origins, axis=0)).float().to(device),
                x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
                y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
                z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
                width=size,
                height=size,
                x_fov=0.7,  # FOV may be tweaked to ensure good orthographic views
                y_fov=0.7,
            ),
        )


    @torch.no_grad()
    # Returns a lists of PIL images
    def generate_orthographic_projections(self, latents, size_of_renders=64, render_mode="nerf"):
        orthographic_cameras = self._create_orthographic_cameras(size_of_renders, self.device)
        projection_lists = []
        for latent in latents:
            images = decode_latent_images(self.xm, latent, orthographic_cameras, rendering_mode=render_mode)
            images = [image.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for image in images]
            projection_lists.append(images)
        return projection_lists

    # Returns a tensor for training
    def generate_orthographic_projections_grad(self, latents, size_of_renders=64, render_mode="nerf"):
        orthographic_cameras = self._create_orthographic_cameras(size_of_renders, self.device)

        # List to hold the projections as tensors
        projection_tensors = []
        
        for latent in latents:
            # Generate projections as tensors using the updated decode_latent_images function
            images = self.decode_latent_images_grad(self.xm, latent, orthographic_cameras, rendering_mode=render_mode)
            
            # Flip the image, and clone to avoid in-place issues
            flipped_images = torch.flip(images, dims=[-2])  # Flip along the vertical axis (height)
            
            # Append the tensor (flipped or not) to the list
            projection_tensors.append(flipped_images)
        
        # Stack projections along the batch dimension
        projections_stacked = torch.stack(projection_tensors)

        # Remove the singleton dimension if necessary (e.g., the second dimension)
        projections_squeezed = projections_stacked.squeeze(1)  # Assuming you have [batch_size, 1, ...]

        # Permute to match the required shape: [batch_size, num_views, num_channels, height, width]
        projections_permuted_correctly = projections_squeezed.permute(0, 1, 4, 2, 3)  # Adjusting the dimensions

        return projections_permuted_correctly
    
    def average_fusion(self, image_latents, text_latents):
        # Simple fusion module: averaging the latent spaces
        return (image_latents + text_latents) / 2
    
    def cross_modal_fusion(self, image_latents, text_latents):
        # Use cross-modal attention instead of averaging
        return self.cross_modal_attention(image_latents, text_latents)

    def generate_latents(self, images, prompts):
        # Generate image latents
        image_latents = sample_latents(
            batch_size=self._batch_size,
            model=self.image_model,
            diffusion=self.diffusion,
            guidance_scale=self.image_guidance_scale,
            model_kwargs=dict(images=images),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        

        # Generate text latents
        text_latents = sample_latents(
            batch_size=self._batch_size,
            model=self.text_model,
            diffusion=self.diffusion,
            guidance_scale=self.text_guidance_scale,
            model_kwargs=dict(texts=prompts),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        #print("image_latents: ", image_latents)
        #print("image_latents shape: ", image_latents.shape)
        #print("text_latents: ", text_latents)
        #print("text_latents shape: ", text_latents.shape)

        # Combine latents with fusion module
        #fused_latents = self.simple_fusion(image_latents, text_latents)

        #if self.fusion_mode == 1:
        #    fused_latents = self.average_fusion(image_latents, text_latents)
        #elif self.fusion_mode == 2:
        #    fused_latents =  self.cross_modal_fusion(image_latents, text_latents)
        #else:
        #    raise ValueError(f'Invalid fusion mode: {self.fusion_mode}')
        
        #print("fused_latents: ", fused_latents)
        
        #return fused_latents, image_latents, text_latents
        return image_latents, text_latents

    def _convert_tensor_batch_to_pil_list(self, image_tensors):
        pil_images = []
        for image_tensor in image_tensors:
            # Convert the tensor image to a PIL image
            pil_image = transforms.ToPILImage()(image_tensor.cpu())  # Move the image to CPU if it's on a different device
            pil_images.append(pil_image)

        return pil_images


    # Function to take an image and remove the background of an image
    def _remove_background(self, image, display_image=False, return_tensor=False):
        # If image is not pil image, convert it from a tensor
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        # Reshape to (3, 224, 224)
        image = image.resize((224, 224))
        
        #print("Image shape: ", image.size)
        
        # Remove background
        image_background_removed = remove(image)
        if display_image:
            display(image_background_removed)
            
        # Convert to RGB if the image has an alpha channel (RGBA)
        if image_background_removed.mode == 'RGBA':
            image_background_removed = image_background_removed.convert('RGB')
        
        if return_tensor:
            # Convert image to tensor
            image_background_removed_tensor = transforms.ToTensor()(image_background_removed)
            return image_background_removed_tensor
        else:
            # Return PIL image
            return image_background_removed
        
    
    def forward(self, images, prompts, remove_background=False):
        # Check if the lengths of images and prompts match
        if len(images) != len(prompts):
            raise ValueError(f"The number of images ({len(images)}) and prompts ({len(prompts)}) must be the same.")

        # Set batch size to the length of the images (or prompts, since they are now guaranteed to be the same length)
        self._batch_size = len(images)

        #print("Image tensors: ", images)

        # Convert images to PIL format if they are tensors
        if isinstance(images[0], torch.Tensor):
            images = self._convert_tensor_batch_to_pil_list(images)

        #print("Images after converting tensors to pil: ", images)

        if remove_background:
            images = [self._remove_background(image) for image in images]

        #print("Prompts: ", prompts)

        #fused_latents, image_latents, text_latents = self.generate_latents(images, prompts)
        image_latents, text_latents = self.generate_latents(images, prompts)

        fused_latents =  self.cross_modal_attention(image_latents, text_latents)

        #self.decode_display_save(fused_latents, self.device, prompt, render_mode="nerf")
        
        return fused_latents, image_latents, text_latents


# ========================================== Datasets =========================================

# Just to test if the crossmodal fusion module could be trained
class DummyDataset(Dataset):
    def __init__(self, num_samples=100):
        super().__init__()
        self.num_samples = num_samples
        # Random pil images (e.g., 3x224x224) list and text prompts as inputs
        self.images = [torch.randn(3, 224, 224) for _ in range(num_samples)]
        self.prompts = ["This is a dummy prompt"] * num_samples  # Simple dummy prompt for all samples
        # Random target 3D latents (for simplicity, same dimensionality as output latents)
        self.targets = torch.randn(num_samples, 1, 1048576)  # Assuming latent_dim = 1048576
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.images[idx]
        prompt = self.prompts[idx]
        target = self.targets[idx]
        return image, prompt, target



# Text2Shape Dataset
class Text2ShapeDataset(Dataset):
    def __init__(self, dataset_dir, csv_path, transform=None):
        self.dataset_dir = dataset_dir
        self.csv_path = csv_path
        self.transform = transform
        
        # Load the CSV file into a DataFrame
        self.annotations = pd.read_csv(csv_path)

        # Filter annotations to keep only those with existing .nrrd files
        self.valid_annotations = self.annotations[self.annotations['modelId'].apply(self._nrrd_exists)]
        
    def _nrrd_exists(self, model_id):
        # Helper function to check if .nrrd file exists for the given model ID
        nrrd_file_path = os.path.join(self.dataset_dir, model_id, f"{model_id}.nrrd")
        return os.path.isfile(nrrd_file_path)

    def __len__(self):
        # Return the number of valid entries in the dataset
        return len(self.valid_annotations)

    def __getitem__(self, idx):
        # Get the model ID and description from the valid annotations
        model_id = self.valid_annotations.iloc[idx]['modelId']
        description = self.valid_annotations.iloc[idx]['description']
        
        # Load the .nrrd file associated with the model_id
        nrrd_file_path = os.path.join(self.dataset_dir, model_id, f"{model_id}.nrrd")
        
        # Load the 3D voxel data from the .nrrd file
        voxel_data, _ = nrrd.read(nrrd_file_path)
        
        # Apply any transforms (if provided)
        if self.transform:
            voxel_data = self.transform(voxel_data)
        
        # Convert to torch tensor (for PyTorch)
        voxel_tensor = torch.tensor(voxel_data, dtype=torch.float32)
        
        # Render the voxel projections into images (XY, YZ, XZ)
        projection_images = render_voxels_as_images(voxel_tensor)
        
        # Convert images to tensors
        for i, img in enumerate(projection_images):
            projection_images[i] = transforms.ToTensor()(img)
        
        # Convert list of images to tensor
        projection_images = torch.stack(projection_images, dim=0)
        
        # Return the 3D voxel tensor and the corresponding description
        return voxel_tensor, description, projection_images


# ========================================= Dataset Utility Functions =========================================

def create_train_test_split(dataset, test_size=0.2, batch_size=8, num_samples=0):
    """
    Create a training and test split from the dataset.

    Args:
        dataset: The dataset to split.
        test_size: Proportion of the dataset to include in the test split.
        num_samples: Number of samples to return from each split. If None, return all samples.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
    """
    # Get indices of the dataset
    indices = list(range(len(dataset)))

    # Split the dataset into training and test indices
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=42
    )

    # If num_samples is specified, take a subset of each split
    if num_samples > 0:
        train_indices = train_indices[:num_samples]
        test_indices = test_indices[:num_samples]

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader


def convert_projection_tensors_into_lists(projection_images):
    batch_size = projection_images.shape[0]
    num_views = projection_images.shape[1]

    # Initialize an empty list to hold the lists of projection images
    list_of_lists = []

    # Loop over each item in the batch
    for i in range(batch_size):
        # List to hold the projection images for the current sample
        sample_projections = []

        # Loop over each view (3 views in this case: XY, YZ, XZ projections)
        for view_idx in range(num_views):
            # Extract the projection image for the current view and sample
            projection_tensor = projection_images[i, view_idx]  # Shape: [3, 64, 64] (RGB channels)

            # Convert the tensor to a PIL image (from tensor format [C, H, W])
            projection_image_pil = transforms.ToPILImage()(projection_tensor)

            # Append the PIL image to the sample_projections list
            sample_projections.append(projection_image_pil)

        # Append the list of 3 projections for this sample to the list_of_lists
        list_of_lists.append(sample_projections)

    return list_of_lists

def pil_images_to_tensor(list_of_lists):
    """
    Convert a list of lists of PIL images to a tensor.

    :param list_of_lists: A list of lists where each inner list contains PIL images.
    :return: A tensor of shape [batch_size, num_views, channels, height, width].
    """
    # Initialize an empty list to hold all image tensors
    all_image_tensors = []

    # Initialize the tensor converter
    to_tensor = transforms.ToTensor()

    # Loop through each sample in the list of lists
    for sample_projections in list_of_lists:
        # Initialize a list to hold the projection tensors for the current sample
        projection_tensors = []

        # Loop through each PIL image in the current sample
        for image in sample_projections:
            # Convert the PIL image to a tensor
            image_tensor = to_tensor(image)  # Shape: [C, H, W]
            projection_tensors.append(image_tensor)

        # Stack the tensors for the current sample to create a tensor of shape [num_views, C, H, W]
        sample_tensor = torch.stack(projection_tensors)  # Shape: [num_views, C, H, W]

        # Append the sample tensor to the list of all image tensors
        all_image_tensors.append(sample_tensor)

    # Stack all sample tensors to create the final tensor of shape [batch_size, num_views, C, H, W]
    final_tensor = torch.stack(all_image_tensors)  # Shape: [batch_size, num_views, C, H, W]

    return final_tensor


def write_ply(
    raw_f: BinaryIO,
    coords: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
):
    """
    Write a PLY file for a mesh or a point cloud.

    :param coords: an [N x 3] array of floating point coordinates.
    :param rgb: an [N x 3] array of vertex colors, in the range [0.0, 1.0].
    :param faces: an [N x 3] array of triangles encoded as integer indices.
    """
    with buffered_writer(raw_f) as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if rgb is not None:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        if faces is not None:
            f.write(bytes(f"element face {len(faces)}\n", "ascii"))
            f.write(b"property list uchar int vertex_index\n")
        f.write(b"end_header\n")

        if rgb is not None:
            rgb = (rgb * 255.499).round().astype(int)
            vertices = [
                (*coord, *rgb)
                for coord, rgb in zip(
                    coords.tolist(),
                    rgb.tolist(),
                )
            ]
            format = struct.Struct("<3f3B")
            for item in vertices:
                f.write(format.pack(*item))
        else:
            format = struct.Struct("<3f")
            for vertex in coords.tolist():
                f.write(format.pack(*vertex))

        if faces is not None:
            format = struct.Struct("<B3I")
            for tri in faces.tolist():
                f.write(format.pack(len(tri), *tri))


def render_voxels_as_images(voxel_tensor, size_of_render=64, num_views=3):
    """
    Render the voxel grid as 2D images by projecting it onto different planes.
    
    :param voxel_tensor: A [C, D, H, W] voxel grid, where C is the number of channels (including RGB)
    :param num_views: The number of views to generate (default 3 for XY, YZ, XZ views)
    :return: A list of PIL images
    """
    # Extract occupancy grid and color channels (assuming the last channel is occupancy, rest are RGB)
    occupancy = voxel_tensor[3]  # Shape: [D, H, W]
    colors = voxel_tensor[0:3]   # Shape: [3, D, H, W]

    images = []

    # Normalize occupancy for rendering (just for visualization purposes)
    occupancy_normalized = torch.clamp(occupancy, 0, 1)
    
    # Render different projections (XY, YZ, XZ)
    for axis in range(num_views):
        # Sum along the axis to create a projection (max to highlight occupied voxels)
        projection, _ = torch.max(occupancy_normalized, dim=axis)
        
        # Handle color channels individually for the projection
        color_r, _ = torch.max(colors[0], dim=axis)  # Red channel
        color_g, _ = torch.max(colors[1], dim=axis)  # Green channel
        color_b, _ = torch.max(colors[2], dim=axis)  # Blue channel

        # Stack the RGB channels along the last dimension (H, W, 3)
        color_projection_image = torch.stack([color_r, color_g, color_b], dim=-1).cpu().numpy()

        # Convert to uint8 format for image
        color_projection_image = (color_projection_image * 255).astype(np.uint8)

        # Create a PIL image from the projection
        img = Image.fromarray(color_projection_image)
        
        # Transformations
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        img = img.resize((size_of_render, size_of_render))
        
        images.append(img)
    
    return images



def voxel_to_mesh(voxel_tensor, output_path='./voxel_mesh_sample.ply'):

    occupancy = voxel_tensor[3] 
    colors = voxel_tensor[0:3]

    occupancy_np = occupancy.cpu().numpy() 
    colors_np = colors.cpu().numpy()

    # Binarize the occupancy data if necessary (e.g., threshold at 0.5)
    occupancy_np = (occupancy_np > 0.5).astype(np.uint8)

    # Perform marching cubes to extract mesh from the occupancy grid
    vertices, faces, _, _ = measure.marching_cubes(occupancy_np)

    # Swap axes to correct orientation
    vertices = vertices[:, [2, 1, 0]]

    # Convert the vertices to integers to index the voxel grid for color (after axis swap)
    vertex_indices = vertices.astype(int)

    # Ensure the indices are within bounds
    vertex_indices = np.clip(vertex_indices, 0, np.array(colors.shape[1:]) - 1)

    # Now extract RGB colors for each vertex based on the new indices
    vertex_colors = colors[:, vertex_indices[:, 0], vertex_indices[:, 1], vertex_indices[:, 2]].T

    #vertex_colors = colors[:, [2, 1, 0]].T

    # Normalize vertex_colors to be in range [0.0, 1.0]
    vertex_colors = vertex_colors / 255.0 if vertex_colors.max() > 1 else vertex_colors
    vertex_colors = vertex_colors.cpu().numpy()


    # Open a binary file for writing
    with open(output_path, 'wb') as f:
        # Use the write_ply function to write the mesh
        write_ply(
            raw_f=f,
            coords=vertices,       # Vertex positions (N x 3)
            rgb=vertex_colors,     # Vertex colors (N x 3) normalized between 0 and 1
            faces=faces            # Faces (N x 3)
        )

    print(f"PLY file saved at {output_path}")


# ========================================= Loss Functions =========================================

# To compute perceptual similarity loss between the ground truth and predicted images, we can leverage a model pre-trained
# on image classification tasks that captures semantic features—typically a variant of the VGG network works well for this purpose.
# The perceptual loss will compare high-level feature representations of the images, not just pixel values.
# L2 loss is used to compute the difference between the feature representations of the images.
class PerceptualLoss(nn.Module):
    def __init__(self, layers: list = None, device: torch.device = torch.device('cpu')):
        super(PerceptualLoss, self).__init__()
        
        # Pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features

        # Replace in-place ReLU with out-of-place ReLU
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)
        
        self.layers = layers or [2, 7, 12, 21, 30]  # Default layers to extract features
        self.vgg = nn.Sequential(*list(vgg)[:max(self.layers)+1])
        
        # Disable gradient computation for VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg = self.vgg.to(device)
        self.device = device
        
    def forward(self, pred_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        # Ensure images are on the correct device
        pred_images = pred_images.to(self.device)
        target_images = target_images.to(self.device)

        loss = 0.0
        
        for view_idx in range(pred_images.size(1)):
            x_pred = pred_images[:, view_idx]  # Avoid in-place modification
            x_target = target_images[:, view_idx]  # Clone to prevent modification

            # Extract features from predicted and ground truth images
            for i, layer in enumerate(self.vgg):
                x_pred = layer(x_pred)
                x_target = layer(x_target)

                if i in self.layers:
                    loss += F.mse_loss(x_pred, x_target)
        
        return loss


# ========================================= Training =========================================

def train_model(model, dataloader, mode, loss_fn, optimizer, num_epochs=10, device='cuda', save_graphic_epoch=0):
    """
    Train the model using the provided dataset and parameters.
    
    Args:
        model: The model to be trained.
        dataloader: DataLoader for the dataset.
        mode: The fusion mode used to save the model weights accordingly.
        loss_fn: Loss function to optimize.
        optimizer: Optimizer for updating model parameters.
        num_epochs: Number of training epochs.
        device: Device to run the training on ('cuda' or 'cpu').
    """
    model.train()  # Set the model to training mode

    for name, param in model.named_parameters():
        if "cross_modal_attention" in name: 
            param.requires_grad = True
        else:
            param.requires_grad = False

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    print(f'{("="*20)}  TRAINING {("="*20)}')

    torch.autograd.set_detect_anomaly(True)

    # Initialize GPU memory tracking
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using the first GPU

    print(f"GPU MEMORY BEFORE TRAINING: {nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1e6} GB")

    for epoch in range(num_epochs):
        total_loss = 0.0
        max_gpu_memory = 0  # To track peak GPU memory usage

        start_time = time.time()

        for batch_idx, (sample_voxels, sample_descriptions, sample_images) in enumerate(dataloader):
            optimizer.zero_grad()

            if torch.isnan(sample_voxels).any() or torch.isnan(sample_images).any():
                print("NaN values detected in the input.")
                break

            # Move tensors to the specified device
            sample_voxels = sample_voxels.to(device)
            sample_images = sample_images.to(device)

            # Use the last projection image in the sample_images for training
            # Assuming sample_images is of shape [batch_size, num_views, c, h, w]
            sample_projection_images = sample_images[:, -1]  # Select the last projection image for each sample (XZ plane)
            
            # Forward pass
            fused_latents, image_latents, text_latents = model.forward(
                images=sample_projection_images,
                prompts=sample_descriptions,
                remove_background=False
            )

            # Generate orthographic projections from the fused latents
            predicted_image_tensors = model.generate_orthographic_projections_grad(
                fused_latents,
                size_of_renders=64,
                render_mode="nerf"
            )


            # Compute loss
            loss = loss_fn(predicted_image_tensors, sample_images)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("NaN or Inf detected in loss.")
                break

            total_loss += loss.item()

            loss.backward()
            optimizer.step()         

            # Track GPU memory usage during the process
            mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            max_gpu_memory = max(max_gpu_memory, mem_info.used)

            if batch_idx == 0 and save_graphic_epoch != 0 and ((epoch + 1) % save_graphic_epoch == 0 or epoch == 0):
                save_samples_plots(predicted_image_tensors, sample_images, sample_descriptions, epoch)
            if batch_idx == 0 and epoch == num_epochs - 1:
                voxel_to_mesh(sample_voxels[0], output_path="./voxel_mesh_sample.ply")
                fused_latents = fused_latents[0]
                model.decode_display_save(fused_latents.unsqueeze(0), save_ply=True)

            torch.cuda.empty_cache() 

        # End of epoch, calculate statistics
        end_time = time.time()
        avg_loss = total_loss / len(dataloader)
        max_gpu_memory_MB = max_gpu_memory / 1e6  # Convert to MB
        total_time = end_time - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Time: {total_time:.2f}s, Peak GPU Memory: {max_gpu_memory_MB:.2f} MB")

        
    # Save model weights at the end of training
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    torch.save(model.state_dict(), MODEL_PATHS[mode])
    print(f"Model saved at {MODEL_PATHS[mode]}.")






# Used to test whether or not weights would change after training
def train_one_epoch_dummy(model, dataloader):
    # Simple loss function
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.cross_modal_attention.parameters(), lr=1e-4)

    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, prompts, targets) in enumerate(dataloader):
        images = images.to(model.device)
        targets = targets.to(model.device)

        # Forward pass through the model
        optimizer.zero_grad()
        latents, _, _ = model(images, prompts)
        
        # Calculate loss between output latents and target latents
        loss = criterion(latents, targets)
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")
    
    return avg_loss

# Check if weights are updated via the dummy training function
def check_weight_updates(model, dataloader):
    initial_weights = torch.clone(list(model.cross_modal_attention.parameters())[0].data)
    
    # Perform one step of training
    train_one_epoch_dummy(model, dataloader)
    
    # Check if weights have changed
    current_weights = list(model.cross_modal_attention.parameters())[0].data
    weight_diff = torch.sum(torch.abs(current_weights - initial_weights)).item()
    
    if weight_diff > 0:
        print(f"Model weights updated. Total weight change: {weight_diff:.6f}")
    else:
        print("Model weights did not update.")




# ========================================= Demo and Display =========================================

def demo_from_samples(samples_dir, model, decode_fused_latents=True, decode_image_latents=False, decode_text_latents=False, save_gif=True, save_ply=False, save_obj=False):
    images = []
    file_names = []

    # Iterate over each file in the samples_dir
    for filename in os.listdir(samples_dir):
        # Construct the full file path
        file_path = os.path.join(samples_dir, filename)
        
        # Check if it is a file and ends with an image extension
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open the image and append to the images list
            image = Image.open(file_path)
            images.append(image)
            # Append the filename (without extension) to the file_names list
            file_names.append(os.path.splitext(filename)[0])

    fused_latents, image_latents, text_latents = model(images, file_names, remove_background=True)

    if decode_fused_latents:
        model.decode_display_save(fused_latents, file_names, save_gif=save_gif, display_gif=False, gif_fps=10, save_ply=save_ply, save_obj=save_obj, render_mode="nerf", size_of_renders=64)

    if decode_image_latents:
        file_names_from_image = [file_name + "_from_image" for file_name in file_names]
        model.decode_display_save(image_latents, file_names_from_image, save_gif=save_gif, display_gif=False, gif_fps=10, save_ply=save_ply, save_obj=save_obj, render_mode="nerf", size_of_renders=64)

    if decode_text_latents:
        file_names_from_text = [file_name + "_from_text" for file_name in file_names]
        model.decode_display_save(text_latents, file_names_from_text, save_gif=save_gif, display_gif=False, gif_fps=10, save_ply=save_ply, save_obj=save_obj, render_mode="nerf", size_of_renders=64)


def save_samples_plots(predicted_images, target_images, descriptions, epoch):
    """
    Save the plots of the predictions and descriptions.

    Args:
        predicted_images: The predicted orthographic projections.
        target_images: The target images from the dataset.
        descriptions: The associated text descriptions.
        epoch: The current epoch number for naming.
    """

    predicted_images = convert_projection_tensors_into_lists(predicted_images)
    target_images = convert_projection_tensors_into_lists(target_images)

    # Number of samples to plot
    num_samples = len(predicted_images)

    for i in range(num_samples):
        fig, axs = plt.subplots(2, 3, figsize=(15, 5))

        for k in range(len(target_images[i])):
            axs[0, k].imshow(predicted_images[i][k])  
            axs[0, k].axis('off')  # Hide axes
            axs[1, k].imshow(target_images[i][k])  
            axs[1, k].axis('off')  # Hide axes

        # Display the sample description in the third column

        plt.suptitle(descriptions[i], fontsize=16)
        plt.tight_layout()
        plt.savefig(f'sample_plots_{i}_epoch_{epoch}.png')
        plt.close(fig)



# ========================================= Main =========================================

def main(FLAGS):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Initializing dataset...")

    # Instantiate the dataset
    dataset_dir = "./text2shape/nrrd_256_filter_div_32/nrrd_256_filter_div_32"
    csv_dir = "./text2shape/captions.tablechair.csv"
    text2shape_dataset = Text2ShapeDataset(dataset_dir=dataset_dir, csv_path=csv_dir)
    train_loader, test_loader = create_train_test_split(text2shape_dataset, test_size=0.2, batch_size=FLAGS.batch_size, num_samples=FLAGS.n)

    # Initialize the MMShapE model
    print("Initializing MMShapE model...")

    # Fusion modes:
    #  1 -> Average fusion (not trainable),
    #  2 -> cross-modal fusion

    model = MMShapE(
        fusion_mode=FLAGS.mode,      
        latent_dim=1048576,
        reduced_dim=512,
        num_heads=8,
        use_transmitter=True,
        output_path="./output"
        ).to(device)

    print("Model: ", model)

    if FLAGS.train:
        
        #optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FLAGS.learning_rate)
        loss_fn = PerceptualLoss()
        train_model(model, train_loader, FLAGS.mode, loss_fn, optimizer, num_epochs=FLAGS.epochs, device=device, save_graphic_epoch=5)
        #dummy_dataset = DummyDataset(num_samples=10)
        #dataloader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)
        #check_weight_updates(model, dataloader)

    if FLAGS.demo:
        # Demo from samples directory. The file names will be considered the text descriptions.
        samples_dir = "./samples"

        demo_from_samples(samples_dir,
                            model,
                            decode_fused_latents=True,
                            decode_image_latents=True,
                            decode_text_latents=True,
                            save_gif=True,
                            save_ply=True,
                            save_obj=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multimodal Shap-E')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help="Determines what fusion mode to use")
    parser.add_argument('--epochs',
                        type=int, default=3,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch_size',
                        type=int, default=8,
                        help='batch size.')
    parser.add_argument('--n',
                        type=int, default=0,
                        help='Number of samples to use when training and/or evaluating. Leave at 0 for all samples.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--train',
                        action='store_true',
                        help='Run training loop.')
    parser.add_argument('--eval',
                        action='store_true',
                        help='Run evaluation loop.')
    parser.add_argument('--demo',
                        action='store_true',
                        help='Run demo')
    
    FLAGS, unparsed = parser.parse_known_args()
  
    main(FLAGS)