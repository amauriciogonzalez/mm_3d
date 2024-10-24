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
from PIL import Image, ImageEnhance
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
from skimage import measure
from scipy.ndimage import rotate
from scipy import linalg
import pandas as pd
import nrrd  # For reading .nrrd files
import trimesh # To export meshes
import nvidia_smi
import time
from typing import Union
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import struct
from typing import BinaryIO, Optional
import random
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor
import csv
import textwrap

matplotlib.use('Agg')

WEIGHTS_DIR = "./weights/"

MODEL_PATHS = {
    2: WEIGHTS_DIR+"mm_shap_e_t2s_crossmodal_attn.pth",
    3: WEIGHTS_DIR+"mm_shap_e_t2s_crossmodal_attn_seq.pth",
    4: WEIGHTS_DIR+"mm_shap_e_t2s_gated_fusion.pth",
    5: WEIGHTS_DIR+"mm_shap_e_t2s_weighted_fusion.pth",
    6: WEIGHTS_DIR+"mm_shap_e_t2s_minor_attn.pth",
    20: WEIGHTS_DIR+"mm_shap_e_ov_crossmodal_attn.pth",
    30: WEIGHTS_DIR+"mm_shap_e_ov_crossmodal_attn_seq.pth",
    40: WEIGHTS_DIR+"mm_shap_e_ov_gated_fusion.pth",
    50: WEIGHTS_DIR+"mm_shap_e_ov_weighted_fusion.pth",
    60: WEIGHTS_DIR+"mm_shap_e_ov_minor_attn.pth",
}


# ================================================================================== General Utility Functions ==================================================================================

def get_model_path(mode, dataset):
    if dataset == "text2shape":
        load_mode = mode
    elif dataset == "objaverse":
        load_mode = mode * 10
    else:
        raise ValueError(f"Invalid Dataset: '{dataset}' to load weights")
        
    return MODEL_PATHS[load_mode]



# ================================================================================== Model Layers for Multimodal Fusion ==================================================================================

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



class CrossModalAttentionSeqLevel(nn.Module):
    def __init__(self, latent_dim, token_dim=1024, num_heads=8):
        super(CrossModalAttentionSeqLevel, self).__init__()
        self.latent_dim = latent_dim
        self.token_dim = token_dim  # Size of each token (1024)
        self.num_heads = num_heads
        
        # Attention layers
        self.query = nn.Linear(token_dim, token_dim)
        self.key = nn.Linear(token_dim, token_dim)
        self.value = nn.Linear(token_dim, token_dim)
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads)
        
    def forward(self, image_latents, text_latents):
        # image_latents and text_latents should have shape [batch_size, 1, latent_dim]
        
        # Reshape latents into sequences of tokens (batch_size, 1024, 1024)
        batch_size = image_latents.size(0)
        image_latents = image_latents.view(batch_size, 1024, self.token_dim)
        text_latents = text_latents.view(batch_size, 1024, self.token_dim)
        
        # Permute for MultiheadAttention compatibility: [seq_len, batch_size, token_dim]
        image_latents = image_latents.permute(1, 0, 2)  # [1024, batch_size, 1024]
        text_latents = text_latents.permute(1, 0, 2)    # [1024, batch_size, 1024]

        # Compute query, key, value for cross-attention
        query = self.query(image_latents)
        key = self.key(text_latents)
        value = self.value(text_latents)

        # Cross-modal attention: image queries text
        attn_output, _ = self.multihead_attn(query, key, value)
        
        # Transpose back to [batch_size, 1024, token_dim]
        attn_output = attn_output.permute(1, 0, 2)
        
        # Reshape back to [batch_size, 1, latent_dim] (1048576)
        attn_output = attn_output.reshape(batch_size, 1, self.latent_dim)
        
        return attn_output


class GatedFusionModule(nn.Module):
    def __init__(self, latent_dim, reduced_dim=1024):
        super(GatedFusionModule, self).__init__()
        
        # Reduce the latent dimension to a smaller size for efficient gating
        self.reduce_dim = nn.Linear(latent_dim, reduced_dim)
        
        # MLP for generating fusion weights
        self.gate_mlp = nn.Sequential(
            nn.Linear(reduced_dim * 2, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, 2),  # Outputs two values, one for image and one for text
            nn.Softmax(dim=-1)  # Softmax ensures that weights sum to 1
        )
        
        # Expand the reduced latent back to the original dimension
        self.expand_dim = nn.Linear(reduced_dim, latent_dim)
    
    def forward(self, image_latents, text_latents):
        # Reduce dimensionality of latents
        reduced_image_latents = self.reduce_dim(image_latents)
        reduced_text_latents = self.reduce_dim(text_latents)
        
        # Concatenate the reduced latents along the last dimension
        combined_latents = torch.cat([reduced_image_latents, reduced_text_latents], dim=-1)
        
        # Generate the fusion weights
        weights = self.gate_mlp(combined_latents)  # Shape: [batch_size, 2]

        # Split the weights into image and text components
        weight_image = weights[:, 0].unsqueeze(-1)  # Shape: [batch_size, 1]
        weight_text = weights[:, 1].unsqueeze(-1)  # Shape: [batch_size, 1]

        # Apply the weights to the respective latents and sum
        fused_latents = (weight_image * reduced_image_latents) + (weight_text * reduced_text_latents)
        
        # Expand the fused latents back to the original dimension
        fused_latents = self.expand_dim(fused_latents)
        
        return fused_latents


class WeightedFusion(nn.Module):
    def __init__(self):
        super(WeightedFusion, self).__init__()
        # Start with weights initialized to 0.5, so it's like averaging initially.
        self.image_weight = nn.Parameter(torch.tensor(0.5))
        self.text_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, image_latents, text_latents):
        # Ensure weights sum to 1 for proper blending
        weights_sum = torch.sigmoid(self.image_weight) + torch.sigmoid(self.text_weight)
        image_weight_normalized = torch.sigmoid(self.image_weight) / weights_sum
        text_weight_normalized = torch.sigmoid(self.text_weight) / weights_sum

        # Weighted sum of latents
        fused_latents = (image_weight_normalized * image_latents) + (text_weight_normalized * text_latents)
        return fused_latents


class MinorAttentionFusionModule(nn.Module):
    def __init__(self, num_heads):
        super(MinorAttentionFusionModule, self).__init__()
        self.cross_attention = nn.MultiheadAttention(1024, num_heads)

    def forward(self, image_latents, text_latents):
        # Assuming image_latents and text_latents are of shape (batch_size, latent_dim)
        # Reshape latents to (batch_size, 1024, 1024) if needed
        batch_size, latent_dim = image_latents.shape
        
        # Reshape to (1024, batch_size, 1024) to match MultiheadAttention requirements
        image_latents = image_latents.view(batch_size, 1024, 1024).permute(1, 0, 2)
        text_latents = text_latents.view(batch_size, 1024, 1024).permute(1, 0, 2)
        
        # Apply cross-attention across rows (sequence length = 1024)
        attn_output, _ = self.cross_attention(image_latents, text_latents, text_latents)
        
        # Reshape back to original shape (batch_size, latent_dim)
        attn_output = attn_output.permute(1, 0, 2).contiguous().view(batch_size, latent_dim)
        
        # Apply the learned attention and fuse the latents
        fused_latents = 0.9 * image_latents.permute(1, 0, 2).contiguous().view(batch_size, latent_dim) + 0.1 * attn_output

        return fused_latents

# ================================================================================== Models ==================================================================================

# ========================================= Base Shap-E (to be inherited) =========================================

class BaseShapE(nn.Module):
    def __init__(self,
                 use_transmitter=True,
                 output_path='./output'):
        super(BaseShapE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if use_transmitter:
            self.xm = load_model('transmitter', device=self.device)
        else:
            self.xm = load_model('decoder', device=self.device)

        for param in self.xm.parameters():
            param.requires_grad = False

        self.diffusion = diffusion_from_config(load_config('diffusion'))

        self._batch_size = 0    # Changes dynamically based on input

        self.output_path = output_path
        self.gif_path = f'{self.output_path}/gifs'
        self.ply_path = f'{self.output_path}/ply_meshes'
        self.obj_path = f'{self.output_path}/obj_meshes'
        self._create_directories()


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
            
            # Append the tensor (flipped or not) to the list
            projection_tensors.append(images)
        
        # Stack projections along the batch dimension
        projections_stacked = torch.stack(projection_tensors)

        # Remove the singleton dimension if necessary (e.g., the second dimension)
        projections_squeezed = projections_stacked.squeeze(1) 

        # Permute to match the required shape: [batch_size, num_views, num_channels, height, width]
        projections_permuted_correctly = projections_squeezed.permute(0, 1, 4, 2, 3)  # Adjusting the dimensions

         # Flip the image, and clone to avoid in-place issues
        projections_permuted_correctly = torch.flip(projections_permuted_correctly, dims=[-2])  # Flip along the vertical axis (height)

        # Flip the images horizontally (along the width axis)
        projection_permuted_correctly = torch.flip(projections_permuted_correctly, dims=[-1])

        return projections_permuted_correctly

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

    

# ========================================= Shap-E =========================================

class ShapE(BaseShapE):
    def __init__(self,
                 input_mode=-1,
                 use_karras=True,
                 karras_steps=16,
                 guidance_scale=15.0,
                 use_transmitter=True,
                 output_path='./output'
                 ):
        super(ShapE, self).__init__()

        self.input_mode = input_mode

        if self.input_mode == -1:
            self.latent_diffusion_model = load_model('image300M', device=self.device)
        elif self.input_mode == -2:
            self.latent_diffusion_model = load_model('text300M', device=self.device)
        else:
            raise ValueError("Choose a valid latent diffusion model mode for Shap-E.")


        if use_transmitter:
            # Transmitter - the encoder and corresponding projection layers for converting encoder outputs into implicit neural representations.
            self.xm = load_model('transmitter', device=self.device)
        else:
            # decoder - just the final projection layer component of transmitter. This is a smaller checkpoint than transmitter since it does not
            # include parameters for encoding 3D assets. This is the minimum required model to convert diffusion outputs into implicit neural representations.
            self.xm = load_model('decoder', device=self.device)

        self.use_karras = use_karras
        self.karras_steps = karras_steps
        self.guidance_scale = guidance_scale

    def generate_latents(self, inputs):
        if self.input_mode == -1:
            model_kwargs = dict(images=inputs)
        elif self.input_mode == -2:
            model_kwargs=dict(texts=inputs)
        
        latents = sample_latents(
            batch_size=self._batch_size,
            model=self.latent_diffusion_model,
            diffusion=self.diffusion,
            guidance_scale=self.guidance_scale,
            model_kwargs=model_kwargs,
            progress=False,
            clip_denoised=True,
            use_fp16=True,
            use_karras=self.use_karras,
            karras_steps=self.karras_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        return latents


    def forward(self, inputs, remove_background=False):
        # Set batch size to the length of the inputs
        self._batch_size = len(inputs)

        if self.input_mode == -1:
            # Convert inputs to PIL format if they are tensors
            if isinstance(inputs[0], torch.Tensor):
                inputs = self._convert_tensor_batch_to_pil_list(inputs)

            if isinstance(inputs, Image.Image) and remove_background:
                inputs = [self._remove_background(image) for image in inputs]

        latents = self.generate_latents(inputs)
        
        return latents



# ========================================= Multimodal Shap-E Pipeline =========================================

class MMShapE(BaseShapE):
    def __init__(self,
                 fusion_mode=2,
                 use_karras=True,
                 image_karras_steps=1,
                 text_karras_steps=16,
                 latent_dim=1048576,
                 reduced_dim=512,
                 num_cm_heads=8,
                 use_transmitter=True,
                 parallelize=False,
                 output_path='./output'
                 ):
        super(MMShapE, self).__init__()

        self.fusion_mode = fusion_mode
        
        # image300M - the image-conditional latent diffusion model.
        self.image_model = load_model('image300M', device=self.device)
        for param in self.image_model.parameters():
            param.requires_grad = False
        
        # text300M - the text-conditional latent diffusion model.
        self.text_model = load_model('text300M', device=self.device)
        for param in self.text_model.parameters():
            param.requires_grad = False

        
        
        self.diffusion = diffusion_from_config(load_config('diffusion'))
        self.d_latent_dim = latent_dim
        self.d_reduced_dim = reduced_dim
        self.text_guidance_scale = 15.0
        self.image_guidance_scale = 3.0
        self.use_karras = use_karras
        self.image_karras_steps = image_karras_steps
        self.text_karras_steps = text_karras_steps

        self.parallelize = parallelize
        
        if fusion_mode == 2:
            self.cross_modal_attention = CrossModalAttention(latent_dim=self.d_latent_dim, reduced_dim=self.d_reduced_dim, num_heads=num_cm_heads)
        elif fusion_mode == 3:
            self.cross_modal_attention_seq_level = CrossModalAttentionSeqLevel(latent_dim=self.d_latent_dim, token_dim=1024, num_heads=num_cm_heads)
        elif fusion_mode == 4:
            self.gated_fusion = GatedFusionModule(latent_dim=self.d_latent_dim, reduced_dim=1024)
        elif fusion_mode == 5:
            self.weighted_fusion = WeightedFusion()
        elif fusion_mode == 6:
            self.minor_attention = MinorAttentionFusionModule(num_heads=num_cm_heads)



    def generate_latents(self, images, prompts):
        # Generate image latents
        image_latents = sample_latents(
            batch_size=self._batch_size,
            model=self.image_model,
            diffusion=self.diffusion,
            guidance_scale=self.image_guidance_scale,
            model_kwargs=dict(images=images),
            progress=False,
            clip_denoised=True,
            use_fp16=True,
            use_karras=self.use_karras,
            karras_steps=self.image_karras_steps,
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
            progress=False,
            clip_denoised=True,
            use_fp16=True,
            use_karras=self.use_karras,
            karras_steps=self.text_karras_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        return image_latents, text_latents
        

    def generate_latents_parallelized(self, images, prompts):
        # Function to generate image latents
        def generate_image_latents():
            return sample_latents(
                batch_size=self._batch_size,
                model=self.image_model,
                diffusion=self.diffusion,
                guidance_scale=self.image_guidance_scale,
                model_kwargs=dict(images=images),
                progress=False,
                clip_denoised=True,
                use_fp16=True,
                use_karras=self.use_karras,
                karras_steps=self.image_karras_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )

        # Function to generate text latents
        def generate_text_latents():
            return sample_latents(
                batch_size=self._batch_size,
                model=self.text_model,
                diffusion=self.diffusion,
                guidance_scale=self.text_guidance_scale,
                model_kwargs=dict(texts=prompts),
                progress=False,
                clip_denoised=True,
                use_fp16=True,
                use_karras=self.use_karras,
                karras_steps=self.text_karras_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )

        # Use ThreadPoolExecutor to run image and text latent generation in parallel
        with ThreadPoolExecutor() as executor:
            # Submit both tasks
            image_future = executor.submit(generate_image_latents)
            text_future = executor.submit(generate_text_latents)

            # Wait for both tasks to finish and retrieve the results
            image_latents = image_future.result()
            text_latents = text_future.result()

        return image_latents, text_latents
        
    
    def forward(self, images, prompts, remove_background=False):
        # Check if the lengths of images and prompts match
        if len(images) != len(prompts):
            raise ValueError(f"The number of images ({len(images)}) and prompts ({len(prompts)}) must be the same.")

        # Set batch size to the length of the images (or prompts, since they are now guaranteed to be the same length)
        self._batch_size = len(images)

        # Convert images to PIL format if they are tensors
        if isinstance(images[0], torch.Tensor):
            images = self._convert_tensor_batch_to_pil_list(images)


        if remove_background:
            images = [self._remove_background(image) for image in images]


        if self.parallelize:
            image_latents, text_latents = self.generate_latents_parallelized(images, prompts)
        else:
            image_latents, text_latents = self.generate_latents(images, prompts)

        if self.fusion_mode == 1:
            fused_latents = (0.5 * image_latents) + (0.5 * text_latents)
        elif self.fusion_mode == 2:
            fused_latents =  self.cross_modal_attention(image_latents, text_latents)
        elif self.fusion_mode == 3:
            fused_latents = self.cross_modal_attention_seq_level(image_latents, text_latents)
        elif self.fusion_mode == 4:
            fused_latents = self.gated_fusion(image_latents, text_latents)
        elif self.fusion_mode == 5:
            fused_latents = self.weighted_fusion(image_latents, text_latents)
        elif self.fusion_mode == 6:
            fused_latents = self.minor_attention(image_latents, text_latents)
        else:
            raise ValueError(f'Invalid fusion mode: {self.fusion_mode}')
        
        return fused_latents, image_latents, text_latents


# =================================================================================== Datasets ==================================================================================

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



# ========================================= Text2Shape =========================================

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


# ========================================= Objaverse =========================================

class ObjaverseDataset(Dataset):
    def __init__(self, image_dir, caption_csv, latent_code_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with rendered images.
            caption_csv (str): Path to the CSV file containing Object_ID, Caption, and Object_Name.
            latent_code_dir (str): Directory with latent vector codes.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.image_dir = image_dir
        self.caption_data, self.object_names = self._load_captions(caption_csv)
        self.latent_code_dir = latent_code_dir
        self.transform = transform
        self.valid_object_ids = self.calculate_valid_object_ids()

    def _load_captions(self, caption_csv):
        """Load captions and object names from a CSV file and return dictionaries mapping object IDs to captions and object names."""
        captions = {}
        object_names = {}
        with open(caption_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                object_id = row['Object_ID']
                captions[object_id] = row['Caption']
                object_names[object_id] = row['Object_Name']  # Extract the object name as well
        return captions, object_names

    def calculate_valid_object_ids(self):
        """Calculate valid object IDs that have images, captions, and latent codes."""
        valid_object_ids = set()
        
        # Iterate through all .pt files in the latent code directory
        for filename in os.listdir(self.latent_code_dir):
            if filename.endswith(".pt"):
                object_id = filename.split(".pt")[0]
                
                # Check if both images and captions exist for this object_id
                image1_path = os.path.join(self.image_dir, f"{object_id}_view_1.png")
                image2_path = os.path.join(self.image_dir, f"{object_id}_view_2.png")
                
                if (object_id in self.caption_data 
                    and os.path.isfile(image1_path) 
                    and os.path.isfile(image2_path)):
                    valid_object_ids.add(object_id)

        return valid_object_ids

    def __len__(self):
        return len(self.valid_object_ids)

    def __getitem__(self, idx):
        object_id = list(self.valid_object_ids)[idx]

        # Load images (SHAPES: [3, 400, 400])
        image1_path = os.path.join(self.image_dir, f"{object_id}_view_1.png")
        image2_path = os.path.join(self.image_dir, f"{object_id}_view_2.png")
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        # Apply any specified transformations to the images
        if self.transform:
            # Apply selective brightness and saturation adjustment
            brightness_factor = 2
            saturation_factor = 2
            threshold = 0.05
            image1 = adjust_image(image1, brightness_factor, saturation_factor, threshold)
            image2 = adjust_image(image2, brightness_factor, saturation_factor, threshold)

            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Load caption and object name
        caption = self.caption_data[object_id]
        object_name = self.object_names[object_id]

        # Convert object_name to a more readable form
        object_name = object_name.replace("_", " ").capitalize()

        # Load latent code
        latent_code_path = os.path.join(self.latent_code_dir, f"{object_id}.pt")
        latent_code = torch.load(latent_code_path)

        return image1, image2, caption, object_name, latent_code




# ================================================================================== Dataset Utility Functions ==================================================================================

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


def adjust_image(image, brightness_factor=1.5, saturation_factor=1.5, threshold=0.05):
    # Convert the image to grayscale to create a mask
    gray_image = transforms.functional.rgb_to_grayscale(image)  # Convert to grayscale (returns a PIL image)
    gray_tensor = transforms.functional.to_tensor(gray_image)   # Convert PIL grayscale image to tensor
    
    # Create a mask based on the threshold
    mask = gray_tensor > threshold  # Create a mask for areas where the object is present
    
    # Enhance brightness and saturation on the original image
    enhancer_brightness = ImageEnhance.Brightness(image)
    enhancer_saturation = ImageEnhance.Color(image)
    bright_image = enhancer_brightness.enhance(brightness_factor)
    saturated_image = enhancer_saturation.enhance(saturation_factor)
    
    # Convert enhanced images to tensors
    bright_tensor = transforms.functional.to_tensor(bright_image)
    saturated_tensor = transforms.functional.to_tensor(saturated_image)
    original_tensor = transforms.functional.to_tensor(image)
    
    # Apply the mask to keep the background unchanged
    adjusted_tensor = original_tensor.clone()  # Clone to keep the original image intact
    adjusted_tensor[mask.expand_as(original_tensor)] = bright_tensor[mask.expand_as(original_tensor)]
    adjusted_tensor[mask.expand_as(original_tensor)] = saturated_tensor[mask.expand_as(original_tensor)]
    
    # Convert the adjusted tensor back to a PIL image
    adjusted_image = transforms.functional.to_pil_image(adjusted_tensor)
    
    return adjusted_image


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


def render_voxels_as_images(voxel_tensor, size_of_render=64, num_views=3, corner_views=False, padding=10):
    """
    Render the voxel grid as 2D images by projecting it onto different planes.
    Only the occupancy is considered for rendering. Adds padding for zoom-out effect.

    :param voxel_tensor: A [C, D, H, W] voxel grid, where C is the number of channels
    :param num_views: The number of standard views to generate (default 3 for XY, YZ, XZ views)
    :param corner_views: Whether to include corner views (default True)
    :param padding: Extra space added around the projection for a zoom-out effect (default 10)
    :return: A list of PIL images
    """
    occupancy = voxel_tensor[3]  # Shape: [D, H, W]

    images = []

    # Normalize occupancy for rendering (0 to 1)
    occupancy_normalized = torch.clamp(occupancy, 0, 1)
    
    # Render different projections (XY, YZ, XZ)
    for axis in range(num_views):
        # Max along the axis to create a projection (highlight occupied voxels)
        projection, _ = torch.max(occupancy_normalized, dim=axis)
        
        # Convert the projection to uint8 format for image
        projection_image = (projection.cpu().numpy() * 255).astype(np.uint8)

        # Add padding (zoom out effect)
        projection_image = np.pad(projection_image, pad_width=padding, mode='constant', constant_values=0)

        # Create a PIL image from the projection
        img = Image.fromarray(projection_image)

        # Transformations
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        img = img.resize((size_of_render, size_of_render))
        
        images.append(img)

    # If corner views are requested, generate them by rotating the occupancy grid
    if corner_views:
        # Apply corner view rotations
        rotations = [(30, 45), (-20, 10), (60, 30)]  # Rotation angles (degrees)

        for rot_x, rot_y in rotations:
            # Rotate the occupancy grid for consistent projection
            rotated_occupancy = rotate_voxel_grid(occupancy_normalized, rot_x, rot_y)
            
            # Project the rotated voxel grid along Z axis (corner view)
            projection, _ = torch.max(rotated_occupancy, dim=2)

            # Convert to uint8 format for image
            projection_image = (projection.cpu().numpy() * 255).astype(np.uint8)

            # Add padding (zoom out effect)
            projection_image = np.pad(projection_image, pad_width=padding, mode='constant', constant_values=0)

            # Create a PIL image from the projection
            img = Image.fromarray(projection_image)

            # Transformations
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            img = img.resize((size_of_render, size_of_render))
            
            images.append(img)

    # Convert all images to 3-channel images
    images_3ch = []
    for img in images:
        img_3ch = np.stack((np.array(img),) * 3, axis=-1)  # Stack the grayscale image to 3 channels
        images_3ch.append(Image.fromarray(img_3ch))

    return images_3ch



def rotate_voxel_grid(voxel_tensor, angle_x, angle_y):
    """
    Rotate the voxel grid around the X and Y axes using the specified angles.
    
    :param voxel_tensor: A [D, H, W] occupancy grid
    :param angle_x: The angle to rotate around the X axis (in degrees)
    :param angle_y: The angle to rotate around the Y axis (in degrees)
    :return: The rotated voxel grid
    """
    voxel_numpy = voxel_tensor.cpu().numpy()  # Convert to numpy for scipy rotation

    # Rotate around X axis
    rotated_x = rotate(voxel_numpy, angle_x, axes=(1, 2), reshape=False, order=1)
    # Rotate around Y axis
    rotated_xy = rotate(rotated_x, angle_y, axes=(0, 2), reshape=False, order=1)

    return torch.tensor(rotated_xy)  # Convert back to tensor



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


# ================================================================================== Loss Functions ==================================================================================

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


# ================================================================================== Training ==================================================================================

# ========================================= Training with Text2Shape =========================================

def train_model_text2shape(model, dataloader, mode, loss_fn, optimizer, num_epochs=10, device='cuda', save_graphic_epoch=0):
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
    torch.cuda.empty_cache()

    model.train()  # Set the model to training mode

    for name, param in model.named_parameters():
        if "cross_modal_attention" in name: 
            param.requires_grad = True
        else:
            param.requires_grad = False


    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    print(f'{("="*20)}  TRAINING {("="*20)}')

    # Initialize GPU memory tracking
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using the first GPU

    print(f"GPU MEMORY BEFORE TRAINING: {nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1e6} GB")

    # Initialize variables to store total times for each operation
    total_latent_time = 0.0
    total_ortho_time = 0.0
    total_loss_time = 0.0
    num_batches = len(dataloader)  # Total number of batches

    for epoch in range(num_epochs):
        total_loss = 0.0
        max_gpu_memory = 0  # To track peak GPU memory usage

        epoch_start_time = time.time()

        for batch_idx, (sample_voxels, sample_descriptions, sample_images) in enumerate(dataloader):
            optimizer.zero_grad()

            if torch.isnan(sample_voxels).any() or torch.isnan(sample_images).any():
                print("NaN values detected in the input.")
                break

            # Move tensors to the specified device
            sample_voxels = sample_voxels.to(device)
            sample_images = sample_images.to(device)

            sample_projection_images = sample_images[:, -1]  # Side projection for each sample

            # Measure time for computing latents
            start_latent_time = time.time()
            fused_latents, _, _ = model.forward(
                images=sample_projection_images,
                prompts=sample_descriptions,
                remove_background=False
            )
            end_latent_time = time.time()
            latent_time = end_latent_time - start_latent_time
            total_latent_time += latent_time

            # Measure time for generating orthographic projections
            start_ortho_time = time.time()
            predicted_image_tensors = model.generate_orthographic_projections_grad(
                fused_latents,
                size_of_renders=64,
                render_mode="nerf"
            )
            end_ortho_time = time.time()
            ortho_time = end_ortho_time - start_ortho_time
            total_ortho_time += ortho_time

            # Measure time for computing loss
            start_loss_time = time.time()
            loss = loss_fn(predicted_image_tensors, sample_images)
            end_loss_time = time.time()
            loss_time = end_loss_time - start_loss_time
            total_loss_time += loss_time

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
        epoch_end_time = time.time()
        avg_loss = total_loss / num_batches
        max_gpu_memory_MB = max_gpu_memory / 1e6  # Convert to MB
        total_epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Time: {total_epoch_time:.2f}s, Peak GPU Memory: {max_gpu_memory_MB:.2f} MB")

    # Calculate average times for each operation
    avg_latent_time = total_latent_time / (num_epochs * num_batches)
    avg_ortho_time = total_ortho_time / (num_epochs * num_batches)
    avg_loss_time = total_loss_time / (num_epochs * num_batches)

    print(f"\nAverage time to compute latents: {avg_latent_time:.4f} seconds per batch")
    print(f"Average time to compute orthographic projections: {avg_ortho_time:.4f} seconds per batch")
    print(f"Average time to compute loss: {avg_loss_time:.4f} seconds per batch")

    # Save model weights at the end of training
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    torch.save(model.state_dict(), MODEL_PATHS[mode])
    print(f"Model saved at {MODEL_PATHS[mode]}.")


# ========================================= Training with Objaverse =========================================

def train_model_objaverse(model, dataloader, mode, loss_fn, optimizer, num_epochs=10, device='cuda'):
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
    torch.cuda.empty_cache()

    model.train()  # Set the model to training mode

    start_training_time = time.time()

    print(f'{("="*20)}  TRAINING {("="*20)}')
    print(f'Number of training samples: {len(dataloader.dataset)}')

    # Initialize GPU memory tracking
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using the first GPU

    print(f"GPU MEMORY BEFORE TRAINING: {nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1e6} GB")

    total_latent_time = 0.0
    total_loss_time = 0.0
    num_batches = len(dataloader)  # Total number of batches
    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0.0
        max_gpu_memory = 0  # To track peak GPU memory usage

        epoch_start_time = time.time()

        for batch_idx, (images1, images2, captions, object_names, latent_codes) in enumerate(dataloader):
            optimizer.zero_grad()

            # images1 is not used for now.

            # Move tensors to the specified device
            images2 = images2.to(device)
            latent_codes = latent_codes.to(device)

            # Measure time for computing latents
            start_latent_time = time.time()
            fused_latents, _, _ = model.forward(
                images=images2,
                prompts=captions,
                remove_background=False
            )
            end_latent_time = time.time()
            latent_time = end_latent_time - start_latent_time
            total_latent_time += latent_time


            # Measure time for computing loss
            start_loss_time = time.time()
            latent_codes = latent_codes.squeeze(1)  # Removes the dimension with size 1
            loss = loss_fn(fused_latents, latent_codes)
            end_loss_time = time.time()
            loss_time = end_loss_time - start_loss_time
            total_loss_time += loss_time

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Track GPU memory usage during the process
            mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            max_gpu_memory = max(max_gpu_memory, mem_info.used)

            torch.cuda.empty_cache()

        # End of epoch, calculate statistics
        epoch_end_time = time.time()
        avg_loss = total_loss / num_batches
        max_gpu_memory_MB = max_gpu_memory / 1e6  # Convert to MB
        total_epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Time: {total_epoch_time:.2f}s, Peak GPU Memory: {max_gpu_memory_MB:.2f} MB")

        if avg_loss < best_loss and (epoch + 1) % 5 == 0:
            if not os.path.exists(WEIGHTS_DIR):
                os.makedirs(WEIGHTS_DIR)
            torch.save(model.state_dict(), MODEL_PATHS[mode*10])
            print(f"Checkpoint saved at {MODEL_PATHS[mode*10]}.")


    end_training_time = time.time()

    training_time_minutes = (end_training_time - start_training_time) / 60

    # Calculate average times for each operation
    avg_latent_time = total_latent_time / (num_epochs * num_batches)
    avg_loss_time = total_loss_time / (num_epochs * num_batches)

    print(f"\nAverage time to compute latents: {avg_latent_time:.4f} seconds per batch")
    print(f"Average time to compute loss: {avg_loss_time:.4f} seconds per batch")
    print(f"Total Training time: {training_time_minutes:.2f} minutes")

    # Save model weights at the end of training
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    torch.save(model.state_dict(), MODEL_PATHS[mode*10])
    print(f"Model saved at {MODEL_PATHS[mode*10]}.")






# Used to test whether or not weights would change after training
def train_one_epoch_dummy(model, dataloader):
    torch.cuda.empty_cache() 

    # Simple loss function
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.cross_modal_attention.parameters(), lr=1e-4)

    print(("="*20) + " DUMMY TRAINING " + ("="*20))

    start_time = time.time()

    model.train()
    total_loss = 0.0

    # Initialize GPU memory tracking
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using the first GPU

    print(f"GPU MEMORY BEFORE TRAINING: {nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1e6} GB")

    max_gpu_memory = 0  # To track peak GPU memory usage
    
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

         # Track GPU memory usage during the process
        mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        max_gpu_memory = max(max_gpu_memory, mem_info.used)
        
        print(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    end_time = time.time()
    avg_loss = total_loss / len(dataloader)
    max_gpu_memory_MB = max_gpu_memory / 1e6  # Convert to MB
    total_time = end_time - start_time
    print(f"Average Loss: {avg_loss:.4f}, Time: {total_time:.2f}s, Peak GPU Memory: {max_gpu_memory_MB:.2f} MB")
    
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





# ================================================================================== Evaluation ==================================================================================

# ========================================= Text2Shape Evaluation =========================================

def evaluate_text2shape(model, dataloader, device='cuda', fid_batch_size=30):
    """
    Evaluate a text-to-shape model using FID, CLIP-R precision, and PSNR.
    
    Args:
        model: The pre-trained text-to-shape model to be evaluated.
        dataloader: DataLoader containing the dataset with real image tensors.
        device: Device to run the evaluation on ('cuda' or 'cpu').
        fid_batch_size: Number of images to process in a batch for FID calculation.
    
    Returns:
        fid_value: Computed FID score between real and generated images.
        average_clip_r_text: Computed CLIP-R precision with respect to text prompts.
        average_clip_r_image: Computed CLIP-R precision with respect to input images.
        overall_clip_r: Average of text-based and image-based CLIP-R precision.
        average_psnr: Computed average PSNR between real and generated images.
    """
    model.eval()

    # FID Prereqs
    real_image_batches = []
    generated_image_batches = []
    fid_preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # CLIP-R Precision Prereqs
    clip_model = SentenceTransformer("clip-ViT-B-32")

    # Initialize Nvidia SMI to monitor GPU memory usage
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using the first GPU

    start_time = time.time()  # Start timing

    all_descriptions = []
    all_input_images = []
    psnr_values = []

    # Lists to store selected viewpoints for CLIP-R precision
    selected_view_pred_images = []
    selected_view_real_images = []

    for batch_idx, (sample_voxels, sample_descriptions, sample_images) in enumerate(dataloader):
        # Move tensors to the specified device
        sample_voxels = sample_voxels.to(device)
        sample_images = sample_images.to(device)
        sample_projection_images = sample_images[:, -1]  # Side projection for each sample

        # Generate the 3D projections
        with torch.no_grad():
            if isinstance(model, MMShapE):
                latents, _, _ = model.forward(images=sample_projection_images, prompts=sample_descriptions, remove_background=False)
            elif isinstance(model, ShapE):
                if model.input_mode == -1:
                    inputs = sample_projection_images
                elif model.input_mode == -2:
                    inputs = sample_descriptions
                else:
                    raise ValueError("Choose a valid ShapE input mode.")
                latents = model.forward(inputs=inputs)

            predicted_image_tensors = model.generate_orthographic_projections_grad(latents, size_of_renders=64, render_mode="nerf")

        # Collect real and generated image tensors for FID calculation
        real_image_batches.append(sample_images.detach().cpu())
        generated_image_batches.append(predicted_image_tensors.detach().cpu())

        # Store descriptions and input images for CLIP-R precision
        all_descriptions += sample_descriptions
        all_input_images += sample_projection_images.detach().cpu()

        # Compute PSNR for the current batch
        for real_img, pred_img in zip(sample_images, predicted_image_tensors):
            selected_view_real_images.append(real_img[-1].cpu())  # Select the last viewpoint
            selected_view_pred_images.append(pred_img[-1].cpu())  # Select the last viewpoint

            # Compute PSNR for the current batch (use all viewpoints)
            psnr_value = psnr(pred_img, real_img)
            psnr_values.append(psnr_value)

    max_gpu_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    max_gpu_memory_MB = max_gpu_memory / 1e6
    print(f"GPU Memory: {max_gpu_memory_MB:.2f} MB")

    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken/60:.4f} minutes")

    # Concatenate all batches
    real_images = torch.cat(real_image_batches)
    batch_size, num_views, num_channels, height, width = real_images.shape
    real_images = real_images.view(batch_size * num_views, num_channels, height, width)

    generated_images = torch.cat(generated_image_batches)
    batch_size, num_views, num_channels, height, width = generated_images.shape
    generated_images = generated_images.view(batch_size * num_views, num_channels, height, width)

    # Preprocess images for FID calculation
    real_images_fid = torch.stack([fid_preprocess(img) for img in real_images])
    generated_images_fid = torch.stack([fid_preprocess(img) for img in generated_images])

    # Compute FID between real and generated images
    fid_value = compute_fid(real_images_fid, generated_images_fid, batch_size=fid_batch_size, cuda=(device == 'cuda'))
    print(f"FID score: {fid_value:.4f}")

    # Calculate CLIP-R precision using the selected viewpoints
    average_clip_r_text, average_clip_r_image, overall_clip_r = compute_clip_r_precision(
        clip_model, selected_view_pred_images, all_descriptions, selected_view_real_images
    )
    print(f"CLIP-R Precision (Text): {average_clip_r_text:.4f}")
    print(f"CLIP-R Precision (Image): {average_clip_r_image:.4f}")
    print(f"Overall CLIP-R Precision: {overall_clip_r:.4f}")

    # Calculate the average PSNR, excluding infinite values
    finite_psnr_values = [p for p in psnr_values if not torch.isinf(torch.tensor(p))]
    if finite_psnr_values:
        average_psnr = sum(finite_psnr_values) / len(finite_psnr_values)
        print(f"Average PSNR: {average_psnr:.4f} dB")
    else:
        average_psnr = None
        print("No finite PSNR values found for averaging.")

    return fid_value, average_clip_r_text, average_clip_r_image, overall_clip_r, average_psnr




# ========================================= Objaverse Evaluation =========================================

def evaluate_objaverse(model, dataloader, device='cuda', fid_batch_size=30, text_ablation=False, karras_ablation=False, test_parallelization=False):
    if text_ablation or karras_ablation or test_parallelization:
        if text_ablation:
            text_input_eval_variations = [
                "Results from using captions as text input",
                "Results from using object names as text input",
                "Results from using '\{object_name\} on its side' as text input",
                "Results from using 'An upright \{object_name\}' as text input"
            ]

            for i, text_input_eval_variation in enumerate(text_input_eval_variations):
                print(f"\n{('='*10)} {text_input_eval_variation} {('='*10)}")
                evaluate_objaverse_instance(model, dataloader, device, fid_batch_size, text_ablation_step=i)
            
        if karras_ablation:
            karras_configs = [(64, 64), (32, 32), (16, 16), (8, 8)]
            for image_karras_steps, text_karras_steps in karras_configs:
                model.image_karras_steps = image_karras_steps
                model.text_karras_steps = text_karras_steps
                print(f"\n{('='*10)} Results from using {(image_karras_steps, text_karras_steps)} image and text karras steps configurations {('='*10)}")
                evaluate_objaverse_instance(model, dataloader, device, fid_batch_size)
        if test_parallelization:
            model.parallelize = True
            print(f"\n{('='*10)} Results from parallelizing both latent diffusion modalities {('='*10)}")
            evaluate_objaverse_instance(model, dataloader, device, fid_batch_size)

            model.parallelize = False
            print(f"\n{('='*10)} Results from sequentializing both latent diffusion modalities {('='*10)}")
            evaluate_objaverse_instance(model, dataloader, device, fid_batch_size)
    else:
        evaluate_objaverse_instance(model, dataloader, device, fid_batch_size)




def evaluate_objaverse_instance(model, dataloader, device='cuda', fid_batch_size=30, text_ablation_step=0):
    """
    Evaluate a model using the Objaverse dataset with metrics like FID, CLIP-R precision, and PSNR.

    Args:
        model: The trained model to be evaluated.
        dataloader: DataLoader containing the Objaverse dataset.
        device: Device to run the evaluation on ('cuda' or 'cpu').
        fid_batch_size: Number of images to process in a batch for FID calculation.

    Returns:
        fid_value: Computed FID score between real and generated images.
        average_clip_r_text: Computed CLIP-R precision with respect to text prompts.
        average_clip_r_image: Computed CLIP-R precision with respect to input images.
        overall_clip_r: Average of text-based and image-based CLIP-R precision.
        average_psnr: Computed average PSNR between real and generated images.
        average_inference_time: Average time taken for inference per sample.
    """
    torch.cuda.empty_cache()

    model.eval()

    # FID Prereqs
    real_image_batches = []
    generated_image_batches = []
    fid_preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # CLIP-R Precision Prereqs
    clip_model = SentenceTransformer("clip-ViT-B-32")

    # Initialize Nvidia SMI to monitor GPU memory usage
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using the first GPU

    start_time = time.time()  # Start timing

    all_descriptions = []
    all_input_images = []
    psnr_values = []
    inference_times = []

    # Initialize lists for storing the selected viewpoint images for CLIP-R precision
    selected_view_pred_images = []
    selected_view_real_images = []

    cameras = create_pan_cameras(64, model.device)
    viewpoint_indices = [0, 4, 8, 12]

    for batch_idx, (images1, images2, captions, object_names, latent_codes) in enumerate(dataloader):
        # Move tensors to the specified device
        images2 = images2.to(device)
        latent_codes = latent_codes.to(device)

        # Time the inference for this batch
        batch_start_time = time.time()

        # Generate the 3D projections
        with torch.no_grad():
            # Forward pass to get predicted latents
            if isinstance(model, MMShapE):
                if text_ablation_step == 0:
                    text_inputs = captions
                elif text_ablation_step == 1:
                    text_inputs = object_names
                elif text_ablation_step == 2:
                    text_inputs = [f"{object_name} on its side" for object_name in object_names]
                else:
                    text_inputs = [f"An upright {object_name}" for object_name in object_names]

                latents, _, _ = model.forward(images=images2, prompts=text_inputs, remove_background=False)
            elif isinstance(model, ShapE):
                if model.input_mode == -1:
                    inputs = images2
                elif model.input_mode == -2:
                    inputs = captions
                else:
                    raise ValueError("Choose a valid ShapE input mode.")
                latents = model.forward(inputs=inputs)

            predicted_image_batches_pil = []
            real_image_batches_pil = []
            for i, pred_latent in enumerate(latents):
                predicted_images = decode_latent_images(model.xm, pred_latent, cameras, rendering_mode='nerf')
                predicted_images = [predicted_images[i] for i in viewpoint_indices]
                real_images = decode_latent_images(model.xm, latent_codes[i], cameras, rendering_mode='nerf')
                real_images = [real_images[i] for i in viewpoint_indices]

                # Append the entire set of viewpoints for FID and PSNR calculation
                predicted_image_batches_pil.append(predicted_images)
                real_image_batches_pil.append(real_images)

                # Extract a single viewpoint (e.g., the first viewpoint)
                selected_view_pred_images.append(predicted_images[0])
                selected_view_real_images.append(real_images[0])

        # Record the time taken for this batch's inference
        batch_inference_time = time.time() - batch_start_time
        inference_times.append(batch_inference_time)

        # Collect real and generated images for FID calculation
        real_image_batches.append(torch.stack([transforms.ToTensor()(img) for batch in real_image_batches_pil for img in batch]))
        generated_image_batches.append(torch.stack([transforms.ToTensor()(img) for batch in predicted_image_batches_pil for img in batch]))

        # Store descriptions and input images for CLIP-R precision
        all_descriptions += captions
        all_input_images += images2.cpu()

        # Compute PSNR for the current batch (compare tensor versions of the images)
        for real_img, pred_imgs in zip(real_image_batches, generated_image_batches):
            for pred_img in pred_imgs:
                psnr_value = psnr(pred_img, real_img)
                psnr_values.append(psnr_value)

    max_gpu_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    max_gpu_memory_MB = max_gpu_memory / 1e6
    print(f"GPU Memory: {max_gpu_memory_MB:.2f} MB")

    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken/60:.4f} minutes")

    # Concatenate all batches
    real_images = torch.cat(real_image_batches)
    generated_images = torch.cat(generated_image_batches)

    # Preprocess images for FID calculation
    real_images_fid = torch.stack([fid_preprocess(img) for img in real_images])
    generated_images_fid = torch.stack([fid_preprocess(img) for img in generated_images])

    # Compute FID between real and generated images
    fid_value = compute_fid(real_images_fid, generated_images_fid, batch_size=fid_batch_size, cuda=(device == 'cuda'))
    print(f"FID score: {fid_value:.4f}")

    # Calculate overall CLIP-R Precision using selected viewpoints
    average_clip_r_text, average_clip_r_image, overall_clip_r = compute_clip_r_precision(
        clip_model, selected_view_pred_images, all_descriptions, selected_view_real_images
    )
    print(f"CLIP-R Precision (Text): {average_clip_r_text:.4f}")
    print(f"CLIP-R Precision (Image): {average_clip_r_image:.4f}")
    print(f"Overall CLIP-R Precision: {overall_clip_r:.4f}")

    # Calculate the average PSNR, excluding infinite values
    finite_psnr_values = [p for p in psnr_values if not torch.isinf(torch.tensor(p))]
    if finite_psnr_values:
        average_psnr = sum(finite_psnr_values) / len(finite_psnr_values)
        print(f"Average PSNR: {average_psnr:.4f} dB")
    else:
        average_psnr = None
        print("No finite PSNR values found for averaging.")

    # Calculate the average inference time per sample
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time per batch: {average_inference_time:.4f} seconds")

    return fid_value, average_clip_r_text, average_clip_r_image, overall_clip_r, average_psnr, average_inference_time





# ========================================= FID =========================================

def get_inception_model(cuda=True):
    """Load pre-trained InceptionV3 model for FID calculation."""
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove final classification layer
    if cuda:
        model.cuda()
    model.eval()
    return model

def get_activations(images, model, batch_size=50, dims=2048, cuda=True):
    """Calculate activations of the pool_3 layer for all images."""
    model.eval()
    if batch_size > len(images):
        batch_size = len(images)

    pred_arr = np.empty((len(images), dims))
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        if cuda:
            batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch)

        pred_arr[i:i + batch_size] = pred.cpu().numpy()

    return pred_arr

def calculate_activation_statistics(images, model, batch_size=50, dims=2048, cuda=True):
    """Calculate the mean and covariance of activations."""
    act = get_activations(images, model, batch_size, dims, cuda)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])).dot(sigma2 + eps * np.eye(sigma2.shape[0])))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def compute_fid(real_images, generated_images, batch_size, cuda=True, dims=2048):
    """Compute FID between two sets of images."""
    model = get_inception_model(cuda)
    mu1, sigma1 = calculate_activation_statistics(real_images, model, batch_size, dims, cuda)
    mu2, sigma2 = calculate_activation_statistics(generated_images, model, batch_size, dims, cuda)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value




# ========================================= CLIP-R Precision =========================================


def compute_clip_r_precision(model, generated_images, text_prompts, input_images):
    """Calculate CLIP-R precision for the generated images with respect to text prompts and input images."""
    text_cosine_scores = []
    image_cosine_scores = []

    for img_tensor, text, input_image in zip(generated_images, text_prompts, input_images):
        # Convert tensors to PIL Image
        if torch.is_tensor(img_tensor):
            generated_image_pil = transforms.ToPILImage()(img_tensor)
        else:
            generated_image_pil = img_tensor
        if torch.is_tensor(input_image):
            input_image_pil = transforms.ToPILImage()(input_image)
        else:
            input_image_pil = input_image
        
        # Encode the generated image, input image, and text
        generated_image_emb = model.encode([generated_image_pil])
        input_image_emb = model.encode([input_image_pil])
        text_emb = model.encode([text])
        
        # Compute cosine similarities
        text_cos_score = util.cos_sim(generated_image_emb, text_emb)
        image_cos_score = util.cos_sim(generated_image_emb, input_image_emb)
        
        # Store the cosine similarities
        text_cosine_scores.append(text_cos_score[0][0].cpu().numpy())
        image_cosine_scores.append(image_cos_score[0][0].cpu().numpy())
    
    # Calculate the average cosine similarities
    average_clip_r_text = np.mean(text_cosine_scores)
    average_clip_r_image = np.mean(image_cosine_scores)
    overall_clip_r = (average_clip_r_text + average_clip_r_image) / 2

    return average_clip_r_text, average_clip_r_image, overall_clip_r


#========================================= PSNR =========================================

def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1 / mse)
        


# ================================================================================== Demo and Display ==================================================================================

# ========================================= Text2Shape Demo =========================================

def demo_text2shape(model, dataloader, device='cuda', num_samples=2):
    """
    Demo function to showcase text-to-shape model results.
    
    Args:
        model: The model to generate results from.
        dataloader: DataLoader containing the dataset.
        device: Device to run the demo on ('cuda' or 'cpu').
        num_samples: Number of samples to demo.
    """
    
    # Record the start time
    start_time = time.time()
    
    model.eval()  # Set the model to evaluation mode
    num_samples_shown = 0

    # Initialize Nvidia SMI to monitor GPU memory usage
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using the first GPU

    
    # Load the randomly selected batch
    for batch_idx, (sample_voxels, sample_descriptions, sample_images) in enumerate(dataloader):
        # Move tensors to the specified device
        sample_images = sample_images.to(device)
        sample_projection_images = sample_images[:, -1]  # Take the last projection image for each sample

        # Calculate latents
        fused_latents, image_latents, text_latents = model.forward(
            images=sample_projection_images,
            prompts=sample_descriptions,
            remove_background=False
        )
        
        # Generate orthographic projections for fused, image, and text latents
        with torch.no_grad():
            predicted_fused_images = model.generate_orthographic_projections_grad(fused_latents, size_of_renders=64, render_mode="nerf")
            predicted_image_images = model.generate_orthographic_projections_grad(image_latents, size_of_renders=64, render_mode="nerf")
            predicted_text_images = model.generate_orthographic_projections_grad(text_latents, size_of_renders=64, render_mode="nerf")


        # Convert projections to lists of images
        predicted_image_images = convert_projection_tensors_into_lists(predicted_image_images)
        predicted_text_images = convert_projection_tensors_into_lists(predicted_text_images)
        predicted_fused_images = convert_projection_tensors_into_lists(predicted_fused_images)
        sample_images = convert_projection_tensors_into_lists(sample_images)

        for i in range(num_samples):
            fig, axs = plt.subplots(4, len(sample_images[i]), figsize=(15, 5))

            for k in range(len(sample_images[i])):
                axs[0, k].imshow(predicted_image_images[i][k])  
                axs[0, k].axis('off')  # Hide axes
                axs[1, k].imshow(predicted_text_images[i][k])  
                axs[1, k].axis('off')  # Hide axes
                axs[2, k].imshow(predicted_fused_images[i][k])  
                axs[2, k].axis('off')  # Hide axes
                axs[3, k].imshow(sample_images[i][k])  
                axs[3, k].axis('off')  # Hide axes

            plt.suptitle(sample_descriptions[i], fontsize=16)
            plt.tight_layout()
            file_name = f'text2shape_demo_{i}.png'
            plt.savefig(file_name)
            plt.close(fig)
            print(f"Saved {file_name}.")

            num_samples_shown += 1

        if num_samples_shown >= num_samples:
            break
    
    # Monitor GPU memory usage
    max_gpu_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used
    max_gpu_memory_MB = max_gpu_memory / 1e6
    print(f"Peak GPU Memory: {max_gpu_memory_MB:.2f} MB")

    # Record the end time and calculate total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Time: {total_time:.2f} seconds")



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




# ========================================= Objaverse Demo =========================================

def demo_objaverse(model, dataloader, device, output_dir='./output/demos/', text_ablation=False, karras_ablation=False):
    """
    Demonstrates the model's predictions on a given number of samples from the test set.
    
    Args:
        model: The trained model to be evaluated.
        dataloader: DataLoader for the test set.
        device: Device to run the inference on ('cuda' or 'cpu').
        output_dir: Directory to save the generated GIFs.
        karras_ablation: If True, performs the karras steps ablation demo instead of the default.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()  # Set the model to evaluation mode

    cameras = create_pan_cameras(64, model.device)
    
    with torch.no_grad():
        for i, (images1, images2, captions, object_names, latent_codes) in enumerate(dataloader):
            # Move tensors to the specified device
            images2 = images2.to(device)
            latent_codes = latent_codes.to(device)

            if text_ablation or karras_ablation:
                if text_ablation:
                    # Text ablation demo procedure
                    for sample_idx in range(len(images2)):
                        fig, axs = plt.subplots(4, 2, figsize=(10, 20))
                        
                        # First row: Input Image and Ground Truth
                        axs[0, 0].imshow(images2[sample_idx].cpu().permute(1, 2, 0))
                        axs[0, 0].set_title("Input Image")
                        axs[0, 0].axis('off')

                        gt_images = decode_latent_images(model.xm, latent_codes[sample_idx], cameras, rendering_mode='nerf')
                        gt_im = axs[0, 1].imshow(gt_images[0])
                        axs[0, 1].set_title("Ground Truth")
                        axs[0, 1].axis('off')

                        # Store all frames for animation
                        fused_images_list = []

                        # Caption-based generation
                        inputs = [
                            (captions[sample_idx], "Caption"),
                            (object_names[sample_idx], "Object Name"),
                            (f"An upright {object_names[sample_idx]}", "Augmented Prompt")
                        ]
                        
                        # Iterate through each input type and generate predictions
                        for row_idx, (text_input, input_type) in enumerate(inputs, start=1):
                            # Generate fused latents using the given text input
                            fused_latents, _, _ = model.forward(
                                images=images2[sample_idx:sample_idx + 1],
                                prompts=[text_input],
                                remove_background=False
                            )
                            fused_images = decode_latent_images(model.xm, fused_latents[0], cameras, rendering_mode='nerf')
                            fused_images_list.append(fused_images)

                            # Display the text input and associated predictions
                            text_input = "\n".join(textwrap.wrap(text_input, width=50))
                            axs[row_idx, 0].text(0.5, 0.5, text_input, ha='center', va='center', fontsize=12)
                            axs[row_idx, 0].set_title(f"{input_type} Input")
                            axs[row_idx, 0].axis('off')

                            fused_im = axs[row_idx, 1].imshow(fused_images[0])
                            axs[row_idx, 1].set_title(f"Prediction with {input_type}")
                            axs[row_idx, 1].axis('off')

                        # Animation function for all rows, including the ground truth
                        def update_ablation(frame):
                            gt_im.set_array(gt_images[frame])
                            for row_idx, fused_images in enumerate(fused_images_list, start=1):
                                axs[row_idx, 1].images[0].set_array(fused_images[frame])
                            return [gt_im] + [im for row in axs[1:] for ax in row for im in ax.images]

                        # Create animation for all rows
                        ani = FuncAnimation(fig, update_ablation, frames=len(fused_images_list[0]), interval=200, blit=False)
                        mode = model.fusion_mode
                        model_path = MODEL_PATHS[mode * 10]
                        model_name = model_path.split('/')[-1].split('.')[0]
                        gif_path = os.path.join(output_dir, f'sample_{sample_idx}_text_ablation-{model_name}.gif')
                        ani.save(gif_path, writer=PillowWriter(fps=10))
                        print(f"Saved text ablation plot for sample {sample_idx} at {gif_path}")

                        # Close the plot to free memory
                        plt.close()

                if karras_ablation:
                    # Karras ablation demo procedure
                    karras_configs = [(64, 64), (32, 32), (16, 16), (8, 8)]
                    for sample_idx in range(len(images2)):
                        fig, axs = plt.subplots(len(karras_configs) + 1, 3, figsize=(15, 5 * (len(karras_configs) + 1)))
                        
                        # Wrap caption text
                        wrapped_caption = "\n".join(textwrap.wrap(captions[sample_idx], width=50))

                        # First row: Caption, input image, ground truth
                        axs[0, 0].text(0.5, 0.5, wrapped_caption, ha='center', va='center', fontsize=12)
                        axs[0, 0].set_title("Caption")
                        axs[0, 0].axis('off')

                        axs[0, 1].imshow(images2[sample_idx].cpu().permute(1, 2, 0))
                        axs[0, 1].set_title("Input Image")
                        axs[0, 1].axis('off')

                        gt_images = decode_latent_images(model.xm, latent_codes[sample_idx], cameras, rendering_mode='nerf')
                        gt_im = axs[0, 2].imshow(gt_images[0])
                        axs[0, 2].set_title("Ground Truth")
                        axs[0, 2].axis('off')

                        # Store all frames for animation
                        fused_images_list = []
                        image_latent_images_list = []
                        text_latent_images_list = []

                        # Iterate over karras configurations and display results
                        for row_idx, (image_steps, text_steps) in enumerate(karras_configs, start=1):
                            model.image_karras_steps = image_steps
                            model.text_karras_steps = text_steps
                            
                            # Measure inference time
                            start_time = time.time()
                            fused_latents, image_latents, text_latents = model.forward(
                                images=images2[sample_idx:sample_idx + 1],
                                prompts=[captions[sample_idx]],
                                remove_background=False
                            )
                            duration = time.time() - start_time

                            fused_images = decode_latent_images(model.xm, fused_latents[0], cameras, rendering_mode='nerf')
                            image_latent_images = decode_latent_images(model.xm, image_latents[0], cameras, rendering_mode='nerf')
                            text_latent_images = decode_latent_images(model.xm, text_latents[0], cameras, rendering_mode='nerf')

                            # Append images for animation
                            fused_images_list.append(fused_images)
                            image_latent_images_list.append(image_latent_images)
                            text_latent_images_list.append(text_latent_images)

                            # Initialize the display for each row's images
                            fused_im = axs[row_idx, 0].imshow(fused_images[0])
                            axs[row_idx, 0].set_title(f"Fused Prediction\n{image_steps}/{text_steps} steps\n{duration:.2f}s")
                            axs[row_idx, 0].axis('off')

                            image_latent_im = axs[row_idx, 1].imshow(image_latent_images[0])
                            axs[row_idx, 1].set_title(f"Image Latent\n{image_steps} steps")
                            axs[row_idx, 1].axis('off')

                            text_latent_im = axs[row_idx, 2].imshow(text_latent_images[0])
                            axs[row_idx, 2].set_title(f"Text Latent\n{text_steps} steps")
                            axs[row_idx, 2].axis('off')

                        # Animation function for all rows, including the ground truth
                        def update_ablation(frame):
                            gt_im.set_array(gt_images[frame])
                            for row_idx, (fused_images, image_latent_images, text_latent_images) in enumerate(
                                zip(fused_images_list, image_latent_images_list, text_latent_images_list), start=1):
                                axs[row_idx, 0].images[0].set_array(fused_images[frame])
                                axs[row_idx, 1].images[0].set_array(image_latent_images[frame])
                                axs[row_idx, 2].images[0].set_array(text_latent_images[frame])
                            return [gt_im] + [im for row in axs[1:] for ax in row for im in ax.images]

                        # Create animation for all rows
                        ani = FuncAnimation(fig, update_ablation, frames=len(fused_images_list[0]), interval=200, blit=False)
                        mode = model.fusion_mode
                        model_path = MODEL_PATHS[mode*10]
                        model_name = model_path.split('/')[-1].split('.')[0]
                        gif_path = os.path.join(output_dir, f'sample_{sample_idx}_karras_all-{model_name}.gif')
                        ani.save(gif_path, writer=PillowWriter(fps=10))
                        print(f"Saved Karras ablation plot for sample {sample_idx} at {gif_path}")

                        # Close the plot to free memory
                        plt.close()
            
            else:
                # Default demo procedure
                model.image_karras_steps = 16
                model.text_karras_steps = 16

                # Get the predicted fused latents from the model
                fused_latents, image_latents, text_latents = model.forward(
                    images=images2,
                    prompts=captions,
                    remove_background=False
                )

                for j, fused_latent in enumerate(fused_latents):
                    gt_gif_path = model.gif_path + '/' + f'sample_{j}_gt.gif'
                    pred_gif_path = model.gif_path + '/' + f'sample_{j}_pred.gif'

                    gt_images = decode_latent_images(model.xm, latent_codes[j], cameras, rendering_mode='nerf')
                    pred_images = decode_latent_images(model.xm, fused_latent, cameras, rendering_mode='nerf')

                    imageio.mimsave(gt_gif_path, gt_images, fps=10)  
                    imageio.mimsave(pred_gif_path, pred_images, fps=10)  
                    
                    # Plotting
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(captions[j], fontsize=10)  # Use the caption as the title

                    # Input image
                    axs[0].imshow(images2[j].cpu().permute(1, 2, 0))
                    axs[0].set_title("Input Image")
                    axs[0].axis('off')

                    # Prepare placeholders for the animated images
                    gt_im = axs[1].imshow(gt_images[0])
                    pred_im = axs[2].imshow(pred_images[0])
                    axs[1].set_title("Ground Truth")
                    axs[1].axis('off')
                    axs[2].set_title("Predicted")
                    axs[2].axis('off')

                    # Animation function
                    def update(frame):
                        gt_im.set_array(gt_images[frame])
                        pred_im.set_array(pred_images[frame])
                        return [gt_im, pred_im]

                    # Create the animation
                    ani = FuncAnimation(fig, update, frames=len(gt_images), interval=200, blit=True)
                    mode = model.fusion_mode
                    model_path = MODEL_PATHS[mode*10]
                    model_name = model_path.split('/')[-1].split('.')[0]
                    gif_path = os.path.join(output_dir, f'sample_{j}_comparison-{model_name}.gif')
                    ani.save(gif_path, writer=PillowWriter(fps=10))
                    print(f"Saved animated comparison plot for sample {j} at {gif_path}")

                    # Close the plot to free memory
                    plt.close()

            break  # Process only the first batch for the demo


# ========================================= General Demos =========================================

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




# ================================================================================== Main ==================================================================================

def main(FLAGS):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ================= Initializing the dataset ================= 
    if FLAGS.dataset == 'text2shape':
        print("Initializing Text2Shape...")

        dataset_dir = FLAGS.t2s_nrrd_dir
        csv_dir = FLAGS.t2s_csv_path
        dataset = Text2ShapeDataset(dataset_dir=dataset_dir, csv_path=csv_dir)
    elif FLAGS.dataset == 'objaverse':
        print("Initializing Objaverse...")

        image_dir = FLAGS.obja_img_dir
        caption_csv = FLAGS.obja_csv_path
        latent_code_dir = FLAGS.obja_latent_dir
        transform = transforms.ToTensor()

        dataset = ObjaverseDataset(
            image_dir=image_dir,
            caption_csv=caption_csv,
            latent_code_dir=latent_code_dir,
            transform=transform
        )
    else:
        raise ValueError(f"Invalid Dataset: {FLAGS.dataset}")

    train_loader, test_loader = create_train_test_split(dataset, test_size=0.01, batch_size=FLAGS.batch_size, num_samples=FLAGS.n)




    # ================= Initializing the model ================= 
    start_time = time.time()
    if FLAGS.mode < 0:
        print("Initializing ShapE model...")

        # Input modes:
        #  -1 -> Image inputs,
        #  -2 -> Text inputs

        model = ShapE(
                        input_mode=FLAGS.mode,
                        use_karras=True,
                        karras_steps=32,
                        guidance_scale=15.0,
                        use_transmitter=True,
                        output_path='./output'
                    ).to(device)
        print(f"karras_steps = {model.karras_steps}")

    else:
        print("Initializing MMShapE model...")

        # Fusion modes:
        #  1 -> Average fusion (not trainable),
        #  2 -> Cross-modal fusion,
        #  3 -> Cross-modal fusion at sequence level,
        #  4 -> Gated fusion,
        #  5 -> Weighted fusion,

        model = MMShapE(
                        fusion_mode=FLAGS.mode,
                        use_karras=True,
                        image_karras_steps=32,
                        text_karras_steps=32,
                        latent_dim=1048576,
                        #reduced_dim=512,
                        reduced_dim=1024,
                        num_cm_heads=8,
                        use_transmitter=True,
                        parallelize=FLAGS.parallelize,
                        output_path='./output'
                    ).to(device)
        print(f"image_karras_steps = {model.image_karras_steps}")
        print(f"text_karras_steps = {model.text_karras_steps}")
        print(f"parallelize = {model.parallelize}")


        if FLAGS.load:
            model_path = get_model_path(FLAGS.mode, FLAGS.dataset)
            print(f"Loading {model_path} weights...")
            model.load_state_dict(torch.load(model_path))

    print("Model: ", model)
    print("\n")
    print("--- %s seconds to load MMShapE---" % (time.time() - start_time))





    # ================= Training ================= 

    if FLAGS.train:
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
        if FLAGS.dataset == "text2shape":
            #optimizer = optim.Adam(model.cross_modal_attention.parameters(), lr=FLAGS.learning_rate)
            #loss_fn = PerceptualLoss()
            loss_fn = torch.nn.MSELoss()
            train_model_text2shape(model, train_loader, FLAGS.mode, loss_fn, optimizer, num_epochs=FLAGS.epochs, device=device, save_graphic_epoch=0)
        elif FLAGS.dataset == "objaverse":
            loss_fn = torch.nn.MSELoss()
            train_model_objaverse(model, train_loader, FLAGS.mode, loss_fn, optimizer, num_epochs=FLAGS.epochs, device=device)
        else:
            raise ValueError(f"Invalid Dataset: '{FLAGS.dataset}' for training")

        #dummy_dataset = DummyDataset(num_samples=10)
        #dataloader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)
        #check_weight_updates(model, dataloader)




    # ================= Evaluation ================= 

    if FLAGS.eval:
        if FLAGS.dataset == "text2shape":
            evaluate_text2shape(model, test_loader, device=device, fid_batch_size=30)
        elif FLAGS.dataset == "objaverse":
            evaluate_objaverse(model, test_loader, device=device, fid_batch_size=30, text_ablation=FLAGS.text_ablation, karras_ablation=FLAGS.karras_ablation, test_parallelization=FLAGS.test_parallelization)
        else:
            raise ValueError(f"Invalid Dataset: '{FLAGS.dataset}' for evaluation")





    # ================= Demo ================= 

    if FLAGS.demo:
        """
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
        """

        if FLAGS.dataset == "text2shape":
            demo_text2shape(model, train_loader, device=device, num_samples=FLAGS.n)
        elif FLAGS.dataset == "objaverse":
            demo_objaverse(model, train_loader, device, output_dir='./output/demos/', text_ablation=FLAGS.text_ablation, karras_ablation=FLAGS.karras_ablation)
        else:
            raise ValueError(f"Invalid Dataset: '{FLAGS.dataset}' for demo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Multimodal Shap-E')

    # ======= Dataset Args =======
    parser.add_argument('--dataset',
                        type=str, default='text2shape',
                        help='''
                        Dataset to use for training, eval, or demo.
                        Compatible datasets:
                        * text2shape
                        * objaverse
                        ''')
    parser.add_argument('--t2s_nrrd_dir',
                        type=str, default="/lustre/fs1/home/cap6411.student3/final_project/text2shape/nrrd_256_filter_div_32/nrrd_256_filter_div_32",
                        help='Text2Shape directory storing all nrrd voxelization files.')
    parser.add_argument('--t2s_csv_path',
                        type=str, default="/lustre/fs1/home/cap6411.student3/final_project/text2shape/captions.tablechair.csv",
                        help='Text2Shape csv file path mapping object ids to captions.')
    parser.add_argument('--obja_img_dir',
                        type=str, default="/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/Objaverse_Dataset/rendered_images/",
                        help='Objaverse directory storing rendered images in the form "<object_id>_view_1.png" and "<object_id>_view_2.png".')
    parser.add_argument('--obja_csv_path',
                        type=str, default="/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/objaverse_csv.csv",
                        help='Objaverse csv file path mapping object ids to captions.')
    parser.add_argument('--obja_latent_dir',
                        type=str, default="/lustre/fs1/home/cap6411.student3/final_project/Cap3D/Cap3D_latentcodes/",
                        #type=str, default="/lustre/fs1/home/cap6411.student3/final_project/Cap3D/uploaded_Cap3D_latentcodes/",
                        help='Objaverse directory storing Shap-E latent codes for objects having names "<object_id>.pt".')
    parser.add_argument('--n',
                        type=int, default=0,
                        help='Number of samples to use when training and/or evaluating. Leave at 0 for all samples.')

    # ======= Model Args =======
    parser.add_argument('--mode',
                        type=int, default=1,
                        help="""
                        Determines what fusion mode or model to use.
                        Current fusion/model modes:
                        * -1 -> img-to-3D Shap-E
                        * -2 -> txt-to-3D Shap-E
                        * 1 -> Average fusion MM-Shap-E
                        * 2 -> Cross-modal fusion MM-Shap-E
                        * 3 -> Cross-modal fusion at sequence level MM-Shap-E
                        * 4 -> Gated fusion MM-Shap-E
                        * 5 -> Weighted fusion MM-Shap-E
                        * 6 -> Minor attention MM-Shap-E 
                        """)
    parser.add_argument('--parallelize',
                        action='store_true',
                        help='An option to parallelize image and text latent diffusion models in late-fusion MM-Shap-E.')
    parser.add_argument('--load',
                        action='store_true',
                        help='An option to load weights before training, eval, or demo')


    # ======= Training Args =======
    parser.add_argument('--epochs',
                        type=int, default=3,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch_size',
                        type=int, default=8,
                        help='batch size.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--train',
                        action='store_true',
                        help='Run training loop.')


    # ======= Eval and Demo Args =======
    parser.add_argument('--text_ablation',
                        action='store_true',
                        help='Run text ablation variations of eval and/or demo.')
    parser.add_argument('--karras_ablation',
                        action='store_true',
                        help='Run karras ablation variations of eval and/or demo.')


    # ======= Eval Args =======
    parser.add_argument('--eval',
                        action='store_true',
                        help='Run evaluation loop.')
    parser.add_argument('--test_parallelization',
                        action='store_true',
                        help='Run evaluation with and without late fusion parallelization.')

    # ======= Demo Args =======
    parser.add_argument('--demo',
                        action='store_true',
                        help='Run demo')
    
    FLAGS, unparsed = parser.parse_known_args()
  
    main(FLAGS)