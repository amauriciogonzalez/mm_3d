import sys
import os

sys.path.insert(0, "./shap_e")

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh, gif_widget, decode_latent_mesh
#from shap_e.util.image_util import load_image
import imageio
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
from rembg import remove
from torchvision import transforms


# ----------------------- Model Layers for Multimodal Fusion -----------------------

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


# ----------------------- Multimodal Shap-E Pipeline -----------------------

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

        self._batch_size = 0
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
                gif_file_path = f"{output_gif_path}_{i}.gif"  # Save each latent as a separate gif
                imageio.mimsave(gif_file_path, images, fps=gif_fps)  # You can adjust the FPS
                print(f"Saved GIF: {gif_file_path}")

            if display_gif:
                display(gif_widget(images))

    def save_latents_as_meshes(self, latents, prompts="", save_ply=False, save_obj=False):
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

        if self.fusion_mode == 1:
            fused_latents = self.average_fusion(image_latents, text_latents)
        elif self.fusion_mode == 2:
            fused_latents = self.cross_modal_fusion(image_latents, text_latents)
        else:
            raise ValueError(f'Invalid fusion mode: {self.fusion_mode}')
        
        #print("fused_latents: ", fused_latents)
        
        return fused_latents, image_latents, text_latents

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

        fused_latents, image_latents, text_latents = self.generate_latents(images, prompts)

        #self.decode_display_save(fused_latents, self.device, prompt, render_mode="nerf")
        
        return fused_latents, image_latents, text_latents


# ----------------------- Datasets -----------------------

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


# ----------------------- Training -----------------------


def train_one_epoch(model, dataloader):
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

# Check if weights are updated
def check_weight_updates(model, dataloader):
    initial_weights = torch.clone(list(model.cross_modal_attention.parameters())[0].data)
    
    # Perform one step of training
    train_one_epoch(model, dataloader)
    
    # Check if weights have changed
    current_weights = list(model.cross_modal_attention.parameters())[0].data
    weight_diff = torch.sum(torch.abs(current_weights - initial_weights)).item()
    
    if weight_diff > 0:
        print(f"Model weights updated. Total weight change: {weight_diff:.6f}")
    else:
        print("Model weights did not update.")


# ----------------------- Demo -----------------------

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




# ----------------------- Main -----------------------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize Dataset and DataLoader
    print("Preparing dummy dataset...")

    dummy_dataset = DummyDataset(num_samples=10)
    dataloader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)

    # Initialize the MMShapE model
    print("Initializing MMShapE model...")

    # Fusion modes:
    #  1 -> Average fusion (not trainable),
    #  2 -> cross-modal fusion

    model = MMShapE(
        fusion_mode=1,      
        latent_dim=1048576,
        reduced_dim=512,
        num_heads=8,
        use_transmitter=True,
        output_path="./output"
        ).to(device)

    print("Model: ", model)


    # Demo from samples directory. The file names will be considered the text descriptions.
    samples_dir = "./samples"

    demo_from_samples(samples_dir,
                        model,
                        decode_fused_latents=True,
                        decode_image_latents=True,
                        decode_text_latents=True,
                        save_gif=True,
                        save_ply=False,
                        save_obj=False)
    


        
    # train and run weight update check
    print("Checking if weights are updated...")
    check_weight_updates(model, dataloader)

if __name__ == "__main__":
    main()