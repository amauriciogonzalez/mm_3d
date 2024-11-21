#----------------------------- GPU, time tracking-----------------------

import torch
import os
import time
import csv
import numpy as np
from PIL import Image
import imageio
import nvidia_smi
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import rembg
from utils.app_utils import (
    remove_background, 
    resize_foreground, 
    set_white_background,
    resize_to_128,
    to_tensor,
    get_source_camera_v2w_rmo_and_quats,
    get_target_cameras,
    export_to_obj
)

from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor

def load_model(device):
    # Load model configuration and weights
    model_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "gradio_config.yaml"))
    model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-multi-category-v1", filename="model_latest.pth")

    # Initialize the model
    model = GaussianSplatPredictor(model_cfg)
    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model.to(device)

    return model, model_cfg

def preprocess_image(input_image, rembg_session, preprocess_background=True, foreground_ratio=0.65):
    # Preprocess image: remove background, resize, and set white background
    if preprocess_background:
        image = remove_background(input_image.convert("RGB"), rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = set_white_background(image)
    else:
        image = set_white_background(input_image if input_image.mode == "RGBA" else input_image)

    image = resize_to_128(image)
    return np.array(image)

def reconstruct_and_export(model, model_cfg, image, device, output_dir):
    # Prepare image for inference
    image_tensor = to_tensor(image).to(device)
    
    # Camera transforms
    view_to_world_source, rot_transform_quats = get_source_camera_v2w_rmo_and_quats()
    view_to_world_source = view_to_world_source.to(device)
    rot_transform_quats = rot_transform_quats.to(device)

    # Model inference
    with torch.no_grad():
        reconstruction_unactivated = model(
            image_tensor.unsqueeze(0).unsqueeze(0),
            view_to_world_source,
            rot_transform_quats,
            None,
            activate_output=False
        )

        # Post-process reconstruction outputs
        reconstruction = {k: v[0].contiguous() for k, v in reconstruction_unactivated.items()}
        reconstruction["scaling"] = model.scaling_activation(reconstruction["scaling"])
        reconstruction["opacity"] = model.opacity_activation(reconstruction["opacity"])

        # Rendering loop
        world_view_transforms, full_proj_transforms, camera_centers = get_target_cameras()
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        loop_renders = []
        image_out_dir = os.path.join(output_dir, "rendered_images")
        os.makedirs(image_out_dir, exist_ok=True)

        for r_idx in range(world_view_transforms.shape[0]):
            image = render_predicted(
                reconstruction,
                world_view_transforms[r_idx].to(device),
                full_proj_transforms[r_idx].to(device),
                camera_centers[r_idx].to(device),
                background,
                model_cfg,
                focals_pixels=None
            )["render"]
            loop_renders.append(torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            imageio.imwrite(os.path.join(image_out_dir, f"render_{r_idx:04d}.png"), loop_renders[-1])

        # Export video and PLY
        loop_out_path = os.path.join(output_dir, "loop.mp4")
        imageio.mimsave(loop_out_path, loop_renders, fps=25)
        ply_out_path = os.path.join(output_dir, "mesh.ply")
        export_to_obj(reconstruction_unactivated, ply_out_path)

        return ply_out_path, loop_out_path, image_out_dir

def main():
    # Initialize Nvidia SMI for GPU monitoring
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using GPU 0

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Paths
    input_csv = "/lustre/fs1/home/cap6411.student4/Project/splatter-image/test_set.csv"
    input_images_dir = "/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/Objaverse_Dataset/rendered_images/"
    output_base_dir = "/lustre/fs1/home/cap6411.student4/Project/splatter-image/OUTPUT/"
    rembg_session = rembg.new_session()

    # Load model
    model, model_cfg = load_model(device)

    # Track total time and per-image time
    total_start_time = time.time()
    total_inference_time = 0
    total_images = 0

    # Process each image from the CSV
    with open(input_csv, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = row['Image1_Name']
            input_image_path = os.path.join(input_images_dir, image_name)
            
            if os.path.exists(input_image_path):
                output_dir = os.path.join(output_base_dir, os.path.splitext(image_name)[0])
                input_image = Image.open(input_image_path)

                # Preprocess image
                preprocessed_image = preprocess_image(input_image, rembg_session)

                # Track inference time per image
                start_time = time.time()

                # Reconstruct and export
                ply_path, video_path, image_path = reconstruct_and_export(
                    model, model_cfg, preprocessed_image, device, output_dir
                )

                # Calculate time taken for this inference
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                total_images += 1


                #print(f"Processed {image_name}:")
                #print(f"  PLY: {ply_path}, Video: {video_path}")
                
                #print(f"  Time taken for this image: {inference_time:.4f} seconds")


    # Log memory usage
    gpu_memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    max_gpu_memory_MB = gpu_memory_info.used / 1e6
    print(f"  GPU Memory Used: {max_gpu_memory_MB:.2f} MB")

    # Summary
    total_time_taken = time.time() - total_start_time
    avg_inference_time = total_inference_time / total_images if total_images > 0 else 0
    print(f"\nSummary:")
    print(f"  Total images processed: {total_images}")
    print(f"  Total inference time: {total_inference_time:.2f} seconds")
    print(f"  Average time per image: {avg_inference_time:.4f} seconds")
    print(f"  Total time taken (including I/O): {total_time_taken / 60:.2f} minutes")

if __name__ == "__main__":
    main()

#----------------------- PSNR, FID, LPIPS, CLIP-R calculation-------------

import torch
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
import lpips
import pandas as pd
from PIL import Image
import os
from rembg import remove
from metrics import compute_fid, compute_clip_r_precision,psnr

def evaluate_generated_images(base_path, real_image_path, test_csv_path, device='cuda', fid_batch_size=30):
    """
    Evaluate pre-generated images using FID, CLIP-R precision, PSNR, and LPIPS.
    
    Args:
        base_path: Base path to the generated images
        real_image_path: Path to the real images
        test_csv_path: Path to test set CSV
        device: Device to run evaluation on
        fid_batch_size: Batch size for FID calculation
    """
    # Load CSV
    test_df = pd.read_csv(test_csv_path)
    
    # FID preprocessing
    fid_preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Image resizing (to common resolution)
    resize_transform = transforms.Resize((256, 256))  # Resize to a common resolution
    
    # Initialize models
    clip_model = SentenceTransformer("clip-ViT-B-32")

    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    # Initialize lists for metrics
    real_images = []
    generated_images = []
    psnr_values = []
    lpips_values = []
    captions = []
    
    to_tensor = transforms.ToTensor()
    
    for _, row in test_df.iterrows():
        object_id = row['Object_ID']
        caption = row['Caption']
        captions.append(caption)
        
        # Load generated image
        gen_img_path = os.path.join(base_path, f"{object_id}_view_1", 'rendered_images/render_0000.png')
        if os.path.exists(gen_img_path):
            gen_img = Image.open(gen_img_path).convert('RGBA')  # Load as RGBA to ensure alpha channel exists if there is one
            gen_img = remove(gen_img)  # Remove the background
            gen_img = gen_img.convert('RGB')
            gen_img = resize_transform(gen_img)  # Resize generated image
            gen_tensor = to_tensor(gen_img)
            generated_images.append(gen_tensor)
            
            # Load real image
            real_img_path = os.path.join(real_image_path, f"{object_id}_view_1.png")
            if os.path.exists(real_img_path):
                real_img = Image.open(real_img_path).convert('RGB')
                real_img = resize_transform(real_img)  # Resize real image
                real_tensor = to_tensor(real_img)
                real_images.append(real_tensor)
                
                # Calculate PSNR
                psnr_value = psnr(gen_tensor, real_tensor)
                if not torch.isinf(torch.tensor(psnr_value)):
                    psnr_values.append(psnr_value)
                
                # Calculate LPIPS
                real_lpips = real_tensor.unsqueeze(0).to(device)
                gen_lpips = gen_tensor.unsqueeze(0).to(device)
                lpips_value = lpips_model(real_lpips, gen_lpips).item()
                lpips_values.append(lpips_value)
    
    # Convert to tensors
    real_images = torch.stack(real_images)
    generated_images = torch.stack(generated_images)
    
    # Preprocess for FID
    real_images_fid = torch.stack([fid_preprocess(img) for img in real_images])
    generated_images_fid = torch.stack([fid_preprocess(img) for img in generated_images])
    
    # Calculate FID
    fid_value = compute_fid(real_images_fid, generated_images_fid, batch_size=fid_batch_size, cuda=(device == 'cuda'))
    print(f"FID score: {fid_value:.4f}")
    
    # Calculate CLIP-R precision
    clip_r_text, clip_r_image, overall_clip_r = compute_clip_r_precision(
        clip_model, generated_images, captions, real_images
    )
    print(f"CLIP-R Precision (Text): {clip_r_text:.4f}")
    print(f"CLIP-R Precision (Image): {clip_r_image:.4f}")
    print(f"Overall CLIP-R Precision: {overall_clip_r:.4f}")
    
    # Calculate average metrics
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else None
    avg_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else None
    
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    
    return fid_value, clip_r_text, clip_r_image, overall_clip_r, avg_psnr, avg_lpips


# Define paths
real_image_path = "/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/Objaverse_Dataset/improved_images/"
test_csv_path = "/lustre/fs1/home/cap6411.student4/Project/splatter-image/test_set_with_captions.csv"
generated_image_folder = "/lustre/fs1/home/cap6411.student4/Project/splatter-image/OUTPUT/"

# Run evaluation
evaluate_generated_images(
    base_path=generated_image_folder,
    real_image_path=real_image_path,
    test_csv_path=test_csv_path,
    device='cuda'
)
