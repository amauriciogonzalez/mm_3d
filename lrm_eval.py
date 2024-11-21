import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import os
import argparse
import sys
import mcubes
import trimesh
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from rembg import remove
import time
import nvidia_smi
from sklearn.model_selection import train_test_split
from skimage import measure
from scipy.ndimage import rotate
from scipy import linalg
from moviepy.editor import VideoFileClip

import pyrender
import csv
from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, "../shap_e")

from shap_e.models.download import load_model
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

from openlrm.datasets.cam_utils import build_camera_principle, build_camera_standard, surrounding_views_linspace, create_intrinsics
from openlrm.utils.logging import configure_logger
from openlrm.runners import REGISTRY_RUNNERS
from openlrm.utils.video import images_to_video
from openlrm.utils.hf_hub import wrap_model_hub

logger = get_logger(__name__)

class LRMInferrer():

    EXP_TYPE: str = 'lrm'

    def __init__(self):

        torch._dynamo.config.disable = True
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.cfg = parse_configs()
        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )

        self.model = self._build_model(self.cfg).to(self.device)

    def _build_model(self, cfg):
        from openlrm.models import model_dict
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model

    def _default_source_camera(self, dist_to_center: float = 2.0, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center],
            [0, 1, 0, 0],
        ]], dtype=torch.float32, device=device)
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, n_views: int, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(n_views=n_views, device=device)
        render_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def infer_planes(self, image: torch.Tensor, source_cam_dist: float):
        N = image.shape[0]
        source_camera = self._default_source_camera(dist_to_center=source_cam_dist, batch_size=N, device=self.device)
        planes = self.model.forward_planes(image, source_camera)
        assert N == planes.shape[0]
        return planes

    def infer_video(self, planes: torch.Tensor, frame_size: int, render_size: int, render_views: int, render_fps: int, dump_video_path: str):
        N = planes.shape[0]
        render_cameras = self._default_render_cameras(n_views=render_views, batch_size=N, device=self.device)
        render_anchors = torch.zeros(N, render_cameras.shape[1], 2, device=self.device)
        render_resolutions = torch.ones(N, render_cameras.shape[1], 1, device=self.device) * render_size
        render_bg_colors = torch.ones(N, render_cameras.shape[1], 1, device=self.device, dtype=torch.float32) * 1.

        frames = []
        for i in range(0, render_cameras.shape[1], frame_size):
            frames.append(
                self.model.synthesizer(
                    planes=planes,
                    cameras=render_cameras[:, i:i+frame_size],
                    anchors=render_anchors[:, i:i+frame_size],
                    resolutions=render_resolutions[:, i:i+frame_size],
                    bg_colors=render_bg_colors[:, i:i+frame_size],
                    region_size=render_size,
                )
            )
        # merge frames
        frames = {
            k: torch.cat([r[k] for r in frames], dim=1)
            for k in frames[0].keys()
        }
        # dump
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        for k, v in frames.items():
            if k == 'images_rgb':
                images_to_video(
                    images=v[0],
                    output_path=dump_video_path,
                    fps=render_fps,
                    gradio_codec=self.cfg.app_enabled,
                )

    def infer_mesh(self, planes: torch.Tensor, mesh_size: int, mesh_thres: float, dump_mesh_path: str):
        grid_out = self.model.synthesizer.forward_grid(
            planes=planes,
            grid_size=mesh_size,
        )
        
        vtx, faces = mcubes.marching_cubes(grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), mesh_thres)
        vtx = vtx / (mesh_size - 1) * 2 - 1

        vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=self.device).unsqueeze(0)
        vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
        vtx_colors = (vtx_colors * 255).astype(np.uint8)
        
        mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

        # dump
        os.makedirs(os.path.dirname(dump_mesh_path), exist_ok=True)
        mesh.export(dump_mesh_path)

    def infer_single(self, image_path: str, source_cam_dist: float, export_video: bool, export_mesh: bool, dump_video_path: str, dump_mesh_path: str):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_views = self.cfg.render_views
        render_fps = self.cfg.render_fps
        mesh_size = self.cfg.mesh_size
        mesh_thres = self.cfg.mesh_thres
        frame_size = self.cfg.frame_size
        source_cam_dist = self.cfg.source_cam_dist if source_cam_dist is None else source_cam_dist

        # prepare image: [1, C_img, H_img, W_img], 0-1 scale
        image = torch.from_numpy(np.array(Image.open(image_path))).to(self.device)
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        if image.shape[1] == 4:  # RGBA
            image = image[:, :3, ...] * image[:, 3:, ...] + (1 - image[:, 3:, ...])
        image = torch.nn.functional.interpolate(image, size=(source_size, source_size), mode='bicubic', align_corners=True)
        image = torch.clamp(image, 0, 1)

        with torch.no_grad():
            planes = self.infer_planes(image, source_cam_dist=source_cam_dist)

            results = {}
            if export_video:
                frames = self.infer_video(planes, frame_size=frame_size, render_size=render_size, render_views=render_views, render_fps=render_fps, dump_video_path=dump_video_path)
                results.update({
                    'frames': frames,
                })
            if export_mesh:
                mesh = self.infer_mesh(planes, mesh_size=mesh_size, mesh_thres=mesh_thres, dump_mesh_path=dump_mesh_path)
                results.update({
                    'mesh': mesh,
                })

    def infer(self):

        image_paths = []
        if os.path.isfile(self.cfg.image_input):
            omit_prefix = os.path.dirname(self.cfg.image_input)
            image_paths.append(self.cfg.image_input)
        else:
            omit_prefix = self.cfg.image_input
            for root, dirs, files in os.walk(self.cfg.image_input):
                for file in files:
                    if file.endswith('.png'):
                        image_paths.append(os.path.join(root, file))
            image_paths.sort()

        # alloc to each DDP worker
        image_paths = image_paths[self.accelerator.process_index::self.accelerator.num_processes]

        for image_path in tqdm(image_paths, disable=not self.accelerator.is_local_main_process):

            # prepare dump paths
            image_name = os.path.basename(image_path)
            uid = image_name.split('.')[0]
            subdir_path = os.path.dirname(image_path).replace(omit_prefix, '')
            subdir_path = subdir_path[1:] if subdir_path.startswith('/') else subdir_path
            dump_video_path = os.path.join(
                self.cfg.video_dump,
                subdir_path,
                f'{uid}.mov',
            )
            dump_mesh_path = os.path.join(
                self.cfg.mesh_dump,
                subdir_path,
                f'{uid}.ply',
            )

            self.infer_single(
                image_path,
                source_cam_dist=None,
                export_video=self.cfg.export_video,
                export_mesh=self.cfg.export_mesh,
                dump_video_path=dump_video_path,
                dump_mesh_path=dump_mesh_path,
            )


def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--infer', type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get('APP_INFER') is not None:
        args.infer = os.environ.get('APP_INFER')
    if os.environ.get('APP_MODEL_NAME') is not None:
        cli_cfg.model_name = os.environ.get('APP_MODEL_NAME')

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(cfg_train.experiment.parent, cfg_train.experiment.child, os.path.basename(cli_cfg.model_name).split('_')[-1])
        cfg.video_dump = os.path.join("exps", 'videos', _relative_path)
        cfg.mesh_dump = os.path.join("exps", 'meshes', _relative_path)

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault('video_dump', os.path.join("dumps", cli_cfg.model_name, 'videos'))
        cfg.setdefault('mesh_dump', os.path.join("dumps", cli_cfg.model_name, 'meshes'))

    cfg.merge_with(cli_cfg)

    """
    [required]
    model_name: str
    image_input: str
    export_video: bool
    export_mesh: bool

    [special]
    source_size: int
    render_size: int
    video_dump: str
    mesh_dump: str

    [default]
    render_views: int
    render_fps: int
    mesh_size: int
    mesh_thres: float
    frame_size: int
    logger: str
    """

    cfg.setdefault('logger', 'INFO')

    # assert not (args.config is not None and args.infer is not None), "Only one of config and infer should be provided"
    assert cfg.model_name is not None, "model_name is required"
    if not os.environ.get('APP_ENABLED', None):
        assert cfg.image_input is not None, "image_input is required"
        assert cfg.export_video or cfg.export_mesh, \
            "At least one of export_video or export_mesh should be True"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg



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

    def get_sample_by_object_id(self, object_id):
        """Fetch a specific sample by its object_id."""
        if object_id not in self.valid_object_ids:
            raise ValueError(f"Object ID {object_id} is not valid.")
        
        # Load images, caption, object name, and latent code as before
        image2_path = os.path.join(self.image_dir, f"{object_id}_view_2.png")
        image2 = Image.open(image2_path).convert('RGB')
        
        if self.transform:
            image2 = self.transform(image2)
        
        caption = self.caption_data[object_id]
        object_name = self.object_names[object_id].replace("_", " ").capitalize()
        latent_code_path = os.path.join(self.latent_code_dir, f"{object_id}.pt")
        latent_code = torch.load(latent_code_path)
        
        return image2_path, image2, caption, object_name, latent_code

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
            #brightness_factor = 20
            #saturation_factor = 2
            #threshold = 0.05
            #image1 = adjust_image(image1, brightness_factor, saturation_factor, threshold)
            #image2 = adjust_image(image2, brightness_factor, saturation_factor, threshold)

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

        return image2_path, image2, caption, object_name, latent_code




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


def evaluate_model_with_views(LRM, image_path):
    try:
        # Run the LRM model to generate the .ply mesh file
        LRM.infer_single(
            image_path,
            source_cam_dist=None,
            export_video=False,
            export_mesh=True,
            dump_video_path="dumps/generated_video.mov",
            dump_mesh_path="dumps/generated_mesh.ply"
        )

        # Load the mesh file generated by LRM as a Trimesh object
        trimesh_mesh = trimesh.load("dumps/generated_mesh.ply")
        
    except Exception as e:
        # If infer_single or loading the mesh fails, return four black 64x64 images
        black_image = Image.new("RGB", (64, 64), color=(0, 0, 0))
        return [black_image, black_image, black_image, black_image]

    # Define four different camera viewpoints (yaw angles in radians)
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    images = []
    camera_distance = 2.5  # Distance of camera from the object

    # Create a renderer with 64x64 viewport dimensions
    renderer = pyrender.OffscreenRenderer(viewport_width=64, viewport_height=64)

    for angle in angles:
        # Setup a fresh scene for each viewpoint
        scene = pyrender.Scene()
        render_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)  # Create a Pyrender mesh from the Trimesh mesh
        scene.add(render_mesh)

        # Set the camera at the specific viewpoint
        camera_pose = np.array([
            [np.cos(angle), 0, np.sin(angle), camera_distance * np.sin(angle)],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), camera_distance * np.cos(angle)],
            [0, 0, 0, 1]
        ])
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)

        # Render the scene and capture the color buffer
        color, _ = renderer.render(scene)

        # Convert color buffer to PIL Image and ensure it is resized to 64x64
        image = Image.fromarray(color).resize((64, 64))
        images.append(image)

    renderer.delete()  # Free up renderer resources

    return images





def evaluate_objaverse_instance(LRM, dataloader, device='cuda', fid_batch_size=30, text_ablation_step=0):
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

    xm = load_model('transmitter', device=device)

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

    # Initialize LPIPS model
    import lpips
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    lpips_values = []

    start_time = time.time()  # Start timing

    all_descriptions = []
    all_input_images = []
    psnr_values = []
    inference_times = []


    # Initialize lists for storing the selected viewpoint images for CLIP-R precision
    selected_view_pred_images = []
    selected_view_real_images = []

    cameras = create_pan_cameras(64, device)
    viewpoint_indices = [0, 4, 8, 12]

    for batch_idx, (images2_paths, images2, captions, object_names, latent_codes) in enumerate(dataloader):
        # Move tensors to the specified device
        images2 = images2.to(device)
        latent_codes = latent_codes.to(device)

        # Time the inference for this batch
        batch_start_time = time.time()

        # Generate the 3D projections
        with torch.no_grad():
            predicted_image_batches_pil = []
            real_image_batches_pil = []
            for i, image_path in enumerate(images2_paths):
                
                video_path = f"dumps/generated_video.mov"
                # Perform inference and generate video
                LRM.infer_single(
                    image_path,
                    source_cam_dist=None,
                    export_video=True,
                    export_mesh=False,
                    dump_video_path=video_path,
                    dump_mesh_path="dumps/generated_mesh.ply"
                )

                predicted_images = convert_video_to_frames(video_path)
                real_images = decode_latent_images(xm, latent_codes[i], cameras, rendering_mode='nerf')
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

         # LPIPS Calculation for the current batch
        for real_imgs, pred_imgs in zip(real_image_batches_pil, predicted_image_batches_pil):
            for real_img, pred_img in zip(real_imgs, pred_imgs):
                real_img_tensor = transforms.ToTensor()(real_img).unsqueeze(0).to(device)
                pred_img_tensor = transforms.ToTensor()(pred_img).unsqueeze(0).to(device)
                lpips_value = lpips_model(real_img_tensor, pred_img_tensor).item()
                lpips_values.append(lpips_value)


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

    # Calculate the average LPIPS
    average_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else None
    print(f"Average LPIPS: {average_lpips:.4f}")

    # Calculate the average inference time per sample
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time per batch: {average_inference_time:.4f} seconds")

    return fid_value, average_clip_r_text, average_clip_r_image, overall_clip_r, average_psnr, average_inference_time, average_lpips


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


def convert_video_to_gif(video_path, gif_path):
    """
    Convert a .mov video file to a .gif file.

    Args:
        video_path (str): Path to the .mov video file.
        gif_path (str): Path to save the output .gif file.
    """
    clip = VideoFileClip(video_path)
    clip.write_gif(gif_path, fps=10)  # Adjust FPS as needed for smoother GIFs
    clip.close()


def convert_video_to_frames(video_path):
    """
    Convert four equally spaced frames from a video file to a GIF.

    Args:
        video_path (str): Path to the video file.
        gif_path (str): Path to save the output .gif file.
        fps (float): Frames per second for the GIF.
    """
    # Load the video clip
    clip = VideoFileClip(video_path)

    # Calculate the times for four equally spaced frames
    duration = clip.duration
    times = [duration * i / 4 for i in range(4)]  # Four timestamps equally spaced

    # Extract frames at the calculated times
    frames = [clip.get_frame(t) for t in times]
    images = [Image.fromarray(frame).resize((64, 64)) for frame in frames]

    # Close the video clip to release resources
    clip.close()

    return images


def convert_video_to_frames_to_gif(video_path, gif_path, fps=2.5):
    """
    Convert four equally spaced frames from a video file to a GIF.

    Args:
        video_path (str): Path to the video file.
        gif_path (str): Path to save the output .gif file.
        fps (float): Frames per second for the GIF.
    """
    images = convert_video_to_frames(video_path)

    # Calculate duration per frame in milliseconds for the specified FPS
    duration_per_frame = int(1000 / fps)

    # Save frames as a GIF
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_per_frame,
        loop=0
    )


def save_images_as_gif(images, gif_path, fps=2.5):
    """
    Save a list of PIL images as a GIF with a specified frame rate (FPS).
    
    Args:
        images (list): List of PIL Image objects.
        gif_path (str): Path where the GIF should be saved.
        fps (float): Frames per second for the GIF.
    """
    # Calculate duration per frame in milliseconds
    duration = int(1000 / fps)  # duration per frame in ms

    # Save images as a GIF
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def perform_inference_on_specific_samples(dataset, LRM, object_ids, use_generated_views=False):
    for object_id in object_ids:
        image2_path, image2, caption, object_name, latent_code = dataset.get_sample_by_object_id(object_id)

        video_path = f"dumps/generated_video.mov"
        gif_path = f"dumps/generated_video_{object_id}.gif"


        if not use_generated_views:
            # Perform inference and generate video
            LRM.infer_single(
                image2_path,
                source_cam_dist=None,
                export_video=True,
                export_mesh=False,
                dump_video_path=video_path,
                dump_mesh_path="dumps/generated_mesh.ply"
            )

            # Convert the video to a GIF
            convert_video_to_frames_to_gif(video_path, gif_path)

        else:
            # Generate views for the sample and obtain four images
            images = evaluate_model_with_views(LRM, image2_path)


            # Convert the images to a GIF
            save_images_as_gif(images, gif_path, fps=2.5)

    


        



def main():
    print("Initializing Objaverse...")

    image_dir = "/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/Objaverse_Dataset/improved_images/"
    caption_csv = '/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/objaverse_csv.csv'
    latent_code_dir = "/lustre/fs1/home/cap6411.student3/final_project/Cap3D/Cap3D_latentcodes/"
    transform = transforms.ToTensor()

    dataset = ObjaverseDataset(
        image_dir=image_dir,
        caption_csv=caption_csv,
        latent_code_dir=latent_code_dir,
        transform=transform
    )

    # Leave num_samples=0 to load all samples!
    train_loader, test_loader = create_train_test_split(dataset, test_size=0.05, batch_size=8, num_samples=0)

    LRM = LRMInferrer()

    evaluate_objaverse_instance(LRM, test_loader)

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
    #perform_inference_on_specific_samples(dataset, LRM, specific_object_ids, use_generated_views=False)
    


if __name__ == "__main__":
    main()