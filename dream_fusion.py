import argparse
import importlib
import logging
import os
import sys
import time
import traceback

import torch.cuda

import matplotlib.pyplot as plt


# Function to evaluate Dream Fusion model on the Objaverse dataset
def evaluate_objaverse_instance_DF(model, dataloader, evaluate=False, device='cuda', fid_batch_size=30, text_ablation_step=0):
    """
    Evaluate a model using the Objaverse dataset with metrics like FID, CLIP-R precision, and PSNR.

    Args:
        model: Ignored. Placeholder for the model to be evaluated.
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
   
    class ColoredFilter(logging.Filter):
        """
        A logging filter to add color to certain log levels.
        """

        RESET = "\033[0m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"

        COLORS = {
            "WARNING": YELLOW,
            "INFO": GREEN,
            "DEBUG": BLUE,
            "CRITICAL": MAGENTA,
            "ERROR": RED,
        }

        RESET = "\x1b[0m"

        def __init__(self):
            super().__init__()

        def filter(self, record):
            if record.levelname in self.COLORS:
                color_start = self.COLORS[record.levelname]
                record.levelname = f"{color_start}[{record.levelname}]"
                record.msg = f"{record.msg}{self.RESET}"
            return True


    def load_custom_module(module_path):
        module_name = os.path.basename(module_path)
        if os.path.isfile(module_path):
            sp = os.path.splitext(module_path)
            module_name = sp[0]
        try:
            if os.path.isfile(module_path):
                module_spec = importlib.util.spec_from_file_location(
                    module_name, module_path
                )
            else:
                module_spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(module_path, "__init__.py")
                )

            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
            return True
        except Exception as e:
            print(traceback.format_exc())
            print(f"Cannot import {module_path} module for custom nodes:", e)
            return False


    def load_custom_modules():
        node_paths = ["custom"]
        node_import_times = []
        for custom_node_path in node_paths:
            possible_modules = os.listdir(custom_node_path)
            if "__pycache__" in possible_modules:
                possible_modules.remove("__pycache__")

            for possible_module in possible_modules:
                module_path = os.path.join(custom_node_path, possible_module)
                if (
                    os.path.isfile(module_path)
                    and os.path.splitext(module_path)[1] != ".py"
                ):
                    continue
                if module_path.endswith("_disabled"):
                    continue
                time_before = time.perf_counter()
                success = load_custom_module(module_path)
                node_import_times.append(
                    (time.perf_counter() - time_before, module_path, success)
                )

        if len(node_import_times) > 0:
            print("\nImport times for custom modules:")
            for n in sorted(node_import_times):
                if n[2]:
                    import_message = ""
                else:
                    import_message = " (IMPORT FAILED)"
                print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
            print()
            
    args = argparse.Namespace(config='configs/dreamfusion-sd-eff.yaml', gpu='0', train=True, validate=True, test=False, export=False, gradio=False, verbose=False, typecheck=False)


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        
        
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook

        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            if not args.gradio:
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                handler.addFilter(ColoredFilter())
            else:
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    load_custom_modules()
    

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

    cameras = create_pan_cameras(64, model.device)
    viewpoint_indices = [0, 4, 8, 12]

    for batch_idx, (images1, images2, captions, object_names, latent_codes) in enumerate(dataloader):
        # Move tensors to the specified device
        images2 = images2.to(device)
        latent_codes = latent_codes.to(device)
        print("captions")
        print(captions[0])
        extras = [f'system.prompt_processor.prompt={captions[0]}']
        print(extras)
        cfg: ExperimentConfig
        cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

        # set a different seed for each device
        pl.seed_everything(cfg.seed + get_rank(), workers=True)
        dm = threestudio.find(cfg.data_type)(cfg.data)


        system: BaseSystem = threestudio.find(cfg.system_type)(
            cfg.system, resumed=cfg.resume is not None
        )

        system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

        
        callbacks = []
        if args.train:
            callbacks += [
                ModelCheckpoint(
                    dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
                ),
                LearningRateMonitor(logging_interval="step"),
                CodeSnapshotCallback(
                    os.path.join(cfg.trial_dir, "code"), use_version=False
                ),
                ConfigSnapshotCallback(
                    args.config,
                    cfg,
                    os.path.join(cfg.trial_dir, "configs"),
                    use_version=False,
                ),
            ]
           
            callbacks += [CustomProgressBar(refresh_rate=1)]

        def write_to_text(file, lines):
            with open(file, "w") as f:
                for line in lines:
                    f.write(line + "\n")

        loggers = []
        if args.train:
            # make tensorboard logging dir to suppress warning
            rank_zero_only(
                lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
            )()
            loggers += [
                TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
                CSVLogger(cfg.trial_dir, name="csv_logs"),
            ] + system.get_loggers()
            rank_zero_only(
                lambda: write_to_text(
                    os.path.join(cfg.trial_dir, "cmd.txt"),
                    ["python " + " ".join(sys.argv), str(args)],
                )
            )()

        trainer = Trainer(
            callbacks=callbacks,
            logger=loggers,
            inference_mode=False,
            accelerator="gpu",
            devices=devices,
            **cfg.trainer,
        )

        def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
            if ckpt_path is None:
                return
            ckpt = torch.load(ckpt_path, map_location="cpu")
            system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

        if args.train:
            print("here3")
            start = time.time()
            trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
            print(f"time is {time.time() - start}")
            print("here4")
            trainer.test(system, datamodule=dm)
        elif args.validate:
            # manually set epoch and global_step as they cannot be automatically resumed
            set_system_status(system, cfg.resume)
            trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
        elif args.test:
            # manually set epoch and global_step as they cannot be automatically resumed
            set_system_status(system, cfg.resume)
            trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
        elif args.export:
            set_system_status(system, cfg.resume)
            trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)


  
        allocated_memory = torch.cuda.memory_allocated()

        # Print memory in MB for easier readability
        print(f"Allocated Memory: {allocated_memory / 1024 ** 2:.2f} MB")
        # Time the inference for this batch
        batch_start_time = time.time()

        print(len(trainer.model.images))
        # Generate the 3D projections
        with torch.no_grad():
            # Forward pass to get predicted latents

            predicted_image_batches_pil = []
            real_image_batches_pil = []
            from torchvision.transforms import Resize
            resize = Resize((64, 64), antialias=True)


            allocated_memory = torch.cuda.memory_allocated()


            # Print memory in MB for easier readability
            print(f"Allocated Memory: {allocated_memory / 1024 ** 2:.2f} MB")
           
            
            predicted_image_batches_pil = [resize(img.permute(2,0,1)).cpu() for img in trainer.model.images]

            allocated_memory = torch.cuda.memory_allocated()

            # Print memory in MB for easier readability
            print(f"Allocated Memory: {allocated_memory / 1024 ** 2:.2f} MB")
                    

            for i, pred_latent in enumerate(latent_codes):
                real_images = decode_latent_images(model.xm, latent_codes[i], cameras, rendering_mode='nerf')
                real_images = [real_images[i] for i in viewpoint_indices]
                # Append the entire set of viewpoints for FID and PSNR calculation
                real_image_batches_pil.append(real_images)
                # Extract a single viewpoint (e.g., the first viewpoint)
                selected_view_pred_images.append(ToPILImage()(predicted_image_batches_pil[0][0]))
                selected_view_real_images.append(real_images[0])


        del trainer
        threestudio.info("DELETE DIR")
        print(os.getcwd())
        print(os.listdir("./outputs/dreamfusion-sd"))
        os.system("rm -r ./outputs/dreamfusion-sd")

        
        torch.cuda.empty_cache()

        if(evaluate):
            # Record the time taken for this batch's inference
            batch_inference_time = time.time() - batch_start_time
            inference_times.append(batch_inference_time)

            # Collect real and generated images for FID calculation
            real_image_batches.append(torch.stack([transforms.ToTensor()(img) for batch in real_image_batches_pil for img in batch]))
            generated_image_batches.append(torch.stack(predicted_image_batches_pil))


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
                    pred_img_tensor = pred_img.unsqueeze(0).to(device)
                    lpips_value = lpips_model(real_img_tensor, pred_img_tensor).item()
                    lpips_values.append(lpips_value)

            

            # Get memory usage in bytes, then convert to MB
            memory_usage = process.memory_info().rss / 1024 ** 2
            print(f"Memory allocated to CPU RAM: {memory_usage:.2f} MB")

    if(evaluate):
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
        print(selected_view_pred_images)
        print(selected_view_real_images)
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
