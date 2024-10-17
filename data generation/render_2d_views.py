import os
import argparse
import glob
import time
import subprocess
from tqdm import tqdm
import signal
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('render_log.txt'),
        logging.StreamHandler()
    ]
)

def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Renders glbs')
    parser.add_argument(
        '--save_folder', type=str, 
        default='/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/Objaverse_Dataset/im/',
        help='Path for saving rendered images'
    )
    parser.add_argument(
        '--folder_assets', type=str,
        default='/lustre/fs1/home/cap6411.student4/.objaverse/hf-objaverse-v1/glbs/',
        help='Path to downloaded 3D assets'
    )
    parser.add_argument(
        '--blender_root', type=str,
        default='/lustre/fs1/home/cap6411.student4/Project/mvs_objaverse/Objaverse_Dataset/blender-4.2.2-linux-x64/blender',
        help='Path to Blender executable'
    )
    parser.add_argument(
        '--timeout', type=int,
        default=300,  # 5 minutes timeout
        help='Timeout for each rendering operation in seconds'
    )
    return parser.parse_args()

def run_blender_with_timeout(cmd, timeout):
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode != 0:
                logging.error(f"Blender process failed with return code {process.returncode}")
                logging.error(f"stderr: {stderr.decode()}")
                return False
            return True
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            logging.error(f"Process timed out after {timeout} seconds")
            return False
            
    except Exception as e:
        logging.error(f"Error running Blender: {str(e)}")
        return False

def process_glb_file(glb_path, opt):
    object_name = os.path.basename(glb_path).replace('.glb', '')
    
    # Check if renders already exist
    expected_files = [
        os.path.join(opt.save_folder, f"{object_name}_view_{i + 1}.png")
        for i in range(2)
    ]
    if all(os.path.exists(f) for f in expected_files):
        logging.info(f"Skipping {object_name} - already rendered")
        return True

    render_cmd = f'{opt.blender_root} -b -P rendering/render_blender.py -- --obj {glb_path} --output {opt.save_folder} --views 2 --resolution 400 > tmp.out'
    
    logging.info(f"Starting render for {object_name}")
    success = run_blender_with_timeout(render_cmd, opt.timeout)
    
    if not success:
        logging.error(f"Failed to render {object_name}")
        return False
    
    time.sleep(2)  # Brief wait for file system
    
    # Rename files
    for i in range(2):
        original_file = os.path.join(opt.save_folder, f"{i:03}.png")
        new_file = os.path.join(opt.save_folder, f"{object_name}_view_{i + 1}.png")
        
        try:
            if os.path.exists(original_file):
                os.rename(original_file, new_file)
                logging.info(f"Renamed: {original_file} to {new_file}")
            else:
                logging.warning(f"Warning: {original_file} not found")
                return False
        except Exception as e:
            logging.error(f"Error renaming file: {str(e)}")
            return False
    
    return True

def main():
    opt = setup_argument_parser()
    
    # Ensure output directory exists
    os.makedirs(opt.save_folder, exist_ok=True)
    
    # Get all folders containing .glb files
    data = sorted(glob.glob(f"{opt.folder_assets}/*/"))
    
    # Count total files
    total_glb_files = sum([len(glob.glob(folder + "*.glb")) for folder in data])
    
    failed_objects = []
    
    with tqdm(total=total_glb_files, desc="Processing objects") as pbar:
        for folder_path in data:
            glb_files = sorted(glob.glob(folder_path + "*.glb"))
            
            if not glb_files:
                logging.info(f"No .glb files found in {folder_path}")
                continue
            
            for glb_path in glb_files:
                success = process_glb_file(glb_path, opt)
                if not success:
                    failed_objects.append(glb_path)
                pbar.update(1)
    
    # Report results
    logging.info("Rendering complete")
    logging.info(f"Total objects processed: {total_glb_files}")
    logging.info(f"Failed objects: {len(failed_objects)}")
    
    if failed_objects:
        with open('failed_renders.txt', 'w') as f:
            for obj in failed_objects:
                f.write(f"{obj}\n")
        logging.info("Failed objects list saved to failed_renders.txt")

if __name__ == "__main__":
    main()