import os
import torch
import imageio.v2 as imageio  
import numpy as np
import torch.nn.functional as F


def read_image(image_path):
    img = imageio.imread(image_path) 
    img = (np.array(img) / 255.).astype(np.float32)  # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # Transpose to [channels, height, width]
    return img

# PSNR Calculation
class PSNR(object):
    def __call__(self, pred, gt):
        mse = torch.mean((pred - gt) ** 2) 
        if mse == 0:
            return float('inf') 
        return 10 * torch.log10(1 / mse)  # PSNR formula

# Function to calculate PSNR for all images
def calculate_psnr(estim_dir, gt_dir):
    psnr_calculator = PSNR()
    psnr_values = []


    estim_fnames = os.listdir(estim_dir)
    gt_fnames = os.listdir(gt_dir)

    # taking only object id
    estim_fnames = {f[:15]: f for f in estim_fnames} 
    gt_fnames = {f[:15]: f for f in gt_fnames}  

    
    for gt_base, gt_fname in gt_fnames.items():
        if gt_base in estim_fnames:
            estim_fname = estim_fnames[gt_base]  

            estim_path = os.path.join(estim_dir, estim_fname)
            gt_path = os.path.join(gt_dir, gt_fname)

         
            estim_img = read_image(estim_path)
            gt_img = read_image(gt_path)

          
            estim_tensor = torch.Tensor(estim_img).cuda() if torch.cuda.is_available() else torch.Tensor(estim_img)
            gt_tensor = torch.Tensor(gt_img).cuda() if torch.cuda.is_available() else torch.Tensor(gt_img)

            # Calculate PSNR for this pair of images
            psnr_value = psnr_calculator(estim_tensor, gt_tensor)  # Remove .item() call here
            psnr_values.append(psnr_value)

            #print(f"PSNR for {gt_fname} and {estim_fname}: {psnr_value} dB")
       # else:
           # print(f"No matching generated image for {gt_fname}")


    return psnr_values


generated_images_dir = r"C:\SURANADI\UCF\CV_Systems\Project\objaverse\2d_images\gen_test"
ground_truth_images_dir =r"C:\SURANADI\UCF\CV_Systems\Project\objaverse\2d_images\gt_test"

# Calculate PSNR 
psnr_results = calculate_psnr(generated_images_dir, ground_truth_images_dir)

# Filter out infinite PSNR values before calculating the average
finite_psnr_results = [psnr for psnr in psnr_results if not torch.isinf(torch.tensor(psnr))]

if finite_psnr_results:
    average_psnr = sum(finite_psnr_results) / len(finite_psnr_results)
    print(f"\nAverage PSNR (excluding inf): {average_psnr} dB")
else:
    print("No finite PSNR values found for averaging.")

