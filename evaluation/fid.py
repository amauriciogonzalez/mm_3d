import os
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm




# Set paths to two folders containing the real and generated images
path1 = "path_for_ground_truth_imgs" 

path2 = "path_for_generated_imgs"  # Folder containing generated images

# Batch size for processing images
batch_size = 2
dims = 2048
use_cuda = torch.cuda.is_available()

# Preprocess transformation for InceptionV3
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images(path):
    """Loads all images from a folder and applies preprocessing."""
    files = list(os.path.join(path, f) for f in os.listdir(path) if f.endswith(('png', 'jpg')))
    images = []
    for file in files:
        img = Image.open(file).convert('RGB')
        images.append(preprocess(img).unsqueeze(0))
    return torch.cat(images)

def get_activations(images, model, batch_size=50, dims=2048, cuda=True):
    """Calculates the activations of the pool_3 layer for all images."""
    model.eval()

    if batch_size > len(images):
        batch_size = len(images)

    pred_arr = np.empty((len(images), dims))
    
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]

        if cuda:
            batch = batch.cuda()

        with torch.no_grad():
            pred = model(batch)

        
        pred_arr[i:i+batch_size] = pred.cpu().numpy()

    return pred_arr


def calculate_activation_statistics(images, model, batch_size=50, dims=2048, cuda=True):
    """Calculates the mean and covariance of activations."""
    act = get_activations(images, model, batch_size, dims, cuda)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculates the Frechet Distance between two multivariate Gaussians."""
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])).dot(sigma2 + eps * np.eye(sigma2.shape[0])))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(real_images, generated_images, batch_size, cuda=True, dims=2048):
    """Calculates the FID score between two sets of images."""
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove the final classification layer
    if cuda:
        model.cuda()

    mu1, sigma1 = calculate_activation_statistics(real_images, model, batch_size, dims, cuda)
    mu2, sigma2 = calculate_activation_statistics(generated_images, model, batch_size, dims, cuda)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

if __name__ == '__main__':
    # Load real and generated images
    gt_images = load_images(path1)
    generated_images = load_images(path2)

    # Calculate FID
    fid_value = calculate_fid(gt_images, generated_images, batch_size, use_cuda, dims)
    print(f'FID: {fid_value}')
