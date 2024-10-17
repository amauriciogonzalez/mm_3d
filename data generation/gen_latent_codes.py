import torch
import os
from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
xm = load_model('transmitter', device=device)


folder_path = r"C:\SURANADI\Project\sample_objs\sample_objs"
latent_dir = r"C:\SURANADI\shap-e\LATENT_CODES"

# Ensure the directory for saving latent codes exists
os.makedirs(latent_dir, exist_ok=True)


for obj_file in os.listdir(folder_path):
    if obj_file.endswith('.obj'):
        
        model_path = os.path.join(folder_path, obj_file)
       
        # Load or create the multimodal batch
        batch = load_or_create_multimodal_batch(
            device,
            model_path=model_path,
            mv_light_mode="basic",
            mv_image_size=256,
            cache_dir="example_data/cactus/cached",
            verbose=True,  
        )

        # Encode the batch to get the latent code
        with torch.no_grad():
            latent = xm.encoder.encode_to_bottleneck(batch)

            # Define the save path for latent code based on object name
            obj_name = os.path.splitext(obj_file)[0] 
            latent_save_path = os.path.join(latent_dir, f"{obj_name}.pt")

            # Save the latent code to a file
            torch.save(latent, latent_save_path)
            print(f"Latent code for {obj_name} saved to {latent_save_path}")
