from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import pandas as pd
import numpy as np

def load_text_prompts(csv_file):
    # Load prompts from a CSV file
    df = pd.read_csv(csv_file)
    return df['text'].tolist()

def encode_image(model, image_path):
    # Encode an image using the CLIP model
    return model.encode(Image.open(image_path))

def compute_average_cosine_similarity(model, image_folder, text_prompts):
    all_cosine_scores = []

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Encode the image
            img_emb = encode_image(model, image_path)
            for text in text_prompts:
                # Encode text descriptions
                text_emb = model.encode([text])
                # Compute cosine similarity
                cos_score = util.cos_sim(img_emb, text_emb)
                all_cosine_scores.append(cos_score[0][0].cpu().numpy())

    # Calculate the average cosine similarity
    if all_cosine_scores:
        average_score = np.mean(all_cosine_scores)
        print(f"Average Cosine Similarity for the validation set: {average_score}")
    else:
        print("No cosine scores computed.")

if __name__ == '__main__':
    # Load CLIP 
    model = SentenceTransformer("clip-ViT-B-32")
    
    # Load text prompts from CSV
    text_prompts = load_text_prompts(r'C:\SURANADI\UCF\CV_Systems\Project\view_2d\prompt_list.csv') 
    
    # Specify the image folder path
    image_folder = r'C:\SURANADI\UCF\CV_Systems\Project\view_2d\output_images\new'
    
    # Compute average cosine similarity
    compute_average_cosine_similarity(model, image_folder, text_prompts)
