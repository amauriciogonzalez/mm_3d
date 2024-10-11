# MM-Shap-E: A Multimodal Text-and-Image-to-3D Pipeline

*This README is still in progress and is subject to change*


## Setup

1. **Clone the Repository**:

```bash
git clone https://github.com/amauriciogonzalez/mm_3d.git
cd mm_3d
```


2. **(Optional) Create and Activate a Virtual Environment**:

```bash
# Windows
virtualenv env
.\env\Scripts\activate

# Linux/Mac
python -m venv env
source env/bin/activate
```
    
3. **Install Requirements**:

```bash   
pip install -r requirements.txt
```



## Arguments (mm_shap_e.py)

### General

- `--dataset`: Specify the dataset to use for training, evaluation, or demo. Compatible options:
    - [`text2shape`](http://text2shape.stanford.edu/)
    - [`objaverse`](https://objaverse.allenai.org/)
- `--n`: Number of samples to use during training/evaluation/demo (0 for all samples).

### Data Paths

- `--t2s_nrrd_dir`: Directory for Text2Shape's NRRD voxelization files.
- `--t2s_csv_path`: CSV path mapping object IDs to captions for Text2Shape.
- `--obja_img_dir`: Directory for Objaverse's rendered images (`<object_id>_view_1.png`, `<object_id>_view_2.png`).
- `--obja_csv_path`: CSV path mapping object IDs to captions for Objaverse.
- `--obja_latent_dir`: Directory for Objaverse's Shap-E latent codes.

### Model

- `--mode`: Choose the fusion mode/model:
    - `-1`: Image-to-3D Shap-E
    - `-2`: Text-to-3D Shap-E
    - `1`: Average Fusion MM-Shap-E
    - `2`: Cross-Modal Fusion MM-Shap-E
- `--load`: Load pretrained weights before training, evaluation, or demo.

### Training

- `--epochs`: Number of training epochs (default: 3).
- `--batch_size`: Batch size for training (default: 8).
- `--learning_rate`: Learning rate for training (default: 0.001).
- `--train`: Run the training loop.

### Evaluation

- `--eval`: Run the evaluation loop.

### Demo

- `--demo`: Run a demo showcasing the model's output.
