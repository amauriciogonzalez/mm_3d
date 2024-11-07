# MM-Shap-E: A Multimodal Text-and-Image-to-3D Pipeline

*This README is subject to change as the project progresses*


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



## Arguments (`mm_shap_e.py`)

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
- `--obja_latent_dir`: Directory for Objaverse's Shap-E latent codes  (`<object_id>.pt` files).

### Model

- `--mode`: Choose the fusion mode/model:
    - `-1`: Image-to-3D Shap-E
    - `-2`: Text-to-3D Shap-E
    - `1`: Average Fusion MM-Shap-E
    - `2`: Cross-Modal Fusion MM-Shap-E
    - `3`: Sequence-Level Cross-Modal Fusion MM-Shap-E (Late Fusion)
    - `4`: Gated Fusion MM-Shap-E (Late Fusion)
    - `5`: Weighted Fusion MM-Shap-E (Late Fusion)
    - `6`: Text-Guided Fusion MM-Shap-E (Late Fusion)
    - `104`: Cross-Modal Fusion MM-Shap-E (Early Fusion)
    - `300`: Text-Guided Fusion MM-Shap-E (Early Fusion)
- `--load`: Load pretrained weights before training, evaluation, or demo.

### Training

- `--epochs`: Number of training epochs (default: 3).
- `--batch_size`: Batch size for training (default: 8).
- `--learning_rate`: Learning rate for training (default: 0.001).
- `--train`: Run the training loop.

### Evaluation

- `--eval`: Run the evaluation loop.
- `--test_parallelization`: Evaluates model with and without parallelization in late-fusion modes.

### Ablation Options

--`text_ablation`: Runs text ablation variations in evaluation/demo.
--`karras_ablation`: Runs Karras ablation variations in evaluation/demo.

### Demo

- `--demo`: Run a demo showcasing the model's output.
- `--demo_all`: Runs a comprehensive demo with most model modes on Objaverse or an in-the-wild image.
- `--img_path`: Specifies the path to an image for in-the-wild demo.
- `--text_input`: Provides the text input for the in-the-wild demo.
