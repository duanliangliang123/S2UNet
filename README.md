S2UNet: Scalable RGB-X Dehazing and Beyond via Structure-Aware Uncertainty Fusion

Overview
This repository contains the official PyTorch implementation for the paper "S2UNet: Scalable RGB-X Dehazing and Beyond via Structure-Aware Uncertainty Fusion".

Our proposed S2UNet overcomes the architectural rigidity of existing multimodal restoration methods. Driven by Structure-Aware Uncertainty Fusion (SAUF) and the Spatial-Spectral Fusion Block (SSFB), the network seamlessly adapts to arbitrary RGB-X configurations (e.g., RGB-NIR, RGB-Thermal, Tri-modal) without structural modifications, achieving state-of-the-art performance on multimodal dehazing and denoising tasks.

📢 Important Note on Supplementary Material
Currently, this supplementary material contains only:

A subset of test images for visual evaluation.

The pre-trained S2UNet-3-16 model weights evaluated on the RNH-he dataset.

Full Release: The full source code is provided in this supplementary material. Comprehensive datasets (including AirSim-VID and Real-NAID) and all pre-trained models (e.g., S2UNet-2-16, S2UNet-2-56, S2UNet-3-40) will be fully publicly released upon the paper's acceptance.

Network Architectures: Dynamic vs. Static
To facilitate both flexible research scaling and straightforward deployment, we provide two versions of the network architecture in the network/ directory:

S2UNet.py (Dynamic & Scalable): This is the core scalable implementation of our paper. It supports an arbitrary number of modalities (e.g., bi-modal, tri-modal, quad-modal). The network dynamically constructs the modality-specific encoders and mutual rectification modules (SARM) based on the in_channels_list parameter (e.g., [3, 1, 1] for RGB+NIR1+NIR2).

Net_tri.py (Static Tri-modal): A static, hard-coded variant optimized strictly for three modalities. This version provides a more straightforward, unrolled architectural reference for the tri-modal experiments discussed in the paper and can be used for fixed tri-modal deployment.

Environment Setup
Python 3.8+

PyTorch 2.0+ & torchvision

NumPy, Pillow (PIL), scikit-image

thop (optional, for calculating FLOPs and Params)

Usage
1. Data Preparation
For evaluating the provided pre-trained model, please place the sample test images in the following directory structure:

Plaintext

data/
└── RNH3K-he/
    └── test/
        ├── hazy/        # Hazy RGB inputs
        ├── vis/         # Clean RGB ground truths
        ├── nir/         # Auxiliary modality 1 (e.g., NIR)
        └── nir2/        # Auxiliary modality 2 (e.g., Thermal/NIR2)
2. Testing the Pre-trained Model (Tri-modal)
We provide two testing scripts corresponding to the two network implementations. Ensure the pre-trained weights (S2UNet_val.pth) are placed in the trained_models/ directory.

Option A: Using the Scalable Architecture (main_S2UNet.py) This script initializes the dynamic S2UNet.py with in_channels_list=[3, 1, 1] and automatically converts the legacy weights for evaluation.

Bash

python main_S2UNet.py
Option B: Using the Static Tri-modal Architecture (main_Tri.py) This script uses the hard-coded Net_tri.py to evaluate the exact same tri-modal task.

Bash

python main_Tri.py
Both scripts will run inference, output the overall PSNR and SSIM metrics, and save the restored images (with metrics text overlaid) to the results/RNH3K-he/S2UNet/ directory.

3. Training (Upon Acceptance)
Note: Full training datasets will be released upon acceptance. To train the models from scratch once the complete datasets are available, simply modify the Config class in either main_S2UNet.py or main_Tri.py to set mode = 'train' and run the script. The training pipeline supports Automatic Mixed Precision (AMP) and dynamic learning rate scheduling.
