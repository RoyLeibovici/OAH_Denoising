# Off-Axis Holography (OAH) Denoising

This repository provides a deep learning-based pipeline for **denoising cell images** acquired from off-axis holography (OAH) video data.  
The pipeline includes:  
- **Segmentation** of individual cells from holographic videos using the pre-trained `CellSAM` model.  
- **Preprocessing** of segmented cells (cropping, proportion filtering, resizing).  
- **Denoising** using a convolutional autoencoder with gated skip connections to reduce speckle noise while preserving morphological details.

---

## Features
- Segmentation of holographic video frames into **cell crops** and **background patches**.
- Automated preprocessing including **size normalization** and **quality filtering**.
- Custom **denoising autoencoder** (CNN + gated U-Net skip connections) with a specialized loss function.
- Flexible processing modes for video, frames, cell images, and resizing.

---

## Repository Structure

OAH_Denoising/  
├── src/    
│ ├── Segmentation/ # CellSAM-based segmentation pipeline   
│ ├── Denoising/ # Autoencoder model and training code  
│ └── init.py   
├── main.py # Entry point for processing    
├── requirements.txt # Python dependencies  
├── pyproject.toml # Build/system configuration 
├── .gitmodules # CellSAM dependency (submodule)    
└── README.md # 


---

## Installation
Clone the repository **recursively** to include the CellSAM submodule:
```bash
git clone --recursive https://github.com/RoyLeibovici/OAH_Denoising.git
cd OAH_Denoising
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Usage
The pipeline supports several modes, selected via command-line arguments in `main.py`:  
   - `video`: Processes raw `.avi` videos, segments frames, and extracts cell crops.  
   - `frames`: Processes raw `.png` frames for segmentation and cropping.  
   - `cells`: Processes existing cropped cells [128×128] for denoising.  
   - `cells-resize`: Normalizes cell crops of arbitrary size to [128×128] before running the model.  


## Example Commands 
We highly recommend running the model with cropped cells, for better results.

Process existing cropped cells:
```bash
python main.py --input ./cells --workdir ./results --mode cells
```

Resize cell crops:

```bash
python main.py --input ./raw_cells --workdir ./output --mode cells-resize
```


Process existing frame images:
```bash
python main.py --input /path/to/frames --workdir ./results --mode frames
```


Process raw video files
```bash
python main.py --input /path/to/video_dir --workdir ./results --mode video
```





