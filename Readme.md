# Efficient Lane Marking Recognition for Duckietown under Varying Lighting Conditions

This repository contains materials related to the paper:

**"Efficient Lane Marking Recognition for Duckietown for Varying Lighting Conditions"**

**Authors:** Ilia Nechaev, Michael Georgiades, Kirill Krinkin

## Abstract
Lane marking recognition is essential for autonomous navigation systems. Traditional color-based segmentation techniques, such as those used in Duckietown, are fast but sensitive to varying lighting conditions. This work proposes a lightweight convolutional neural network (CNN) designed specifically to improve robustness and efficiency for lane segmentation tasks on resource-constrained hardware, such as Jetson Nano. The network employs depthwise separable convolutions, batch normalization, and float16 quantization. A custom CUDA-based preprocessing pipeline integrated into a TensorRT engine enables real-time inference. Experimental results demonstrate high segmentation accuracy and resilience to lighting variations without significant computational overhead.

## Repository Structure
The repository consists of three main components:

### 1. Neural Network Training (Jupyter Notebook)
- **Path:** `Training.ipynb`
- This notebook contains the training pipeline for the CNN model, including:
    - Data loading and preprocessing
    - Data augmentation strategies
    - Model architecture definition
    - Training routines with monitoring and evaluation

### 2. CUDA-based Inference Example
- **Path:** `InferenceExample/`
- This directory contains source code for running inference using the trained model, optimized for CUDA and TensorRT deployment on Jetson Nano:
    - `main.cu` – CUDA example demonstrating efficient preprocessing and inference.
    - `CMakeLists.txt` – CMake configuration file to build the CUDA example.
    - `Dockerfile` – Provides a Dockerized environment for easy setup and deployment.
    - `RoadMarkupsSegmentationNet.engine` - A pre-compiled TensorRT engine file optimized for inference on Jetson Nano, quantized to float16

## Requirements
- NVIDIA Jetson Nano or compatible device
- Docker

## Setup and Usage

### Training
To train the model from scratch:
1. Open the notebook:
```bash
cd notebooks
jupyter notebook train_lane_segmentation.ipynb
```
2. Follow instructions within the notebook.

### Inference
Make sure you are using an NVIDIA Jetson Nano or compatible device with Docker installed.
```bash
cd src
docker build -t lane-segmentation .
docker run --runtime=nvidia -it --privileged lane-segmentation
# Inside the container
./nn /path/to/image
```
The output is saved as `out.png` in the same directory.

## Dataset
The training dataset is available publicly at:
- [Duckietown Road Markups Segmentation Dataset (Hugging Face)](https://huggingface.co/datasets/SPGC/DuckietownRoadMarkupsSegmentation)

Please cite our paper if you use this dataset in your work.

## Results
The developed CNN achieves:
- Dice coefficient: 0.97 (dataset test subset), 0.96 (unseen lighting conditions)
- Inference speed: ~35 FPS (CNN only), ~26 FPS (full pipeline) on Jetson Nano

## Citation
If you use our model, code, or dataset, please cite:
```bibtex
@inproceedings{nechaev2024lanemarking,
  author = {Ilia Nechaev and Michael Georgiades and Kirill Krinkin},
  title = {Efficient Lane Marking Recognition for Duckietown for Varying Lighting Conditions},
  year = {2025},
}
```
