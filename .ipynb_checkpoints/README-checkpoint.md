# MeshVAE

MeshVAE is a nice little generative neural network

---
## How to: setup an environment
#### 1. Create new conda environment:
```
conda create -n pytorch3d python=3.9
```
#### 2. Activate environment:
```
conda activate pytorch3d
```
#### 3. Install cuda toolkit:
```
conda install nvidia/label/cuda-12.4.0::cuda-toolkit
```
#### 4. Install pytorch:
```
pip3 install torch torchvision torchaudio
```
#### 5. Install necessary dependencies:
```
conda install -c iopath iopath && \
conda install ipykernel
```
#### 6. Set environment variables for Pytorch3d compilation: 
```
export CUDA_HOME=$CONDA_PREFIX && \
export FORCE_CUDA=1
```
#### 7. Install Pytorch3D from source
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
#### 8. Create a new jupyter kernel for your environment:
```
python -m ipykernel install --user --name pytorch3d --display-name "PyTorch3D"
```
---
## How to: train MeshVAE:
### 1. Go to `dataset` folder and follow instructions to create dataset
### 2. Go to `vgg_finetuning` folder and follow instructions to create weights
### 3. Launch `MeshVAE.ipynb` notebook and wait...