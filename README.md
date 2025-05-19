# MeshVAE

MeshVAE is a nice little generative neural network

## Example generation:

| desk | sofa | chair | table |
| --- | --- | --- | --- |
| <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/desk.png" width="128" title="desk" alt="desk"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/divan_1.gif" width="128" title="sofa" alt="sofa"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/stool_nizhe.png" width="128" title="chair" alt="chair"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/stol.png" width="128" title="table" alt="table"/> |
| <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/desk2.png" width="128" title="desk" alt="desk"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/divan_2.gif" width="128" title="sofa" alt="sofa"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/stool2.png" width="128" title="chair" alt="chair"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/stol2.png" width="128" title="table" alt="table"/> |


## VAE and VAE+GAN comparison
| Original | MeshVAE | MeshVAE+GAN |
| --- | --- | --- |
| <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/bed_orig.png" width="128" title="bed target" alt="bed target"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/bed_vae.png" width="128" title="bed vae" alt="bed vae"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/bed_gan.png" width="128" title="bed gan" alt="bed gan"/> |
| <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/desk2.png" width="128" title="desk" alt="desk"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/divan_2.gif" width="128" title="sofa" alt="sofa"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/stool2.png" width="128" title="chair" alt="chair"/> | <img src="https://github.com/egichSerg/MeshVAE/blob/main/images/stol2.png" width="128" title="table" alt="table"/> |


---

## Model architecture

Based on VAE architecture, it uses VAE+GAN to achieve even better results than just VAE. As its architecture is so small, it generates 64x64x64 models in ~4ms!

### Architecture overview:

![MeshVAE architecture]()

### VAE+GAN schema:

![MeshVAE+GAN schema]()

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
