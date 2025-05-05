# Made for ModelNet10 dataset
[Source: ModelNet10 - Princeton 3D Object Dataset](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset)

## How to: generate a dataset
---
### 1. Download and unzip in this folder. You should have this hierarchy:
```
working_dir
|
|----\
|     \
other  ModelNet
       |
       |-------\
       |        \
   ModelNet10/  metadata_modelnet10.csv
```
### 2. Change root_dir variable in `create_dataset_images.ipynb` notebook to current work directory
### 3. Launch notebook