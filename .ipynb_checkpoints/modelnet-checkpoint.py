import torch
import numpy as np
import pandas as pd

from typing import Tuple
import torchvision.transforms as transforms

class ModelNet(torch.utils.data.Dataset):
  # 2. Init the subclass
  def __init__(self,
               data_sheet: pd.DataFrame,
               device,
               transform=None):
    # 3. Create attributes
    self.device = device
    self.data_table = data_sheet
    self.classes = self.data_table['class'].unique()
    self.class_to_idx = {class_: i for i, class_ in enumerate(self.classes)}
    self.transform = transform
      

  def __len__(self) -> int:
    """Returns the total number of samples"""
    return len(self.data_table)

  # 6. Overwrite __getitem__()
  def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
    mesh_densities = torch.load(self.data_table.iloc[index]['density_path'])
    mesh_colors = torch.load(self.data_table.iloc[index]['color_path'])
    mesh = torch.cat((mesh_densities, mesh_colors))
    label = self.class_to_idx[self.data_table.iloc[index]['class']]

    # Transform if necessary
    if self.transform:
      return self.transform(mesh), label

    return mesh, label



class ModelNetRenders(torch.utils.data.Dataset):
  # 2. Init the subclass
  def __init__(self,
               data_sheet: pd.DataFrame,
               dataset_root_dir,
               device,
               transform=None):
    # 3. Create attributes
    self.device = device
    self.data_table = data_sheet
    self.dataset_dir = dataset_root_dir
    self.classes = self.data_table['class'].unique()
    self.class_to_idx = {class_: i for i, class_ in enumerate(self.classes)}
    self.transform = transform
      

  def __len__(self) -> int:
    """Returns the total number of samples"""
    return len(self.data_table)

  # 6. Overwrite __getitem__()
  def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
      renders = torch.from_numpy(np.load(self.dataset_dir / self.data_table.iloc[index].object_path.strip(' ')))
      label = self.class_to_idx[self.data_table.iloc[index]['class']]

      if self.transform:
          renders = self.transform(renders)
      
      return renders.unsqueeze(1).expand(-1, 3, -1, -1, -1), label


class ModelNetRendersNoLabels(torch.utils.data.Dataset):
  # 2. Init the subclass
  def __init__(self,
               data_sheet: pd.DataFrame,
               dataset_root_dir,
               device,
               transform=None):
    # 3. Create attributes
    self.device = device
    self.data_table = data_sheet
    self.dataset_dir = dataset_root_dir
    self.classes = self.data_table['class'].unique()
    self.class_to_idx = {class_: i for i, class_ in enumerate(self.classes)}
    self.transform = transform

  def __len__(self) -> int:
    """Returns the total number of samples"""
    return len(self.data_table)

  # 6. Overwrite __getitem__()
  def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
      renders = torch.from_numpy(np.load(self.dataset_dir / self.data_table.iloc[index].object_path.strip(' ')))
      label = self.class_to_idx[self.data_table.iloc[index]['class']]
      
      if self.transform:
          renders = self.transform(renders)
      
      return renders.unsqueeze(1)
