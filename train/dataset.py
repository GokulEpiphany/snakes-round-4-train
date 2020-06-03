from torchvision.datasets.vision import VisionDataset
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import string
import codecs
import pandas as pd

class SnakeDataset(VisionDataset):
  def __init__(self,root,is_train=True,transform=None,target_transform=None,csv_file=None):
    """
    Args:
      root -> Path of the root directory
      csv_file-> Path to csv file
    """
    super(SnakeDataset,self).__init__(root,transform=transform,target_transform=target_transform)
    self.train = is_train
    if self.train:
      img_pth = os.path.join(root,'train_images')
    else:
      img_pth = os.path.join(root,'validate_images')
    csv_file = pd.read_csv(csv_file,low_memory=False)
    self.hashed_ids = csv_file['hashed_id']
    given_classes = csv_file['scientific_name']
    self.classes = sorted(np.unique(given_classes))
    self.class_to_idx = {_class:i for i,_class in enumerate(self.classes)}
    self.labels = [self.class_to_idx[x] for x in given_classes]
    self.transform = transform
    self.img_dir = img_pth
    print(len(self.classes))
    assert len(self.classes) == 783

  def __getitem__(self,index):
    img_full_path = os.path.join(self.img_dir,'{}.jpg'.format(self.hashed_ids[index]))
    with open(img_full_path,'rb') as f:
      img = Image.open(f)
      img = img.convert('RGB')
    if self.transform is not None:
      sample = self.transform(img)
    target = self.labels[index]
    return sample,target

  def __len__(self):
    return len(self.hashed_ids)

    

