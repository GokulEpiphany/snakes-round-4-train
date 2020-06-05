import random
import json
import numpy as np
import pandas as pd
import argparse
import base64

import aicrowd_helpers
import time
import traceback

import time
from os.path import join as pjoin
import torch
import torchvision as tv
import bit_common
import bit_hyperrule
import big_transfer.models as models
import big_transfer.lbtoolbox as lb
from dataset import SnakeDataset

import glob
import os
import json

AICROWD_TEST_IMAGES_PATH = os.getenv("AICROWD_TEST_IMAGES_PATH","./data/validate_images_small/")
AICROWD_TEST_METADATA_PATH = os.getenv("AICROWD_TEST_METADATA_PATH","./data/validate_labels_small.csv")
AICROWD_PREDICTIONS_OUTPUT_PATH = os.getenv("AICROWD_PREDICTIONS_OUTPUT_PATH","random_prediction.csv")

VALID_SNAKE_SPECIES = list(pd.read_csv("round4_classes.csv")["scientific_name"])


def run():
  aicrowd_helpers.execution_start()

  #MAGIC HAPPENS BELOW
  torch.backends.cudnn.benchmark=True
  device = torch.device("cuda:0")
  assert torch.cuda.is_available()
  precrop,crop = bit_hyperrule.get_resolution_from_dataset('snakes_dataset') # verify
  valid_tx = tv.transforms.Compose([
    tv.transforms.Resize((crop,crop)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
  given_df = pd.read_csv(AICROWD_TEST_METADATA,PATH)

  valid_set = SnakeDataset(AICROWD_TEST_IMAGES_PATH,is_train=False,transform=valid_tx,target_transform=None,csv_file=validate_csv_file) # verify
  valid_loader = torch.utils.data.DataLoader(
  valid_set,batch_size=32,shuffle=False,num_workers=0,pin_memory=True,drop_last=False)
  model = models.KNOWN_MODELS['BiT-M-50x1'](head_size= len(VALID_SNAKE_SPECIES),zero_head=True)
  model = torch.nn.DataParallel(model)
  optim = torch.optim.SGD(model.parameters(),lr=0.003,momentum=0.9)
  model_loc = pjoin('models','initial.pth.tar')
  checkpoint = torch.load(model_loc,map_location='cpu')
  model.load_state_dict(checkpoint['model'])
  model = model.to(device)
  model.eval()
  results = np.empty((0,783),float)
  for b,(x,y) in enumerate(data_loader): #add name to dataset, y must be some random label
    with torch.no_grad():
      x = x.to(device,non_blocking=True)
      y = y.to(device,non_blocking=True)
      logits = model(x)
      softmax_op = torch.nn.Softmax(dim=1)
      probs = softmax_op(logits)
      data_to_save = probs.data.cpu().numpy()
      results = np.concatenate((results,data_to_save),axis=0)
  filenames = given_df[['hashed_id']]
  df = pd.DataFrame(data=results,index=flat_filenames,columns=VALID_SNAKE_SPECIES)
  df.index.name='hashed_id'



  aicrowd_helpers.execution_success({
    "predictions_output_path": AICROWD_PREDICTIONS_OUTPUT_PATH
  })


if __name__ == "__main__":
  try:
    run()
  except Exception as e:
    error = traceback.format_exc()
    print(error)
    aicrowd_helpers.execution_error(error)



