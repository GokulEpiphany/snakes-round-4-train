import cv2
import os
import pandas as pd
from functools import partial
import argparse

parser = argparse.ArgumentParser(description='Arguments for error images')
parser.add_argument('--split',type=str,help='Pick a split between valid or train')

def main(args):
    error_images = 0
    root_dir = partial(os.path.join,'..','data')
    metadata_root = partial(os.path.join,'..','data','metadata')
    if args.split == 'train':
        img_pth = root_dir('train_images')
        csv_file = metadata_root('train_labels.csv')
    else:
        img_pth = root_dir('valid_images')
        csv_file = metadata_root('valid_labels.csv')
    df = pd.read_csv(csv_file,low_memory=False)
    df_new = df[['hashed_id','scientific_name','country']]
    df_new['country']=df_new['country'].str.lower().replace(' ','-',regex=True)
    print(df_new.head())
    for idx,row in df_new.iterrows():
        if idx % 1000 ==0:
            print(idx)
        im = cv2.imread(os.path.join(img_pth,row['hashed_id']+'.jpg'))
    print(len(df_new))


if __name__== "__main__":
    args = parser.parse_args()
    main(args)
