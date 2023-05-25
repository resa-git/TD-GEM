import os
#import edit
#import run_pti
#import alignment
import torch
import sys


impath = '../dataset/ds_200'
folder = 'ds_200'
os.makedirs(f'outputs/{folder}/edit_upper', exist_ok = True)
dd = [f for f in os.listdir(f'outputs/{folder}/checkpoints') if f.endswith('pth')]
dataset  =[d for d in os.listdir(impath) if "checkpoints" not in d and d.endswith('jpg')]
print('I found in total ' , len(dataset), ' of files')
for i, dn in enumerate(dataset):
    print(i, dn)
    d = dn.split('.')[0] #exclude .png, e.g. seed0000
    matches = [x for x in dd if f"model_{d}.p" in x]
    #print(matches)
    if len(matches) != 0:
        check_name = matches[0].split('.')[0] + '.pkl' #e.g. model_seed0000.pkl
        #print(d, check_name)
        #print('-----------')
        if True:
            os.system(f"python edit.py --network outputs/{folder}/checkpoints/model_{d}.pth --attr_name upper_length --outdir outputs/{folder}/edit_upper --real True --real_w_path outputs/{folder}/embeddings/test/PTI/{d}/0.pt --real_img_path {impath}/{dn} --output_name {d}_upper")
            #print(f"python edit.py --network outputs/{folder}/checkpoints/model_{d}.pkl --attr_name upper_length --outdir outputs/{folder}/edit_upper --real True --real_w_path outputs/{folder}/embeddings/test/PTI/{d}/0.pt --real_img_path {impath}/{dn} --output_name {d}_upper")
    #break
