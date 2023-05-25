import os
#import edit
#import run_pti
#import alignment
import torch
import sys
# print(os.getcwd())
# sys.path.append(os.getcwd())
# from torch_utils.models import Generator

# config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
# decoder = Generator(size = 1024,
#                                 style_dim=config["latent"],
#                                 n_mlp=config["n_mlp"],
#                                  channel_multiplier=config["channel_multiplier"])

# w_hat = torch.random(1, 18, 512)
# x_hat, _ = decoder([w_hat], input_is_latent=True, randomize_noise=False)
#run_pti.run_PTI()
#edit.main()
#alignment.main()
##################################
##################################
# impath = 'outputs/ptis/images'
# folder = 'ptis'
# dd = [f for f in os.listdir(f'outputs/{folder}/checkpoints') if f.endswith('pkl')]
# dataset  =[d for d in os.listdir(impath) if "checkpoints" not in d and d.endswith('png')]
# #print(dataset)
# for i, dn in enumerate(dataset):
#     print(i, dn)
#     d = dn.split('.')[0] #exclude .png, e.g. seed0000
#     matches = [x for x in dd if f"model_{d}.p" in x]
#     #print(matches)
#     if len(matches) != 0:
#         check_name = matches[0].split('.')[0] + '.pkl' #e.g. model_seed0000.pkl
#         #print(d, check_name)
#         #print('-----------')
#         if True:
#             os.system(f"python edit.py --network outputs/{folder}/checkpoints/model_{d}.pkl --attr_name upper_length --outdir outputs/{folder}/edit_upper --real True --real_w_path outputs/{folder}/embeddings/test/PTI/{d}/0.pt --real_img_path {impath}/{dn} --output_name {d}_upper")
#             #print(f"python edit.py --network outputs/{folder}/checkpoints/model_{d}.pkl --attr_name upper_length --outdir outputs/{folder}/edit_upper --real True --real_w_path outputs/{folder}/embeddings/test/PTI/{d}/0.pt --real_img_path {impath}/{dn} --output_name {d}_upper")
#     #break
            
##################################
#              change pkl to pth
##################################
# import legacy
# fold = './outputs/ds_2000/checkpoints'
# Gpkl = [f.rsplit('.')[0] for f in os.listdir(fold) if f.endswith('.pkl')]
# Gpth =  [f.rsplit('.')[0] for f in os.listdir(fold) if f.endswith('.pth')]
# Gs1 = list(set(Gpkl) - set(Gpth))
# Gs = [g+'.pkl' for g in Gs1]
# for g in Gs:
#     ckpt_path = os.path.join(fold, g)
#     print(ckpt_path)
#     legacy.convert(ckpt_path, ckpt_path.replace('.pkl','.pth'), G_only=True)
#     #break
#     print(g)


##########################################################
#              convert pkl to pth
##########################################################
import legacy
import shutil
#fold_im = '../dataset/selectfromg1'
#%fold_100 = '../dataset/ds_100'
#G0 = [f.rsplit('.')[0] for f in os.listdir(fold_100) if f.endswith('.jpg')]
#G1 = [f.rsplit('.')[0] for f in os.listdir(fold_im) if f.endswith('.jpg')]
#Gs = list(set(G1).union(set(G0)))
#Gs =  Gs[:200]



fold_im = '../dataset/old'
Gs = [f.rsplit('.')[0] for f in os.listdir(fold_im) if f.endswith('.jpg')]
Gs =  sorted(Gs[:200])


##########################################################
#       copy the checkpionts and convert pkl to pth.
##########################################################
for im in Gs:
    shutil.copy2(f'../dataset/old/{im}.jpg', '../dataset/ds_200')
    #os.system(f'cp ../dataset/old/{im}.jpg  ../dataset/ds_200')
    #print(im)
i = 0
for im in Gs:
    try:
       shutil.copy2(f'./outputs/ds_100/checkpoints/model_{im}.pth', './outputs/old/checkpoints') 
    except:
        i = i + 1
        print(i)

i = 0
for im in Gs:
    try:
       shutil.copy2(f'./outputs/ds_7k_g1/checkpoints/model_{im}.pkl', './outputs/old/checkpoints') 
    except:
        i = i + 1
        print(i)

import legacy
fold = './outputs/old/checkpoints'
Gpkl = [f.rsplit('.')[0] for f in os.listdir(fold) if f.endswith('.pkl')]
Gpth =  [f.rsplit('.')[0] for f in os.listdir(fold) if f.endswith('.pth')]
Gs1 = list(set(Gpkl) - set(Gpth))
Gs = [g+'.pkl' for g in Gs1]
for g in Gs:
    ckpt_path = os.path.join(fold, g)
    print(ckpt_path)
    legacy.convert(ckpt_path, ckpt_path.replace('.pkl','.pth'), G_only=True)
    #break
    print(g)
##########################################################
## copy the embeddings (we had problem with permisions and also needed to switch ot os.system!)
##########################################################
from distutils.dir_util import copy_tree
for i, im in enumerate(Gs):
    try:
       copy_tree(f'./outputs/ds_100/embeddings/test/PTI/{im}', './outputs/old/embeddings/test/PTI/') 
       #os.system(f'cp -r ./outputs/ds_100/embeddings/test/PTI/{im} ./outputs/old/embeddings/test/PTI/')
    except:
        print(i + 1)

for i, im in enumerate(Gs):
    try:
       copy_tree(f'./outputs/ds_7k_g1/embeddings/test/PTI/{im}', './outputs/old/embeddings/test/PTI/') 
       #os.system(f'cp -r ./outputs/ds_7k_g1/embeddings/test/PTI/{im} ./outputs/old/embeddings/test/PTI/')
    except:
        print(i + 1)