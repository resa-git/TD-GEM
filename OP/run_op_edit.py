import torch
import os
#pip install git+https://github.com/openai/CLIP.git
#pip install timm
results_dir = "/home/jovyan/work/results/Op"
base_path="/home/jovyan/work/styleGAN-Human/outputs/ds_200"
image_path = '/home/jovyan/work/dataset/ds_200'
images = [im for im in os.listdir(image_path) if '.png' in im or 'jpg' in im]
#print(images)
#images = ['seed0000.png', 'seed4000.png']
#images = ['seed4000.png']


for image in images:
    print(f"------ {image} --------------")
    os.system(f'python ./scripts/opt_clip_delta.py --img_name {image} --image_path {image_path} --results_dir {results_dir} --base_path {base_path} --step 51 --stylegan_weights styleGAN-Human/pretrained_models/stylegan2_1024.pth')
    
import argparse
import torch
import os

parser = argparse.ArgumentParser(description="Your script description here")
parser.add_argument("--results_dir", type=str, default="/home/jovyan/work/results/Op", help="Path to results directory")
parser.add_argument("--base_path", type=str, default="/home/jovyan/work/styleGAN-Human/outputs/ds_200", help="Path to directory where the latent code is located")
parser.add_argument("--image_path", type=str, default="/home/jovyan/work/dataset/ds_200", help="Path to image directory")
parser.add_argument("--step", type=int, default=51, help="The maximum itteration")
parser.add_argument("--stylegan_weights", type=str, default="styleGAN-Human/pretrained_models/stylegan2_1024.pth", help="Path to styleGAN weights")

args = parser.parse_args()

image_path = args.image_path
results_dir = args.results_dir
base_path = args.base_path
step = args.step
stylegan_weights = args.stylegan_weights

images = [im for im in sorted(os.listdir(image_path)) if im.endswith(('.png', '.jpg'))]


for image in images:
    print(f"------ {image} --------------")
    os.system(f'python ./scripts/opt_clip_delta.py --img_name {image} --image_path {image_path} --results_dir {results_dir} --base_path {base_path} --step {step} --stylegan_weights {stylegan_weights}')

#####################################################################
#                 RUN the code as
#-------------------------------------------------------------------
# python your_script.py --results_dir /path/to/results --base_path /path/to/base --image_path /path/to/images --step 51 --stylegan_weights /path/to/stylegan_weights.pth
#####################################################################

