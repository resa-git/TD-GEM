import torch
import os
results_dir = "/home/jovyan/work/styleGAN-Human/outputs/ptis/wpkle"
base_path="/home/jovyan/work/styleGAN-Human/outputs/ptis"
image_path = '/home/jovyan/work/dataset/test_100'
images = [im for im in os.listdir(image_path) if '.png' in im or 'jpg' in im]
os.makedirs(results_dir, exist_ok=True)

w = []
for image in images:
    w.append(torch.load( os.path.join(base_path, f"embeddings/test/PTI/{image.split('.')[0]}/0.pt")))

wt = torch.cat(w)
torch.save(wt, os.path.join(results_dir, 'train.pt'))
torch.save(wt, os.path.join(results_dir, 'test.pt'))
print('Done!')