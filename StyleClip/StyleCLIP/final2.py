#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!pip install git+https://github.com/openai/CLIP.git


# In[6]:


#!pip install timm


# In[1]:
import sys

import os
#os.chdir('/home/jovyan/work/ML-master/StyleClip/StyleCLIP')
#sys.path.append('/home/jovyan/work/ML-master/StyleClip/StyleCLIP')

# In[2]:


#timm.list_models('convnext*')


# In[3]:


import sys
import os
import PIL.Image
import torch
import numpy as np
import clip
import math
import matplotlib.pyplot as plt
from argparse import Namespace
from torch_utils.models import Generator
import timm
import torch.nn.functional as F


# In[8]:


class CLIPLoss(torch.nn.Module):
    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.opts = opts
        self.model, self.preprocess = clip.load("ViT-B/32", device = opts.device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(32, 16))
        #self.avg_pool = torch.nn.AvgPool2d(kernel_size=(32, 32))

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        #image = F.pad(self.avg_pool(self.upsample(image)), (56, 56) , "constant", 1)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity


# In[12]:


class IDLoss(torch.nn.Module):
    def __init__(self, opts):
        super(IDLoss, self).__init__()
        #self.m = timm.create_model('resnet50', pretrained=True, num_classes=0).to(opts.device)
        self.m = timm.create_model('convnext_base', pretrained=True, num_classes=0).to(opts.device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(32, 16))
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x, x_hat):
        xr = self.avg_pool(self.upsample(x)) 
        xr_hat = self.avg_pool(self.upsample(x_hat)) 
        #yr = self.m(xr)
        #yr_hat = self.m(xr_hat)
        return self.criterion(self.m(xr), self.m(xr_hat))
        #return self.criterion(self.m(x), self.m(x_hat))
        


# create files from a seed and save it to the file
# 

# /pti/pti_configs/hyperparameters.py: </br>
# first_inv_type = 'w+' -> Use pretrained e4e encoder</br>
# first_inv_type = 'w' -> Use projection and optimization</br>
# /pti/pti_configs/paths_config.py:</br>
# input_data_path: path of real images</br>
# e4e: path of e4e_w+.pt</br>
# stylegan2_ada_shhq: pretrained stylegan2-ada model for SHHQ</br>

# ## Clip Functions

# In[13]:


class Style(torch.nn.Module):
    def __init__(self, opts):
        super(Style, self).__init__()
        if opts.stylegan_weights is None:
            opts.stylegan_weights = os.path.join(opts.base_path, f"checkpoints/model_{opts.img_name.split('.')[0]}.pth")
        self.opts = opts
        config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
        self.decoder = Generator(size = 1024,
                                 style_dim=config["latent"],
                                 n_mlp=config["n_mlp"],
                                 channel_multiplier=config["channel_multiplier"])
        self.latent_avg = None
        self.mean_latent = None
        self.load_weights()

    def load_weights(self):
        print('Loading decoder weights from pretrained!')
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt['g_ema'])
        self.decoder.eval().to(self.opts.device)
        self.latent_avg =ckpt['latent_avg']
        self.mean_latent = self.latent_avg.clone().to(self.opts.device)


# In[14]:


class Coach:
    def __init__(self, opts):
        self.opts = opts
        self.device = opts.device
        self.real =  self.opts.img_name is not None
        self.net = Style(opts=opts)
        if self.opts.id_lambda > 0:
            self.id_loss = IDLoss(opts).to(opts.device).eval()
        if self.opts.clip_lambda > 0:
              self.clip_loss = CLIPLoss(opts)
        if self.opts.l2_lambda > 0:
              self.latent_l2_loss = torch.nn.MSELoss().to(self.device).eval()
        self.x0 = None
        self.w0 = None
    
    def imshow(self, x):
        x_ = (x.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).detach().clone().to(torch.uint8)
        img = PIL.Image.fromarray(x_[0].cpu().numpy(), 'RGB')
        #display(img)   
    
    # The learning rate adjustment function.
    def get_lr(self, t, initial_lr, rampdown=0.50, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)

        return initial_lr * lr_ramp
    
    def get_x(self, w):
        x0, _ = self.net.decoder(w, truncation=1, input_is_latent=True, real=True)
        return x0
    
    def get_xw(self):
        with torch.no_grad():
            if self.real  == 1:
                w0 = torch.load( os.path.join(self.opts.base_path, f"embeddings/test/PTI/{self.opts.img_name.split('.')[0]}/0.pt"))
                x0, _ = self.net.decoder(w, truncation=0.7, input_is_latent=True, real=True)
            else:
                w = torch.from_numpy(np.random.RandomState(self.opts.seed).randn(1, 512)).float().to(self.opts.device)
                mean_latent = self.net.decoder.mean_latent(4096).to(self.opts.device)
                x0, w0 = self.net.decoder([w], return_latents=True, truncation=0.7, truncation_latent=mean_latent, real = 0)
        self.x0 = x0
        self.w0 = w0
        return x0, w0
        

    def train(self, network_pkl = None):
        x0, w0 = self.get_xw()
        clip_loss = CLIPLoss(self.opts)
        txt = torch.cat([clip.tokenize(self.opts.description)]).to(self.opts.device)

        # Initialize the latent vector to be updated.
        w_hat = w0.detach().clone()
        w_hat.requires_grad = True
        
        x_interp = []
        optimizer = torch.optim.Adam([w_hat], lr=self.opts.lr)
        for i in range(self.opts.step):
            # Adjust the learning rate.
            t = i /self.opts.step
            lr = self.get_lr(t, self.opts.lr)
            optimizer.param_groups[0]["lr"] = lr

            # Generate an image using the latent vector. what about truncation!
            x_hat, _ = self.net.decoder([w_hat], input_is_latent=True, randomize_noise=False)

            # Calculate the loss value.
            c_loss = clip_loss(x_hat, txt)
            l2_loss =  self.latent_l2_loss(w0, w_hat)#     ((w0 - w_hat) ** 2).sum()
            id_loss = self.id_loss(x0, x_hat)
            #print(((w0 - w_hat) ** 2).sum(), self.latent_l2_loss(w0, w_hat))
            loss = c_loss + self.opts.l2_lambda * l2_loss + self.opts.id_lambda * id_loss

            # Get gradient and update the latent vector.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the current state.
            if i%50 == 0:
                print(f"it: {i:04d}, lr: {lr:.4f}, loss: {loss.item():.8f}")
                with torch.no_grad():
                    x_interp.append(x_hat.clone().detach())

        print("Done!")
        with torch.no_grad():
            img_gen, _ = self.net.decoder([w_hat], input_is_latent=True, randomize_noise=False)
        return img_gen, x_interp


# In[19]:


args = Namespace()
args.description = 'A really long sleeve'
args.lr_rampup = 0.05
args.lr = 0.1
args.step =1451
args.l2_lambda = 46 #0.005*9216 # The weight for similarity to the original image.
args.clip_lambda = 1
args.id_lambda = 40
args.save_intermediate_image_every = args.step - 1
args.results_dir = 'outputs/ptif/fake_edit_results'
args.device = torch.device('cuda:0') 
#args.device = torch.device('cpu') 

base_path = "/home/jovyan/work/StyleGAN-Human/outputs/ptif/fake_edit_results"
args.stylegan_weights = '../pretrained_models/stylegan2_1024.pth'
args.img_name = None
args.newG = None

args.seed = 4000
#image = [im for im in os.listdir(base_path) if '.png' in im]
seed = 4000

print("--------------------")
#img_name = f"seed_clip_{seed}"
#print(img_name)
#break
#args, real, network_pkl = None, img_name = None, base_path = None, seed = None
c = Coach(args)
x, x_intp = c.train()
print("--------------------") 
#c.imshow(x)


# # In[20]:


# c.imshow(x_intp[-1])


# # In[16]:


# c.imshow(x_intp[-1])


# # In[ ]:


# len(x_intp)


# # In[ ]:


# for i in range(14):
#     c.imshow(x_intp[i])


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:


# # lr = []
# # for t in range(1451):
# #     #print(t)
# #     lr.append(c.get_lr(t/1451, 0.01))
# # plt.plot(lr)


# # In[ ]:


# # c.imshow(c.x0)


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:


# # args = Namespace()
# # args.description = 'A long sleeve'
# # args.lr_rampup = 0.05
# # args.lr = 0.1
# # args.step = 151
# # args.l2_lambda = 0.005*9216 # The weight for similarity to the original image.
# # args.clip_lambda = 1
# # args.id_lambda = -1
# # args.save_intermediate_image_every = args.step - 1
# # args.results_dir = 'outputs/ptif/fake_edit_results'
# # args.device = torch.device('cuda:0') 
# # base_path = "/home/jovyan/work/StyleGAN-Human/outputs/ptif/fake_edit_results"
# # args.stylegan_weights = '../pretrained_models/stylegan2_1024.pth'
# # args.img_name = None
# # args.newG = None

# # args.seed = 4000
# # #image = [im for im in os.listdir(base_path) if '.png' in im]
# # seed = 4000

# # print("--------------------")
# # #img_name = f"seed_clip_{seed}"
# # #print(img_name)
# # #break
# # #args, real, network_pkl = None, img_name = None, base_path = None, seed = None
# # c = Coach(args)
# # x = c.train()
# # print("--------------------") 
# # c.show_image(x)


# # In[ ]:





# # In[ ]:





# # In[ ]:


# #3258.3564, device='cuda:0', grad_fn=<SumBackward0>) tensor(0.3536, device='cuda:0', g


# # In[ ]:


# #3258.3564/0.3536, 512*18


# # In[ ]:





# # In[ ]:


# #0.005*9216

