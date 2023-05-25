
import sys
sys.path.append('/home/jovyan/work/styleGAN-Human')
import os
#import pickle
import PIL.Image
import torch
import numpy as np
import clip
import math
#import matplotlib.pyplot as plt
#from argparse import Namespace
from torch_utils.models import Generator
#from tensorboardX import SummaryWriter
import timm
#import torch.nn.functional as F
import legacy
import argparse
#import torchvision
#import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torchvision
import PIL.Image


class CLIPLoss(torch.nn.Module):
    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.opts = opts
        self.model, self.preprocess = clip.load("ViT-B/32", device = opts.device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(32, 16))

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        #image = F.pad(self.avg_pool(self.upsample(image)), (56, 56) , "constant", 1)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

class IDLoss(torch.nn.Module):
    def __init__(self, opts):
        super(IDLoss, self).__init__()
        #self.m = timm.create_model('resnet50', pretrained=True, num_classes=0).to(opts.device)
        #self.m = timm.create_model('convnext_base', pretrained=True, num_classes=0).to(opts.device)
        self.m = timm.create_model('convnext_tiny', pretrained=True, num_classes=0).to(opts.device)
        
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(32, 16))
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x, x_hat):
        xr = self.avg_pool(self.upsample(x)).detach()
        xr_hat = self.avg_pool(self.upsample(x_hat)) 
        #yr = self.m(xr)
        #yr_hat = self.m(xr_hat)
        return self.criterion(self.m(xr), self.m(xr_hat))
        #return self.criterion(self.m(x), self.m(x_hat))
        #n_samples = x.shape[0]
        #xr = self.avg_pool(self.upsample(x)) 
        #xr = xr.detach()
        #xr_hat = self.avg_pool(self.upsample(x_hat)) 
        #loss = 0
        #sim_improvement = 0
        #count = 0
        #for i in range(n_samples):
        #    loss += self.criterion(self.m(xr), self.m(xr_hat))
        #    count += 1
        #return loss / count, sim_improvement / count

       

class Style(torch.nn.Module):
    def __init__(self, opts):
        super(Style, self).__init__()
        if opts.stylegan_weights is None: #if not None, the input is the path of the pth file
            if not opts.multi_id:
                opts.stylegan_weights = os.path.join(opts.base_path, f"checkpoints/model_{opts.img_name.split('.')[0]}.pkl")
                print(opts.stylegan_weights )
                if not os.path.exists(opts.stylegan_weights.replace('.pkl','.pth')):
                    legacy.convert(opts.stylegan_weights, opts.stylegan_weights.replace('.pkl','.pth'), G_only=True)
                opts.stylegan_weights = os.path.join(opts.base_path, f"checkpoints/model_{opts.img_name.split('.')[0]}.pth")
            else:
                opts.stylegan_weights = os.path.join(opts.base_path, f"checkpoints/model_GANULKUAERIA_multi_id.pth")
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
        #self.mean_latent = self.decoder.mean_latent(4096).to(self.opts.device) #antoher way maybe better
        self.mean_latent = self.latent_avg.clone().to(self.opts.device)

class Coach:
    def __init__(self, opts):
        self.opts = opts
        self.device = opts.device
        self.real =  self.opts.img_name is not None
        self.net = Style(opts=opts)
        self.id_loss = IDLoss(opts).to(opts.device).eval()
        self.clip_loss = CLIPLoss(opts)
        self.latent_l2_loss = torch.nn.MSELoss().to(self.device).eval()
        self.transform = transforms.Compose( # normalize to (-1, 1)
                        [transforms.ToTensor(),
                        transforms.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5))])
    
    def imshow(self, x):
        x_ = (x.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).detach().clone().to(torch.uint8)
        img = PIL.Image.fromarray(x_[0].cpu().numpy(), 'RGB')
        #display(img)  
        return img
    
    # The learning rate adjustment function.
    def get_lr(self, t, initial_lr, rampdown=0.50, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)

        return initial_lr * lr_ramp
    
    def get_x(self, w):
        x0, _ = self.net.decoder(w, truncation=1, input_is_latent=True, real=True)
        return x0
    def get_w(self):
        with torch.no_grad():
            if self.real  == 1:
                w0 = torch.load( os.path.join(self.opts.base_path, f"embeddings/test/PTI/{self.opts.img_name.split('.')[0]}/0.pt"))
            else:
                w = torch.from_numpy(np.random.RandomState(self.opts.seed).randn(1, 512)).float().to(self.opts.device)
                x0, w0 = self.net.decoder([w], return_latents=True, truncation=self.opts.truncation, truncation_latent=self.net.mean_latent, real = 0)
        return w0
    
    def get_xw(self, w_in=None):
        with torch.no_grad():
            if self.real  == 1:
                w0 = self.get_w() ## read from a file
                x0, _ = self.net.decoder(w0, truncation=self.opts.truncation, truncation_latent=self.net.mean_latent, input_is_latent=True, real=True)
                #x0, _ = self.net.decoder(w0, truncation=1, input_is_latent=True, real=True)
            else:
                if w_in is None:
                    w = torch.from_numpy(np.random.RandomState(self.opts.seed).randn(1, 512)).float().to(self.opts.device)
                else:
                    w = w_in
                #mean_latent = self.net.decoder.mean_latent(4096).to(self.opts.device)
                x0, w0 = self.net.decoder([w], return_latents=True, truncation=self.opts.truncation, truncation_latent=self.net.mean_latent, real = 0)
        return x0, w0
        

    def train(self, network_pkl = None, w_in = None):
        layers = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])
        x0, w0 = self.get_xw(w_in = None)
        clip_loss = CLIPLoss(self.opts)
        txt = torch.cat([clip.tokenize(self.opts.description)]).to(self.opts.device)

        # Initialize the latent vector to be updated.
        mask = torch.zeros_like(w0.clone().detach())
        mask[:, layers, :] = 1
        mask.requires_grad = False
        
        w_hat = w0.detach().clone()
        w_hat.requires_grad = True

        
        optimizer = torch.optim.Adam([w_hat], lr=self.opts.lr)
        for i in range(self.opts.step):
            # Adjust the learning rate.
            t = i /self.opts.step
            lr = self.get_lr(t, self.opts.lr)
            optimizer.param_groups[0]["lr"] = lr

            # Generate an image using the latent vector. what about truncation!
            w_hat2 = mask * w_hat + (1 - mask) * w0
            x_hat, _ = self.net.decoder(w_hat2, input_is_latent=True, randomize_noise=False, real = 1, truncation=self.opts.truncation, truncation_latent=self.net.mean_latent)
            #x_hat, _ = self.net.decoder(w_hat2, input_is_latent=True, randomize_noise=False, real = 1)

            # Calculate the loss value.
            c_loss = clip_loss(x_hat, txt)
            l2_loss =  self.latent_l2_loss(w0, w_hat)#     ((w0 - w_hat) ** 2).sum()
            id_loss = self.id_loss(x0, x_hat)
            #print(((w0 - w_hat) ** 2).sum(), self.latent_l2_loss(w0, w_hat))
            if self.opts.id_lambda>0:
                loss = c_loss + self.opts.l2_lambda * l2_loss + self.opts.id_lambda * id_loss
            else:
                loss = c_loss + self.opts.l2_lambda * l2_loss 

            # Get gradient and update the latent vector.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the current state.
            if i%50 == 0:
                with torch.no_grad():
                    print(f"it: {i:04d}, lr: {lr:.4f}, loss: {loss.item():.8f}, dw: {self.latent_l2_loss(w0, w_hat).item():.8f}")
                #with torch.no_grad():
                #    x_interp.append(x_hat.clone().detach())

        print("Done!")
        with torch.no_grad():
            img_gen, w_final = self.net.decoder([w_hat], return_latents=True, input_is_latent=True, randomize_noise=False)
        return img_gen.detach().clone().cpu(), w_final.detach().clone().cpu(), x0.detach().clone().cpu(), w0.detach().clone().cpu()
  
    def get_real_image(self):
        real_img_path = os.path.join(self.opts.image_path, self.opts.img_name)
        real_image = self.transform(PIL.Image.open(real_img_path).convert('RGB'))
        return real_image.unsqueeze(0)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--description', default='A long sleeve', type=str, help='')
    #parser.add_argument('--description', default='A blue dress', type=str, help='')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_rampup', default=0.05, type=float)
    parser.add_argument('--lr_rampdown', default=0.05, type=float)
    parser.add_argument('--step', default=51, type=int)
    parser.add_argument('--save_intermediate_image_every', default=350, type=int)
    parser.add_argument('--clip_lambda', default=1, type=float)
    parser.add_argument('--l2_lambda', default=1, type=float)
    parser.add_argument('--id_lambda', default=20, type=float)
    parser.add_argument('--multi_id', default=0, type=int)
    parser.add_argument('--stylegan_weights', default=None, type=str)
    #parser.add_argument('--image_path', default="/home/jovyan/work/styleGAN-Human/aligned_image/latent_image/latent_image", type=str)
    #parser.add_argument('--base_path', default="/home/jovyan/work/styleGAN-Human/outputs/ptis_wp", type=str)   
    #parser.add_argument('--results_dir', default="/home/jovyan/work/styleGAN-Human/outputs/ptis_wp/op_edit", type=str)  
    parser.add_argument('--image_path', default="/home/jovyan/work/dataset/ds_200", type=str)
    parser.add_argument('--base_path', default="/home/jovyan/work/styleGAN-Human/outputs/ds_200", type=str)   
    parser.add_argument('--results_dir', default="/home/jovyan/work/results/Op", type=str)  
    
    parser.add_argument('--img_name', default="image_000003.jpg", type=str) 
    parser.add_argument('--truncation', default=1.0, type=float) 
    
    ########################
    # sample run code
    # ./scripts/opt_clip.py --image_path ./dataset/ds_200  --base_path ./styleGAN-Human/outputs/ds_200 --results_dir ./results/Op
    ########################
    #args.save_intermediate_image_every = args.step - 1  
    #args.device = torch.device('cuda:0') 
    #args.device = torch.device('cpu') 
    #args.multi_id = 0
    #args.stylegan_weights = None #'/home/jovyan/work/ML-master/styleGAN-Human/pretrained_models/stylegan2_1024.pth' #set None to read from a file with img_name
    #args.image_path = "/home/jovyan/work/styleGAN-Human/aligned_image/latent_image/latent_image"
    #args.base_path = "/home/jovyan/work/styleGAN-Human/outputs/ptis_wp"    
    #args.results_dir = '/home/jovyan/work/styleGAN-Human/outputs/ptis_wp/op_edit' #relative to the current dir not used at the moment

    return parser


def main(args):
    c = Coach(args)
    x_real=c.get_real_image()
    x1, w1, x0, w0 = c.train()
    name = args.img_name.split('.')[0]
    #xL = []
    #for i in range(2, 6):
    #     wL = w0 + i * (w1 - w0) 
    #     xL.append(c.get_x(wL.cuda()).clone().detach().cpu())
    #x00 = c.get_x(wL.cuda()).clone().detach().cpu()
    #torch.save([x1, w1, x0, w0], f'{args.results_dir}/{name}.pt')
    torchvision.utils.save_image(torch.cat([x_real.detach().cpu(), x1.detach().cpu()]), f'{args.results_dir}/{name}.jpg' , normalize=True, scale_each=True, range=(-1, 1), nrow=1)
    #final = torch.cat((x0, x1, xL[0], xL[1], xL[2], xL[3]), 3)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device('cuda:0') 

    if not os.path.exists(args.results_dir): 
        os.makedirs(args.results_dir, exist_ok=True)

    #if args.output_dir:
    #    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
