
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
    '''
    Calculate the CLIP loss
    '''
    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.opts = opts
        self.model, self.preprocess = clip.load("ViT-B/32", device = opts.device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(32, 16))

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

class IDLoss(torch.nn.Module):
    '''
    Calculate the Identity P loss
    '''
    def __init__(self, opts):
        super(IDLoss, self).__init__()
        #self.m = timm.create_model('resnet50', pretrained=True, num_classes=0).to(opts.device)
        #self.m = timm.create_model('convnext_base', pretrained=True, num_classes=0).to(opts.device)
        self.m = timm.create_model('convnext_tiny', num_classes=0).to(opts.device)
        
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(32, 16))
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x, x_hat):
        xr = self.avg_pool(self.upsample(x)).detach()
        xr_hat = self.avg_pool(self.upsample(x_hat)) 
        return self.criterion(self.m(xr), self.m(xr_hat))

       

class Style(torch.nn.Module):
    '''
    Load the Generator
    '''
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
    '''
    Train the model
    '''
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
    
    # Only for image display
    def imshow(self, x):
        """Display an image tensor.

        Args:
            x (torch.Tensor): Image tensor to display.

        Returns:
            PIL.Image.Image: The displayed image.
        """
                
        x_ = (x.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).detach().clone().to(torch.uint8)
        img = PIL.Image.fromarray(x_[0].cpu().numpy(), 'RGB')
        #display(img)  
        return img
    
    # The learning rate adjustment function.
    def get_lr(self, t, initial_lr, rampdown=0.50, rampup=0.05):
        """Adjust the learning rate based on the current iteration.

        Args:
            t (float): Current iteration as a fraction of total iterations.
            initial_lr (float): Initial learning rate.
            rampdown (float, optional): Rampdown factor. Defaults to 0.50.
            rampup (float, optional): Rampup factor. Defaults to 0.05.

        Returns:
            float: Adjusted learning rate.
        """
                
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)

        return initial_lr * lr_ramp
    
    def get_x(self, w):
        """Get an image from the given latent vector.

        Args:
            w (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Obtained image.
        """
                
        x0, _ = self.net.decoder(w, truncation=1, input_is_latent=True, real=True)
        return x0
    def get_w(self):
        """Get the latent vector.

        Returns:
            torch.Tensor: Latent vector.
            real: True means that the latent code corresponds to a real image found via GAN inversion
        """
        with torch.no_grad():
            if self.real  == 1:
                w0 = torch.load( os.path.join(self.opts.base_path, f"embeddings/test/PTI/{self.opts.img_name.split('.')[0]}/0.pt"))
            else:
                w = torch.from_numpy(np.random.RandomState(self.opts.seed).randn(1, 512)).float().to(self.opts.device)
                x0, w0 = self.net.decoder([w], return_latents=True, truncation=self.opts.truncation, truncation_latent=self.net.mean_latent, real = 0)
        return w0
    
    def get_xw(self, w_in=None):
        """Get the generated image and latent vector.
        We assumed mostly that real is true
        Args:
            w_in (torch.Tensor, optional): Input latent vector. Defaults to None.

        Returns:
            torch.Tensor: Generated image.
            torch.Tensor: Latent vector.
        """

        with torch.no_grad():
            if self.real  == 1:
                w0 = self.get_w() ## read from a file
                x0, _ = self.net.decoder(w0, truncation=self.opts.truncation, truncation_latent=self.net.mean_latent, input_is_latent=True, real=True)
            else:
                if w_in is None:
                    w = torch.from_numpy(np.random.RandomState(self.opts.seed).randn(1, 512)).float().to(self.opts.device)
                else:
                    w = w_in
                x0, w0 = self.net.decoder([w], return_latents=True, truncation=self.opts.truncation, truncation_latent=self.net.mean_latent, real = 0)
        return x0, w0
        

    def train(self, network_pkl = None, w_in = None):
        """Train the model.

        Args:
            network_pkl (str, optional): Path to save the network weights. Defaults to None.
            w_in (torch.Tensor, optional): Input latent vector. Defaults to None.

        Returns:
            torch.Tensor: Generated image.
            torch.Tensor: Final latent vector.
            torch.Tensor: Real image.
            torch.Tensor: Initial latent vector.
        """

        #Modify this line if you want to inject to selection of the layers   
        #layers = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])
        layers = torch.LongTensor([i for i in range(18)]) 
        
        x0, w0 = self.get_xw(w_in = None)
        clip_loss = CLIPLoss(self.opts)
        txt = torch.cat([clip.tokenize(self.opts.description)]).to(self.opts.device)

        # Initialize the latent vector to be updated.
        mask = torch.zeros_like(w0.clone().detach())
        mask[:, layers, :] = 1
        mask.requires_grad = False
        
        dw = torch.empty_like(w0.clone().detach())
        torch.nn.init.uniform_(dw, -1e-3, 1e-3)
        dw.requires_grad = True

        
        optimizer = torch.optim.Adam([dw], lr=self.opts.lr)
        for i in range(self.opts.step):
            # Adjust the learning rate.
            t = i /self.opts.step
            lr = self.get_lr(t, self.opts.lr)
            optimizer.param_groups[0]["lr"] = lr

            # Generate an image using the latent vector. what about truncation!
            w_hat2 = mask * (w0 + 0.1 * dw) + (1 - mask) * w0
            x_hat, _ = self.net.decoder(w_hat2, input_is_latent=True, randomize_noise=False, real = 1, truncation=self.opts.truncation, truncation_latent=self.net.mean_latent)

            # Calculate the loss value.
            c_loss = clip_loss(x_hat, txt)
            l2_loss =  self.latent_l2_loss(w0, w_hat2)#     ((w0 - w_hat) ** 2).sum()
            id_loss = self.id_loss(x0, x_hat)

            if self.opts.id_lambda>0:
                loss = c_loss + self.opts.l2_lambda * l2_loss + self.opts.id_lambda * id_loss
            else:
                loss = c_loss + self.opts.l2_lambda * l2_loss 

            # Get gradient and update the latent vector.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the current state.
            if i%self.opts.print_every_n_step == 0:
                with torch.no_grad():
                    print(f"it: {i:04d}, lr: {lr:.4f}, loss: {loss.item():.8f}, dw: {self.latent_l2_loss(w0, w_hat2).item():.8f}")

        print("Done!")
        with torch.no_grad():
            img_gen, w_final = self.net.decoder(w_hat2, return_latents=True, input_is_latent=True, randomize_noise=False, real = True)
        return img_gen.detach().clone().cpu(), w_final.detach().clone().cpu(), x0.detach().clone().cpu(), w0.detach().clone().cpu()
  
    def get_real_image(self):
        real_img_path = os.path.join(self.opts.image_path, self.opts.img_name)
        real_image = self.transform(PIL.Image.open(real_img_path).convert('RGB'))
        return real_image.unsqueeze(0)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--description', default='A long sleeve', type=str, help='The description according to which the image is edited')
    parser.add_argument('--lr', default=0.1, type=float, help='The base learning rate')
    parser.add_argument('--lr_rampup', default=0.05, type=float)
    parser.add_argument('--lr_rampdown', default=0.05, type=float)
    parser.add_argument('--step', default=51, type=int)
    parser.add_argument('--print_every_n_step', default=50, type=int)
    parser.add_argument('--save_intermediate_image_every', default=350, type=int)
    parser.add_argument('--clip_lambda', default=1, type=float, help='clip loss coeffient')
    parser.add_argument('--l2_lambda', default=1, type=float, help='l2 loss coeffient')
    parser.add_argument('--id_lambda', default=5, type=float, help='identity loss coeffient')
    parser.add_argument('--multi_id', default=0, type=int, help='if multi pti inversion is used (we used only single pti inversion)')
    parser.add_argument('--stylegan_weights', default=None, type=str, help='For pti single model should be none, it is read based on base_path')
    parser.add_argument('--image_path', default="/home/jovyan/work/dataset/ds_200", type=str, help='path of the image')
    parser.add_argument('--base_path', default="/home/jovyan/work/styleGAN-Human/outputs/ds_200", type=str, help='path for embedings')   
    parser.add_argument('--results_dir', default="/home/jovyan/work/results/Op", type=str, help='path to save the result files')  
    
    parser.add_argument('--img_name', default="image_000003.jpg", type=str, help='The name of the source image') 
    parser.add_argument('--truncation', default=1.0, type=float, help='Use 1 if the image is real in constrast with an  image generated from a seed') 
    
    ########################
    # sample run code, please uncomment the args first and set the approperiate values
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
    ########################
    return parser


def main(args):
    c = Coach(args)
    x_real=c.get_real_image()
    x1, w1, x0, w0 = c.train()
    name = args.img_name.split('.')[0]
    torchvision.utils.save_image(torch.cat([x_real.detach().cpu(), x1.detach().cpu()]), f'{args.results_dir}/{name}.jpg' , normalize=True, scale_each=True, range=(-1, 1), nrow=1)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device('cuda:0') 

    if not os.path.exists(args.results_dir): 
        os.makedirs(args.results_dir, exist_ok=True)

    main(args)
