import torch
from torch import nn
import timm
#from models.facial_recognition.model_irse import Backbone
class IDLoss(torch.nn.Module):
    def __init__(self, opts):
        super(IDLoss, self).__init__()
        #self.m = timm.create_model('resnet50', pretrained=True, num_classes=0).to(opts.device)
        #self.m = timm.create_model('convnext_base', pretrained=True, num_classes=0).to(opts.device)
        #self.m = timm.create_model('convnext_tiny', pretrained=True, num_classes=0).to(opts.device)
        self.m = timm.create_model('convnext_tiny', num_classes=0).to(opts.device)
        
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(32, 16))
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x, x_hat):
        n_samples = x.shape[0]
        xr = self.avg_pool(self.upsample(x)) 
        xr = xr.detach()
        xr_hat = self.avg_pool(self.upsample(x_hat)) 
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            loss += self.criterion(self.m(xr), self.m(xr_hat))
            count += 1
        return loss / count, sim_improvement / count


# class IDLoss(nn.Module):
#     def __init__(self, opts):
#         super(IDLoss, self).__init__()
#         print('Loading ResNet ArcFace for ID Loss')
#         self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
#         self.facenet.load_state_dict(torch.load(opts.ir_se50_weights))
#         self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
#         self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
#         self.facenet.eval()
#         self.opts = opts

#     def extract_feats(self, x):
#         if x.shape[2] != 256:
#             x = self.pool(x)
#         x = x[:, :, 35:223, 32:220]  # Crop interesting region
#         x = self.face_pool(x)
#         x_feats = self.facenet(x)
#         return x_feats

#     def forward(self, y_hat, y):
#         n_samples = y.shape[0]
#         y_feats = self.extract_feats(y)  # Otherwise use the feature from there
#         y_hat_feats = self.extract_feats(y_hat)
#         y_feats = y_feats.detach()
#         loss = 0
#         sim_improvement = 0
#         count = 0
#         for i in range(n_samples):
#             diff_target = y_hat_feats[i].dot(y_feats[i])
#             loss += 1 - diff_target
#             count += 1

#         return loss / count, sim_improvement / count
