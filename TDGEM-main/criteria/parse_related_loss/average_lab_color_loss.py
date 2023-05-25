import torch
from torch import nn
from criteria.parse_related_loss.unet import unet
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict
import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import torchvision

sys.path.append('./criteria/parse_related_loss')

from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr


class AvgLabLoss(torch.nn.Module):
    def __init__(self, opts):
        super(AvgLabLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.bg_mask_l2_loss = torch.nn.MSELoss()
        self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
        print('Loading UNet for AvgLabLoss')
        self.parsenet = deeplab_xception_transfer.deeplab_xception_transfer_projection_v3v5_more_savemem(n_classes=20, os=16, hidden_layers=128,source_classes=7,)
        self.parsenet.load_state_dict(torch.load(opts.parsenet_weights))
        self.parsenet.eval()
        self.shrink = torch.nn.AdaptiveAvgPool2d((512, 512))
        self.magnify = torch.nn.AdaptiveAvgPool2d((1024, 1024))
        self.device = torch.device('cuda')#torch.device('cuda:0')
        self.label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

    def gen_hair_mask2(self, input_image):
        labels_predict = self.parsenet(self.shrink(input_image)).detach()
        mask_512 = (torch.unsqueeze(torch.max(labels_predict, 1)[1], 1)==13).float()
        mask_1024 = self.magnify(mask_512)
        return mask_1024
    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def flip_cihp(self, tail_list):
        '''

        :param tail_list: tail_list size is 1 x n_class x h x w
        :return:
        '''
        # tail_list = tail_list[0]
        tail_list_rev = [None] * 20
        for xx in range(14):
            tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
        tail_list_rev[14] = tail_list[15].unsqueeze(0)
        tail_list_rev[15] = tail_list[14].unsqueeze(0)
        tail_list_rev[16] = tail_list[17].unsqueeze(0)
        tail_list_rev[17] = tail_list[16].unsqueeze(0)
        tail_list_rev[18] = tail_list[19].unsqueeze(0)
        tail_list_rev[19] = tail_list[18].unsqueeze(0)
        return torch.cat(tail_list_rev,dim=0)


    def decode_labels(self, mask, num_images=1, num_classes=20):
        """Decode batch of segmentation masks.

        Args:
          mask: result of inference after taking argmax.
          num_images: number of images to decode from the batch.
          num_classes: number of classes to predict (including background).

        Returns:
          A batch with num_images RGB images of the same size as the input.
        """
        n, h, w = mask.shape
        assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_, j_] = self.label_colours[k]
            outputs[i] = np.array(img)
        return outputs

    def read_img(self, img_path):
        _img = Image.open(img_path).convert('RGB')  # return is RGB pic
        return _img

    def img_transform(self, img, transform=None):
        sample = {'image': img, 'label': 0}

        sample = transform(sample)
        return sample
    
    def gen_hair_mask(self, img):
            # 0:background	1:hat	2:hair	3:-	4:sunglass	5:shirt	6:dress
            # 7:coats	8:-	9:pant	10:neck 	11:scarf	12:skirt	13:face
            # 14:left arm	15:right arm	16:left leg	17:right leg	18:left shoe	19:right shoe
            # first mask 5, 6, 7, 10, 14, 15

        img = torchvision.transforms.ToPILImage()(img.squeeze(0)).convert("RGB")
        adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
        # adj2 = adj2_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20).transpose(2, 3)
        adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).to(self.device).transpose(2, 3)

        adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
        # adj3 = adj1_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 7)
        adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).to(self.device)

        # adj2 = torch.from_numpy(graph.cihp2pascal_adj).float()
        # adj2 = adj2.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 7, 20)
        cihp_adj = graph.preprocess_adj(graph.cihp_graph)
        adj3_ = Variable(torch.from_numpy(cihp_adj).float())
        # adj1 = adj3_.unsqueeze(0).unsqueeze(0).expand(opts.gpus, 1, 20, 20)
        adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).to(self.device)

        # multi-scale
        scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]

        testloader_list = []
        testloader_flip_list = []
        for pv in scale_list:
            composed_transforms_ts = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            composed_transforms_ts_flip = transforms.Compose([
                # tr.Keep_origin_size_Resize(max_size=(1024, 1024)),
                # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.Scale_only_img(pv),
                tr.HorizontalFlip_only_img(),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img()])

            testloader_list.append(self.img_transform(img, composed_transforms_ts))
            # print(img_transform(img, composed_transforms_ts))
            testloader_flip_list.append(self.img_transform(img, composed_transforms_ts_flip))
        # 1 0.5 0.75 1.25 1.5 1.75 ; flip:
        for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
            inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
            inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
            inputs = inputs.unsqueeze(0)
            inputs_f = inputs_f.unsqueeze(0)
            inputs = torch.cat((inputs, inputs_f), dim=0)
            if iii == 0:
                _, _, h, w = inputs.size()

            # Forward pass of the mini-batch
            inputs = Variable(inputs, requires_grad=False)

            with torch.no_grad():
                inputs = inputs.to(self.device)
                # outputs = parsenet.forward(inputs)
                outputs = self.parsenet.forward(inputs, adj1_test, adj3_test, adj2_test)
                outputs = (outputs[0] + self.flip(self.flip_cihp(outputs[1]), dim=-1)) / 2
                outputs = outputs.unsqueeze(0)

                if iii > 0:
                    outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                    outputs_final = outputs_final + outputs
                else:
                    outputs_final = outputs.clone()
        ################ plot pic
        predictions = torch.max(outputs_final, 1)[1]
        results = predictions.cpu().numpy()
        vis_res = self.decode_labels(results)

        imga = torch.Tensor(vis_res[0]).permute(2,0,1)
        mask = torch.zeros(h, w, dtype=torch.long)
        idx0 = (imga==torch.tensor(self.label_colours[5], dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        idx1 = (imga==torch.tensor(self.label_colours[6], dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        idx2 = (imga==torch.tensor(self.label_colours[7], dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        idx3 = (imga==torch.tensor(self.label_colours[10], dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        idx4 = (imga==torch.tensor(self.label_colours[14], dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        idx5 = (imga==torch.tensor(self.label_colours[15], dtype=torch.uint8).unsqueeze(1).unsqueeze(2))

        validx0 = (idx0.sum(0) == 3)
        validx1 = (idx1.sum(0) == 3)
        validx2 = (idx2.sum(0) == 3)
        validx3 = (idx3.sum(0) == 3)
        validx4 = (idx4.sum(0) == 3)
        validx5 = (idx5.sum(0) == 3)

        validx = torch.logical_or(validx0,validx1)
        validx = torch.logical_or(validx,validx2)
        validx = torch.logical_or(validx,validx3)
        validx = torch.logical_or(validx,validx4)
        validx = torch.logical_or(validx,validx5)

        mask[validx] = torch.tensor(1, dtype=torch.long)
        return mask.to(self.device)
        #     # cal lab written by liuqk
    def f(self, input):
        output = input * 1
        mask = input > 0.008856
        output[mask] = torch.pow(input[mask], 1 / 3)
        output[~mask] = 7.787 * input[~mask] + 0.137931
        return output

    def rgb2xyz(self, input):
        assert input.size(1) == 3
        M_tmp = self.M.to(input.device).unsqueeze(0)
        M_tmp = M_tmp.repeat(input.size(0), 1, 1)  # BxCxC
        output = torch.einsum('bnc,bchw->bnhw', M_tmp, input)  # BxCxHxW
        M_tmp = M_tmp.sum(dim=2, keepdim=True)  # BxCx1
        M_tmp = M_tmp.unsqueeze(3)  # BxCx1x1
        return output / M_tmp

    def xyz2lab(self, input):
        assert input.size(1) == 3
        output = input * 1
        xyz_f = self.f(input)
        # compute l
        mask = input[:, 1, :, :] > 0.008856
        output[:, 0, :, :][mask] = 116 * xyz_f[:, 1, :, :][mask] - 16
        output[:, 0, :, :][~mask] = 903.3 * input[:, 1, :, :][~mask]
        # compute a
        output[:, 1, :, :] = 500 * (xyz_f[:, 0, :, :] - xyz_f[:, 1, :, :])
        # compute b
        output[:, 2, :, :] = 200 * (xyz_f[:, 1, :, :] - xyz_f[:, 2, :, :])
        return output
    def cal_hair_avg(self, input, mask):
        x = input * mask
        mask = mask.unsqueeze(0).unsqueeze(0)
        sum = torch.sum(torch.sum(x, dim=2, keepdim=True), dim=3, keepdim=True) # [n,3,1,1]
        mask_sum = torch.sum(torch.sum(mask, dim=2, keepdim=True), dim=3, keepdim=True) # [n,1,1,1]
        mask_sum[mask_sum == 0] = 1
        avg = sum / mask_sum
        return avg

    def forward(self, fake, real):
        #the mask is [n,1,h,w]
        #normalize to 0~1
        mask_fake = self.gen_hair_mask(fake) #bring out the clothing area
        mask_real = self.gen_hair_mask(real)
        fake_RGB = (fake + 1) / 2.0
        real_RGB = (real + 1) / 2.0
        #from RGB to Lab by liuqk
        fake_xyz = self.rgb2xyz(fake_RGB)
        fake_Lab = self.xyz2lab(fake_xyz)
        real_xyz = self.rgb2xyz(real_RGB)
        real_Lab = self.xyz2lab(real_xyz)
        #cal average value
        fake_Lab_avg = self.cal_hair_avg(fake_Lab, mask_fake)
        real_Lab_avg = self.cal_hair_avg(real_Lab, mask_real)

        loss = self.criterion(fake_Lab_avg, real_Lab_avg)
        mask = torch.logical_or(mask_fake, mask_real)*1
        lossb = self.bg_mask_l2_loss(fake*(1 - mask), real*(1 - mask))
        return loss, lossb



# class AvgLabLoss(nn.Module):
#     def __init__(self, opts):
#         super(AvgLabLoss, self).__init__()
#         self.criterion = nn.L1Loss()
#         self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
#         print('Loading UNet for AvgLabLoss')
#         self.parsenet = unet()
#         self.parsenet.load_state_dict(torch.load(opts.parsenet_weights))
#         self.parsenet.eval()
#         self.shrink = torch.nn.AdaptiveAvgPool2d((512, 512))
#         self.magnify = torch.nn.AdaptiveAvgPool2d((1024, 1024))

#     def gen_hair_mask(self, input_image):
#         labels_predict = self.parsenet(self.shrink(input_image)).detach()
#         mask_512 = (torch.unsqueeze(torch.max(labels_predict, 1)[1], 1)==13).float()
#         mask_1024 = self.magnify(mask_512)
#         return mask_1024

