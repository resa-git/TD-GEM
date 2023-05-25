import sys
import os
sys.path.append('../../../styleGAN-Human/')
import legacy
fold = '../../../styleGAN-Human/outputs/ds_2000/checkpoints'
Gs = os.listdir(fold)
for g in Gs:
    ckpt_path = os.path.join(fold, g)
    print(ckpt_path)
    legacy.convert(ckpt_path, ckpt_path.replace('.pkl','.pth'), G_only=True)
    break