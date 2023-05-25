import os
import run_pti
#import edit
import edit_new
#os.chdir("/home/jovyan/vol/StyleGAN-Human")


real_data_path = '/home/jovyan/work/StyleGAN-Human/aligned_image/latent_image'
latent_data_path = '/home/jovyan/work/StyleGAN-Human/outputs/temp'


print(1)


import cv2
from PIL import Image
from matplotlib import pyplot as plt

cv2.imshow()
cv2.waitKey(0)

plt.imshow(img/255)
plt.savefig('foo.png')

plt.show()

Image.fromarray(data).show()

data = {}
data['img'] = img 
data['trans_info'] = []
img1 = self.compose.transforms[0](data)['img']

img1 = self.compose({'img':img})['img']

plt.imshow(img1.transpose(2, 1, 0).clip(0, 1))
plt.savefig('foo.png')