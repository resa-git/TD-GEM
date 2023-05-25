import torch
from torch.utils.data import Dataset
from mapper2.training import train_utils
import os

class LatentsDataset(Dataset):

	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts
		if latents=='train':
			self.imgs = sorted(train_utils.make_dataset(self.opts.latents_train_path))
			self.emb = [os.path.join(opts.latents_train_path2, 'embeddings/test/PTI', os.path.basename(im).rsplit('.')[0], '0.pt') for im in self.imgs]
			self.gen = [os.path.join(opts.latents_train_path2, 'checkpoints', 'model_'+os.path.basename(im).rsplit('.')[0]+'.pth') for im in self.imgs]
			self.latents = torch.cat([torch.load(w_path) for w_path in self.emb ])
		else:
			self.imgs = sorted(train_utils.make_dataset(self.opts.latents_test_path))
			self.emb = [os.path.join(opts.latents_test_path2, 'embeddings/test/PTI', os.path.basename(im).rsplit('.')[0], '0.pt') for im in self.imgs]
			self.gen = [os.path.join(opts.latents_test_path2, 'checkpoints', 'model_'+os.path.basename(im).rsplit('.')[0]+'.pth') for im in self.imgs]
			self.latents = torch.cat([torch.load(w_path) for w_path in self.emb ])

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, index):
		return self.latents[index], self.gen[index], self.imgs[index]

class StyleSpaceLatentsDataset(Dataset):

	def __init__(self, latents, opts):
		padded_latents = []
		for latent in latents:
			latent = latent.cpu()
			if latent.shape[2] == 512:
				padded_latents.append(latent)
			else:
				padding = torch.zeros((latent.shape[0], 1, 512 - latent.shape[2], 1, 1))
				padded_latent = torch.cat([latent, padding], dim=2)
				padded_latents.append(padded_latent)
		self.latents = torch.cat(padded_latents, dim=2)
		self.opts = opts

	def __len__(self):
		return len(self.latents)

	def __getitem__(self, index):
		return self.latents[index]
