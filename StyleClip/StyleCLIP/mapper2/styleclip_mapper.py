import torch
from torch import nn
from mapper2 import latent_mappers
from torch_utils.models import Generator


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class StyleCLIPMapper(nn.Module):

	def __init__(self, opts):
		super(StyleCLIPMapper, self).__init__()
		self.opts = opts
		# Define architecture
		self.mapper = self.set_mapper()
		#self.decoder = Generator(self.opts.stylegan_size, 512, 8)
		config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
		self.decoder = Generator(size = 1024,
								 style_dim=config["latent"],
								 n_mlp=config["n_mlp"],
								 channel_multiplier=config["channel_multiplier"])
		self.latent_avg = None
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		#self.load_weights(opts.stylegan_weights)

	def set_mapper(self):
		if self.opts.work_in_stylespace:
			mapper = latent_mappers.WithoutToRGBStyleSpaceMapper(self.opts)
		elif self.opts.mapper_type == 'SingleMapper':
			mapper = latent_mappers.SingleMapper(self.opts)
		elif self.opts.mapper_type == 'LevelsMapper':
			mapper = latent_mappers.LevelsMapper(self.opts)
		else:
			raise Exception('{} is not a valid mapper'.format(self.opts.mapper_type))
		return mapper

	def load_weights(self, stylegan_weights):
		if self.opts.checkpoint_path is not None:
			print('Loading from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.mapper.load_state_dict(get_keys(ckpt, 'mapper'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
		else:
			#print('Loading decoder weights from pretrained!')
			ckpt = torch.load(stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'])
			self.decoder.eval()
			self.latent_avg =ckpt['latent_avg']

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.mapper(x)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images
