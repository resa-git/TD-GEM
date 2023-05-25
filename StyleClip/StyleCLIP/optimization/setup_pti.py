#!/usr/bin/env python
# coding: utf-8
import os

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

opts = Namespace(e4e='./pti/e4e_w+.pt',
                 stylegan2_ada_shhq='./pretrained_models/stylegan_human_v2_1024.pkl',
                 ir_se50= '' ,
                 checkpoints_dir='./outputs/ptis/checkpoints', 
                 embedding_base_dir='./outputs/ptis/embeddings',
                 experiments_output_dir='./outputs/ptis',
                 input_data_path='./aligned_image/real_image/selected',
                 input_data_id='test',
                 pti_results_keyword='PTI',
                 e4e_results_keyword='e4e',
                 sg2_results_keyword='SG2',
                 sg2_plus_results_keyword='SG2_Plus',
                 multi_id_model_type='multi_id')

def write_path_config(fpath = '', opts=opts):
    
    lines = ['import os',
            '\n',
            '## Pretrained models paths',
            f'e4e = \'{opts.e4e}\'',
            f'stylegan2_ada_shhq = \'{opts.stylegan2_ada_shhq}\'',
            f'ir_se50 = \'{opts.ir_se50}\' #\'./model_ir_se50.pth\' ',
            '\n',
            '## Dirs for output files',
            f'checkpoints_dir = \'{opts.checkpoints_dir}\'',
            f'embedding_base_dir = \'{opts.embedding_base_dir}\'',
            f'experiments_output_dir = \'{opts.experiments_output_dir}\'',
            '\n',
            '## Input info',
            '### Input dir, where the images reside',
            f'input_data_path = \'{opts.input_data_path}\'',
            '### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator',
            f'input_data_id = \'{opts.input_data_id}\'',
            '\n',
            '## Keywords',
            f'pti_results_keyword = \'{opts.pti_results_keyword}\'',
            f'e4e_results_keyword = \'{opts.e4e_results_keyword}\'',
            f'sg2_results_keyword = \'{opts.sg2_results_keyword}\'', 
            f'sg2_plus_results_keyword = \'{opts.sg2_plus_results_keyword}\'',
            f'multi_id_model_type = \'{opts.multi_id_model_type}\'']
    fname = os.path.join(fpath, "paths_config.py")
    with open(fname, "w") as f:
        for l in lines:
            f.write(l)
            f.write('\n')


## Architechture
optf = Namespace(lpips_type = 'alex',
                 first_inv_type = 'w+',
                 optim_type = 'adam',
                latent_ball_num_of_samples = 1,
                 locality_regularization_interval = 1,
                 use_locality_regularization = False,
                 regulizer_l2_lambda = 0.1,
                 regulizer_lpips_lambda = 0.1,
                 regulizer_alpha = 30,
                 pt_l2_lambda = 1,
                 pt_lpips_lambda = 1,
                 LPIPS_value_threshold = 0.0001,
                 max_pti_steps = 14500,
                 first_inv_steps = 450,
                 max_images_to_invert = 30,
                 pti_learning_rate = 5e-4,
                 first_inv_lr = 8e-3,
                 train_batch_size = 1,
                 use_last_w_pivots = False)


# In[38]:


def write_hyperparameters(fpath = '', opts=opts):
    lines = ['## Architechture',
             f'lpips_type = \'{opts.lpips_type}\'',
             f'first_inv_type = \'{opts.first_inv_type}\'',
             f'optim_type = \'{opts.optim_type}\''
             '\n',
             '## Locality regularization',
             f'latent_ball_num_of_samples = {opts.latent_ball_num_of_samples}',
             f'locality_regularization_interval = {opts.locality_regularization_interval}',
             f'use_locality_regularization = {opts.use_locality_regularization}',
             f'regulizer_l2_lambda = {opts.regulizer_l2_lambda}',
             f'regulizer_lpips_lambda = {opts.regulizer_lpips_lambda}',
             f'regulizer_alpha = {opts.regulizer_alpha}',
             '\n',
             '## Loss',
             f'pt_l2_lambda = {opts.pt_l2_lambda}',
             f'pt_lpips_lambda = {opts.pt_lpips_lambda}',
             '\n',
             '## Steps',
             f'LPIPS_value_threshold = {opts.LPIPS_value_threshold}',
             f'max_pti_steps = {opts.max_pti_steps}',
             f'first_inv_steps = {opts.first_inv_steps}',
             f'max_images_to_invert = {opts.max_images_to_invert}',
             '\n',
             '## Optimization',
             f'pti_learning_rate = {opts.pti_learning_rate}',
             f'first_inv_lr = {opts.first_inv_lr}',
             f'train_batch_size = {opts.train_batch_size}',
             f'use_last_w_pivots = {opts.use_last_w_pivots}']
    fname = os.path.join(fpath, "hyperparameters.py")
    with open(fname, "w") as f:
        for l in lines:
            f.write(l)
            f.write('\n')

#write_path_config(opts=opts)
#write_hyperparameters(fpath = '', opts = optf)





