## NEEDS tqdm, torchvision==0.7.0?, not a lot of requirements
# should clone this repository https://github.com/baegwangbin/surface_normal_uncertainty.git
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from data.dataloader_custom import CustomLoader
from models.NNET import NNET
import utils.utils as utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
## makes it so I dont have to use argparse. kinda ok 
class Params:
    def __init__(self):
        self.architecture = 'GN'
        self.pretrained = 'scannet'
        self.sampling_ratio = 0.4
        self.importance_ratio = 0.7
        self.input_height = 480
        self.input_width = 640
        self.imgs_dir = './examples'

# Create an instance of Args class
args = Params()
def test(model, test_loader, device, results_dir):
    alpha_max = 60
    kappa_max = 30

    with torch.no_grad():
        for data_dict in tqdm(test_loader):

            img = data_dict['img'].to(device)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]

            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            # to numpy arrays
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()

            # save results
            img_name = data_dict['img_name'][0]

            # 1. save input image
            img = utils.unnormalize(img[0, ...])

            target_path = '%s/%s_img.png' % (results_dir, img_name)
            plt.imsave(target_path, img)

            # 2. predicted normal
            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)                  # (B, H, W, 3)

            target_path = '%s/%s_pred_norm.png' % (results_dir, img_name)
            plt.imsave(target_path, pred_norm_rgb[0, :, :, :])
    # read arguments from txt file
device = torch.device('cuda:0')
    # load checkpoint
checkpoint = 'checkpoint-path'
model = NNET(args).to(device)
model = utils.load_checkpoint(checkpoint, model).eval()
    # test the model
results_dir = args.imgs_dir + '/results'
os.makedirs(results_dir, exist_ok=True)
test_loader = CustomLoader(args, args.imgs_dir).data
test(model, test_loader, device, results_dir)
