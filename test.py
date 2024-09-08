from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from util import save_tensor_img, Logger
from tqdm import tqdm
from torch import nn
import os
from models.main import *
import argparse
import numpy as np
import cv2
import subprocess


def main(args):
    # Init model

    device = torch.device("cuda")
    model = CFPAM()
    model = model.to(device)
    try:
        modelname = os.path.join(args.param_root, "cfpam/baseline+PVT+FPM/model-220.pt")
        cfpamnet_dict = torch.load(modelname)
        print('loaded', modelname)
    except:
        cfpamnet_dict = torch.load(os.path.join(args.param_root, 'YYYYMMMMM.pth'))

    model.to(device)
    model.cfpamnet.load_state_dict(cfpamnet_dict)
    model.eval()
    model.set_mode('test')

    tensor2pil = transforms.ToPILImage()
    for testset in ['CoCA', 'CoSOD3k', 'CoSal2015']:
        if testset == 'CoCA':
            test_img_path = './dataset/test_data/CoCA/Image/'
            test_gt_path = './dataset/test_data/CoCA/GroundTruth/'
            saved_root = os.path.join(args.save_root, 'CoCA')
        elif testset == 'CoSOD3k':
            test_img_path = './dataset/test_data/CoSOD3k/Image/'
            test_gt_path = './dataset/test_data/CoSOD3k/GroundTruth/'
            saved_root = os.path.join(args.save_root, 'CoSOD3k')
        elif testset == 'CoSal2015':
            test_img_path = './dataset/test_data/CoSal2015/Image/'
            test_gt_path = './dataset/test_data/CoSal2015/GroundTruth/'
            saved_root = os.path.join(args.save_root, 'CoSal2015')
        elif testset == 'ym':
            test_img_path = './dataset/test_data/ym/Image/'
            test_gt_path = './dataset/test_data/ym/GroundTruth/'
            saved_root = os.path.join(args.save_root, 'ym')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)

        test_loader = get_loader(
            test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=16, pin=True)

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            scaled_preds = model(inputs, gts)
            scaled_preds = torch.sigmoid(scaled_preds[-1])
            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)
            num = gts.shape[0]
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--size',
                        default=256,
                        type=int,
                        help='input size')
    parser.add_argument('--param_root', default='data1', type=str, help='model folder')
    parser.add_argument('--save_root', default='CoSODmaps/pred/CFPAM', type=str, help='Output folder')
    parser.add_argument('--eval_master', default='eval_co_sod_master/main.py', type=str)
    args = parser.parse_args()
    main(args)
    # subprocess.call(['python', args.eval_master])
