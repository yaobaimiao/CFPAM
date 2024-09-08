import argparse
import os
import time

import numpy as np
import pytorch_toolbelt.losses as PTL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from adv import Discriminator
from config import Config
from dataset import get_loader
from dataset2 import build_data_loader
from evaluation.dataloader import EvalDataset
from evaluation.evaluator import Eval_thread
from loss import *
from matplotlib import pyplot as plt
from models.main import *
from torch.optim import *
from tqdm import tqdm
from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Parameter from command line

parser = argparse.ArgumentParser(description='')
parser.add_argument("-img_root", type=str, default="./dataset/train_data/DUTS_class/img")  ###########
parser.add_argument("-co_gt_root", type=str, default="./dataset/train_data/DUTS_class/gt")  ##########
parser.add_argument("-img_root_coco", type=str, default="./dataset/train_data/CoCo9k/img")  ################
parser.add_argument("-co_gt_root_coco", type=str, default="./dataset/train_data/CoCo9k/gt")  #####################
parser.add_argument("-img_syn_root", type=str, default="./dataset/train_data/DUTS_class_syn/"  ######################
                                                       "img_png_seamless_cloning_add_naive/img")
parser.add_argument("-img_rev_syn_root", type=str,
                    default="./dataset/train_data/DUTS_class_syn/"  ##########################
                            "img_png_seamless_cloning_add_naive_reverse_2/img")
parser.add_argument("-co_gt_rev_syn_root", type=str,
                    default="./dataset/train_data/DUTS_class_syn/"  ###########################
                            "img_png_seamless_cloning_add_naive_reverse_2/gt")
parser.add_argument("-train_w_coco_prob", type=float, default=0.5)  #######################
parser.add_argument("-max_num", type=int, default=8)  #############################
parser.add_argument("-img_size", type=int, default=256)  ####################################
parser.add_argument("-scale_size", type=int, default=288)  #####################################

parser.add_argument('--loss',
                    default='Scale_IoU',
                    type=str,
                    help="Options: '', ''")
parser.add_argument("-batch_size", default=1, type=int)
parser.add_argument('--bs', '--batch_size', default=1, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1e-4,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=220, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='CoCo',
                    type=str,
                    help="Options: 'CoCo'")
parser.add_argument('--testsets',
                    default='CoCA',
                    type=str,
                    help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
parser.add_argument('--size',
                    default=256,
                    type=int,
                    help='input size')
parser.add_argument('--tmp', default='./data1/cfpam/ceshi', help='Temporary folder')
parser.add_argument('--save_root', default='./CoSODmaps/pred2', type=str, help='Output folder')
parser.add_argument('--freeze', default=True)
args = parser.parse_args()
config = Config()

train_loader = build_data_loader(args, mode='train')

for testset in ['CoCA']:
    if testset == 'CoCA':
        test_img_path = './dataset/test_data/CoCA/Image/'
        test_gt_path = './dataset/test_data/CoCA/GroundTruth/'
        saved_root = os.path.join(args.save_root, 'CoCA')
    elif testset == 'CoSOD3k':
        test_img_path = './data/images/CoSOD3k/'
        test_gt_path = './data/gts/CoSOD3k/'
        saved_root = os.path.join(args.save_root, 'CoSOD3k')
    elif testset == 'CoSal2015':
        test_img_path = './data/images/CoSal2015/'
        test_gt_path = './data/gts/CoSal2015/'
        saved_root = os.path.join(args.save_root, 'CoSal2015')
    elif testset == 'CoCo':
        test_img_path = './data/images/CoCo/'
        test_gt_path = './data/gts/CoCo/'
        saved_root = os.path.join(args.save_root, 'CoCo')
    else:
        print('Unkonwn test dataset')
        print(args.dataset)

    test_loader = get_loader(
        test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=16, pin=True)

# make dir for tmp
os.makedirs(args.tmp, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.tmp, "log.txt"))
set_seed(123)

# Init model
device = torch.device("cuda")

model = CFPAM()
model = model.to(device)
model.apply(weights_init)

path = './models/pvt_v2_b5_.pth'
save_model = torch.load(path)
model_dict = model.cfpamnet.backbone.state_dict()  # backbone parameter dictionary
state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
model.cfpamnet.backbone.load_state_dict(model_dict)
# model.cfpamnet.load_state_dict(save_model)

backbone_params = list(map(id, model.cfpamnet.backbone.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params,
                     model.cfpamnet.parameters())
all_params = [{'params': base_params}, {'params': model.cfpamnet.backbone.parameters(), 'lr': 0.025 * args.lr}]

# Setting optimizer
optimizer = optim.Adam(params=all_params, lr=args.lr, weight_decay=0)
disc = Discriminator(channels=3, img_size=args.size).to(device)
optimizer_disc = optim.Adam(params=all_params, lr=args.lr, weight_decay=0)
# setting scheduler
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.996)
# scheduler_disc = lr_scheduler.ExponentialLR(optimizer_disc, gamma=0.99)

adv_criterion = nn.BCELoss().to(device)
bce_criterion = nn.BCELoss().to(device)
for key, value in model.named_parameters():
    print(key, value.requires_grad)

# log model and optimizer pars
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Optimizer_disc details:")
logger.info(optimizer_disc)
logger.info("Scheduler details:")
logger.info(scheduler)

logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
exec('from loss import ' + args.loss)
IOUloss = eval(args.loss + '()')


def main():
    val_measures = []
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.cfpamnet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    loss_iou1 = []
    loss_scl1 = []
    loss_bce1 = []
    adv_loss_g1 = []
    adv_loss_disc1 = []
    legend_added = False
    for epoch in range(args.start_epoch, args.epochs):
        loss_iou, loss_scl, loss_bce, adv_loss_g, adv_loss_disc = train(epoch)
        loss_iou1.append(loss_iou)
        loss_scl1.append(loss_scl)
        loss_bce1.append(loss_bce)
        # loss_mfinal1.append(loss_mfinal)
        adv_loss_g1.append(adv_loss_g)
        adv_loss_disc1.append(adv_loss_disc)
        plt.figure(figsize=(10, 10))
        plt.plot(loss_iou1, color='red', label='iou')
        plt.plot(loss_scl1, color='b', label='scl')
        plt.plot(loss_bce1, color='g', label='bce')
        plt.plot(adv_loss_g1, color='m', label='loss_g')
        plt.plot(adv_loss_disc1, color='y', label='loss_disc')
        if not legend_added:
            plt.legend(loc='best')
            legend_added = True
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        # plt.close()
        logger.info('Epoch[{0:03d}/{1}],Learning Rate:{2}'.format(epoch, args.epochs, scheduler.get_last_lr()))
        # logger.info('Epoch[{0:03d}/{1}],Learning Rate:{2}'.format(epoch, args.epochs, scheduler_disc.get_last_lr()))
        scheduler.step()  # Renewal lr
        # scheduler_disc.step()
        if config.validation:
            measures = validate(model, test_loader, args.testsets)
            val_measures.append(measures)
            logger.info(
                'Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with S_measure {:.4f}'.format(
                    epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
                    np.max(np.array(val_measures)[:, 0]))
            )
            # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.cfpamnet.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            },
            path=args.tmp)
        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.tmp, weight_file) for weight_file in
                                       os.listdir(args.tmp) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model.cfpamnet.state_dict(),
                           os.path.join(args.tmp, 'best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))
        if measures[0] >= 0.7500:
            torch.save(model.cfpamnet.state_dict(),
                       os.path.join(args.tmp, 'ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))
        if (epoch + 1) % 9990 == 0 or epoch == 0:
            torch.save(model.cfpamnet.state_dict(), args.tmp + '/model-' + str(epoch + 1) + '.pt')

        if epoch >= 199:
            torch.save(model.cfpamnet.state_dict(), args.tmp + '/model-' + str(epoch + 1) + '.pt')
    # cfpamnet_dict = model.cfpamnet.state_dict()
    # torch.save(cfpamnet_dict, os.path.join(args.tmp, 'final.pth'))


def sclloss(x, xt, xb):
    # print(x.shape)
    cosc = (1 + compute_cos_dis(x, xt)) * 0.5
    cosb = (1 + compute_cos_dis(x, xb)) * 0.5
    # print(compute_cos_dis(x, xt))
    # print(compute_cos_dis(x, xb))
    loss = -torch.log(cosc + 1e-7) - torch.log(1 - cosb + 1e-7)
    return loss.sum()


def bceloss(pred, gts):
    loss = 0
    for i in range(len(pred)):
        loss += (bce_criterion(torch.sigmoid(pred[i]), gts) + bce_criterion(1 - torch.sigmoid(pred[i]), 1 - gts)) / 2
    return loss / len(pred)


def train(epoch):
    # Switch to train mode
    model.train()
    model.set_mode('train')
    loss_iou_sum = []
    loss_scl_sum = []
    loss_bce_sum = []
    adv_loss_g_sum = []
    adv_loss_disc_sum = []

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["imgs"].to(device).squeeze(0)
        gts = batch["co_gts"].to(device).squeeze(0)
        pred, proto, protogt, protobg = model(inputs, gts)  # pred  3* torch.Size([8, 1, 224, 224])
        valid = torch.ones((pred[-1].shape[0], 1), requires_grad=False).to(device)
        fake = torch.zeros((pred[-1].shape[0], 1), requires_grad=False).to(device)
        # G
        optimizer.zero_grad()
        # new loss1
        adv_loss_g = adv_criterion(disc(pred[-1] * inputs), valid)
        # old loss1
        loss_iou = IOUloss(pred, gts)
        # old loss2
        loss_scl = 0.1 * sclloss(proto, protogt, protobg)
        # new loss4
        loss_bce = 0.1 * bceloss(pred, gts)
        # sum loss
        loss = loss_iou + adv_loss_g + loss_scl + loss_bce
        loss.backward()
        optimizer.step()
        # D
        # The discriminator is disc and the pred[-1] is the generator!
        # inputs may not be added.detach()
        optimizer_disc.zero_grad()
        adv_loss_real = adv_criterion(disc(gts * inputs), valid)
        adv_loss_fake = adv_criterion(disc(pred[-1].detach() * inputs.detach()), fake)
        # new loss2
        adv_loss_disc = (adv_loss_real + adv_loss_fake) / 2
        adv_loss_disc.backward()
        optimizer_disc.step()

        # # sum
        loss_iou_sum.append(float(loss_iou))
        loss_scl_sum.append(float(loss_scl))
        loss_bce_sum.append(float(loss_bce))
        adv_loss_g_sum.append(float(adv_loss_g))
        adv_loss_disc_sum.append(float(adv_loss_disc))
        if batch_idx % 20 == 0:
            logger.info('Epoch[{0:03d}/{1}] Iter[{2:03d}/{3}]  '
                        'Train Loss:loss_iou:{4:.4f},  loss_scl:{5:.4f},  loss_bce:{6:.4f},  adv_loss_g:{7:.4f},  adv_loss_disc:{8:.6f}'.format(
                epoch,
                args.epochs,
                batch_idx,
                len(train_loader),
                loss_iou,
                loss_scl,
                loss_bce,
                adv_loss_g,
                adv_loss_disc
            ))
    print('\n',
          '-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*')
    logger.info(
        'Avg_loss_iou:{0:.4f},  Avg_loss_scl:{1:.4f},  Avg_loss_bce:{2:.4f},  Avg_adv_loss_g:{3:.4f},  Avg_adv_loss_disc:{4:.6f}'.format(
            sum(loss_iou_sum) / len(loss_iou_sum),
            sum(loss_scl_sum) / len(loss_scl_sum),
            sum(loss_bce_sum) / len(loss_bce_sum),
            sum(adv_loss_g_sum) / len(adv_loss_g_sum),
            sum(adv_loss_disc_sum) / len(adv_loss_disc_sum),
        ))
    print('-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*-#-*',
          '\n')
    return sum(loss_iou_sum) / len(loss_iou_sum), sum(loss_scl_sum) / len(loss_scl_sum), sum(loss_bce_sum) / len(
        loss_bce_sum), sum(adv_loss_g_sum) / len(adv_loss_g_sum), sum(adv_loss_disc_sum) / len(adv_loss_disc_sum)


def validate(model, test_loaders, testsets):
    model.eval()

    testsets = testsets.split('+')
    measures = []
    for testset in testsets[:1]:
        print('Validating {}...'.format(testset))
        # test_loader = test_loaders[testset]

        saved_root = os.path.join(args.save_root, testset)

        for batch in test_loader:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs, gts)[-1].sigmoid()

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))

        eval_loader = EvalDataset(
            saved_root,  # preds
            os.path.join('./dataset/test_data/CoCA/GroundTruth')  # GT
        )
        evaler = Eval_thread(eval_loader, cuda=True)
        # Use S_measure for validation
        s_measure = evaler.Eval_Smeasure()
        if s_measure > config.val_measures['Smeasure']['CoCA'] and 0:
            # TODO: evluate others measures if s_measure is very high.
            e_max = evaler.Eval_Emeasure().max().item()
            f_max = evaler.Eval_fmeasure().max().item()
            print('Emax: {:4.f}, Fmax: {:4.f}'.format(e_max, f_max))
        measures.append(s_measure)

    model.train()
    return measures


if __name__ == '__main__':
    main()
