import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.utils.data
import random
import shutil
from skimage import measure
import skimage
import logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from torchvision.utils import make_grid
from skimage import measure, io
from augmention import generate_unsup_data
from CE_Net import New_Semic_Seg
from torch.nn import functional as F
import numpy as np
import utils
from data_folder import DataFolder
from hausdorff_loss import HausdorffERLoss
from options_semi import Options
from my_transforms import get_transforms
from loss import LossVariance, dice_loss, FlowLoss
from torch import nn
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.cuda.amp import GradScaler, autocast
from NVP import RNVP
from distributions import FastGMM
import yaml
import argparse
from utils2 import clip_grad_norm, bits_per_dim
from functools import partial
from attack import attack
from scipy.signal import find_peaks
from collections import defaultdict
import seaborn as sns
import signal

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument('--alpha', type=float, default=1, help='The weight for the variance term in loss')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--save-dir', type=str, default='./experiments/GlaS_0.125/newcrop/train_2025-5-25', help='directory to save training results')
parser.add_argument('--gpu', type=list, default=[], help='GPUs for training')
def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status

to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')
to_warm_status = partial(to_status, status='warm_up')


log_density_dict = defaultdict(list)

def main():
    global opt, best_iou, num_iter, tb_writer, logger, logger_results, args, cfg, scaler_nf, scaler,best_iou
    best_iou = 0
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    scaler = GradScaler()
    scaler_nf = GradScaler()
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()
    torch.backends.cudnn.enabled = False

    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))



    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    model = New_Semic_Seg(3,3)
    teacher_model = deepcopy(model)
    for p in teacher_model.parameters():
        p.requires_grad = False

    means = torch.randn(cfg['flow']['n_components'], cfg['flow']['input_dims'])   
    nf_model = RNVP(cfg, means, learnable_mean=True)
    nf_model= nf_model.cuda()

    # model = nn.DataParallel(model,device_ids=[0])
    model = model.cuda()
    teacher_model = teacher_model.cuda()


    torch.backends.cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])
    optimizer_nf = torch.optim.Adam([{'params':nf_model.parameters()}], lr=cfg['flow']['lr'], betas=(0.5, 0.9))

    # ----- define criterion ----- #
    global mseloss, criterion_hau, loss_flow 
    prior = FastGMM(nf_model.means)

    loss_flow = FlowLoss(prior)
    mseloss = torch.nn.MSELoss(reduction='mean').cuda()
    criterion = torch.nn.NLLLoss(reduction='none').cuda()
    criterion_hau = HausdorffERLoss()
    
    if opt.train['alpha'] > 0:
        logger.info('=> Using variance term in loss...')
        global criterion_var
        criterion_var = LossVariance()

    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'train1': get_transforms(opt.transform['train1']),
                       'valA': get_transforms(opt.transform['val']),
                       'valB': get_transforms(opt.transform['val'])}


    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'valA', 'valB']:
        img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], x)
        target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], x)
        weight_map_dir = '{:s}/{:s}'.format(opt.train['weight_map_dir'], x)
        dir_list = [img_dir, weight_map_dir, target_dir]
        if opt.dataset == 'MultiOrgan':
            post_fix = ['weight.png', 'label.png']
        else:
            post_fix = ['anno_weight.png', 'anno_label.png']
        num_channels = [3, 1, 3]
        dsets[x] = DataFolder(dir_list, post_fix, num_channels, data_transforms[x])

    dsets2 = {}
    for x1 in ['train1']:
        img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], 'unlabel' + x1)
        dir_list = [img_dir]
        if opt.dataset == 'MultiOrgan':
            post_fix = ['weight.png', 'label.png']
        else:
            post_fix = []
        num_channels = [3]
        dsets2[x1] = DataFolder(dir_list, post_fix, num_channels, data_transforms[x1])

    train_loader = DataLoader(dsets['train'], batch_size=8, shuffle=True,
                              num_workers=opt.train['workers'], drop_last=False)
    train_loader1 = DataLoader(dsets2['train1'], batch_size=8, shuffle=True,
                               num_workers=opt.train['workers'], drop_last=False)

    val_loader = DataLoader(dsets['valB'], batch_size=1, shuffle=False,
                            num_workers=opt.train['workers'], drop_last=True)
    val_loader1 = DataLoader(dsets['valA'], batch_size=1, shuffle=False,
                             num_workers=opt.train['workers'], drop_last=True)




    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if opt.train['checkpoint']:
        if os.path.isfile(opt.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(opt.train['checkpoint']))
            checkpoint = torch.load(opt.train['checkpoint'])
            opt.train['start_epoch'] = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(opt.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.train['checkpoint']))



    # ----- training and validation ----- #
    for epoch in range(opt.train['start_epoch'], opt.train['num_epochs']):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch + 1, opt.train['num_epochs']))
        train_results = train(train_loader, train_loader1, model, teacher_model, nf_model, optimizer, optimizer_nf, criterion, epoch)
        train_loss, train_loss_ce, train_loss_var, train_unsup_loss, train_unsup_loss_pt,train_nf_loss, train_pixel_acc, train_iou = train_results

        # evaluate on validation set
        with torch.no_grad():

            val_loss, val_pixel_acc, val_iou = validate(val_loader, model, criterion)
            val_loss1, val_pixel_acc1, val_iou1 = validate(val_loader1, model, criterion)


            is_best = val_iou > best_iou
            best_iou = max(val_iou, best_iou)


            if (val_iou >= 0.79):
                # val_loss1, val_pixel_acc1, val_iou1 = validate(val_loader1, model, criterion)
                is_second = val_iou1 >= 0.79
                cp_flag = (epoch + 1) % opt.train['checkpoint_freq'] == 0
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_iou': best_iou,
                    'optimizer': optimizer.state_dict(),
                }, epoch, is_best, opt.train['save_dir'], cp_flag, is_second)


        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch + 1, train_loss, train_loss_ce, train_loss_var, train_pixel_acc,
                                    train_iou, val_loss,val_iou))
        # tensorboard logs
        tb_writer.add_scalars('epoch_losses',
                              {'train_loss': train_loss, 'train_loss_ce': train_loss_ce,
                               'train_unsup_loss': train_unsup_loss, 'train_unsup_loss_pt': train_unsup_loss_pt,
                                'train_nf_loss': train_nf_loss,
                               'train_loss_var': train_loss_var, 'val_loss': val_loss}, epoch)


        tb_writer.add_scalars('student_epoch_accuracies',
                              {'train_pixel_acc': train_pixel_acc, 'train_iou': train_iou,
                               'val_iouB': val_iou, 'val_iouA': val_iou1}, epoch)


    tb_writer.close()




def train(train_loader, train_loader1, model, teacher_model, nf_model, optimizer, optimizer_nf,criterion, epoch):
    ite = 0
    # Loss_list = list()

    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(8)
    # switch to train mode

    label_iter = iter(train_loader1)

    for i, sample in enumerate(train_loader):
        ite += 1
        input, weight_map, target = sample
        weight_map = weight_map.float().div(20)
        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        weight_map_var = weight_map.cuda()
        if torch.max(target) == 255:
            target = target / 255
        if target.dim() == 4:
            target = target.squeeze(1)
        try:
            input1 = next(label_iter)
        except StopIteration:
            label_iter = iter(train_loader1)
            input1 = next(label_iter)

        input_var = input.cuda()
        target_var = target.cuda()
        input_var1 = input1[0].cuda()
 

        # compute teacher_model,teacher_model predicts on all data
        teacher_model.train()
        with torch.no_grad():
            out_labeled,out_unlabeled,output_rep = teacher_model(input_var,input_var1,label=None, nf_model=None, loss_flow=None, cfg=None, eps=0, adv=False)


            output_ema_u = F.softmax(out_unlabeled, dim=1)
            logits_u_aug, label_u = torch.max(output_ema_u,dim=1)

        input_var1_aug, label_u_aug, logits_u_aug = generate_unsup_data(input_var1, label_u.clone(),
                                                                            logits_u_aug.clone(), mode="cutmix")

        ignore_mask = ((logits_u_aug < 0.65).long()) * 255




        # compute student_model
        model.train()
        # label_u[label_u == 0] = 255
        unsup_loss_pt=0.0
        consistency_loss=0.0
        # loss_contrastive=0.0
        if epoch < 200:
            percent = 100
        else:
            percent = 80

        label_st = label_u_aug.clone()
        label_st[label_st == 0] = 255
        with autocast():
            if epoch >= cfg['trainer']['nf_start_epoch'] + 1:
                model.apply(to_mix_status)

                out_labeled,out_unlabeled, out_all_unlabeled_pt, res_head_u1 = model(input_var, input_var1_aug,label_st, nf_model, loss_flow, cfg, eps=cfg['adv']['eps'],
                                         adv=True)


                valid_mask = (ignore_mask != 255).float()  # [B, H, W]



                log_prob_maps = F.log_softmax(out_labeled, dim=1)
                loss_map = criterion(log_prob_maps, target_var)
                loss_map = loss_map * weight_map_var
                loss_CE = loss_map.mean()


                unsup_loss = compute_unsupervised_loss_conf_weight(
                    label_u_aug.clone(),percent,out_unlabeled)


                prob_u = F.softmax(output_ema_u, dim=1)  # [B, 3, 416, 416]
                confidence_u, label_u = torch.max(prob_u, dim=1)  # [B, 416, 416]
                H_feat, W_feat = res_head_u1.shape[2:]

                label_u_down = F.interpolate(label_u.unsqueeze(1).float(), size=(H_feat, W_feat),
                                             mode='nearest').squeeze(1).long()
                confidence_down = F.interpolate(confidence_u.unsqueeze(1), size=(H_feat, W_feat),
                                                mode='nearest').squeeze(1)

                prob_u = F.softmax(out_unlabeled, dim=1)
                anchor_feat, pos_feat, neg_feat = extract_student_features_by_pseudolabel(
                    label_u_down, confidence_down, res_head_u1,
                    anchor_class=2, positive_class=1, negative_class=0,
                    threshold=0.1, max_points=50
                )


                unsup_loss_pt = compute_unsupervised_loss_conf_weight(
                    label_u_aug.clone(),percent,out_all_unlabeled_pt)

                if anchor_feat is not None:
                    loss_contrastive = criterion_loss(anchor_feat, pos_feat, neg_feat)


                lambda_u = ramp_up_weight(epoch, max_epoch=1000, max_weight=1.0)  # 无标签损失最大权重为1.0
                lambda_pt = 0.5 * lambda_u
                lambda_cons = 0.3 * lambda_u
                lambda_ctr = 0.25 * lambda_u

                loss = loss_CE + lambda_u * unsup_loss + lambda_pt * unsup_loss_pt  + lambda_cons * loss_contrastive

            else:
                output_labeled, output_unlabeled, _= model(input_var, input_var1_aug,label=None, nf_model=None, loss_flow=None, cfg=None, eps=0, adv=False)

                unsup_loss = compute_unsupervised_loss_conf_weight(label_u_aug,percent,output_unlabeled)


                log_prob_maps = F.log_softmax(output_labeled,dim=1)
                loss_map = criterion(log_prob_maps, target_var)
                loss_map *= weight_map_var
                loss_CE = loss_map.mean()

                loss= loss_CE +  unsup_loss


        if opt.train['alpha'] != 0:
            if epoch >= cfg['trainer']['nf_start_epoch'] + 1:
                prob_maps = F.softmax(out_labeled, dim=1)
            else:
                prob_maps = F.softmax(output_labeled,dim=1)
    
                # label instances in target
            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
                loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss_total = loss + opt.train['alpha'] * loss_var


        else:
            loss_var = torch.ones(1) * -1
            loss_total = loss



        ###############################Training of Normalizing FLow ################################
        nf_loss=0.0
        if epoch >= cfg['trainer']['nf_start_epoch']:

            nf_model.train()
            with autocast():
                label_l_small = F.interpolate(target_var.unsqueeze(1).float(), size=output_rep.shape[2:],
                                              mode="nearest").squeeze(1)

                label_u_small = F.interpolate(label_u_aug.unsqueeze(1).float(), size=output_rep.shape[2:],
                                              mode="nearest").squeeze(1)
                ignore_mask_small = F.interpolate(ignore_mask.unsqueeze(1).float(), size=output_rep.shape[2:],
                                                  mode="nearest").squeeze(1).long()

                label_u_small[ignore_mask_small == 255] = 255

                b, c, h, w = output_rep.size()  # pred_all, rep_all, fts_all
                total_n = int(b * h * w)

                fts_all = output_rep.detach().permute(0, 2, 3, 1).reshape(total_n, c)
                label_nf = torch.cat([label_l_small.long(), label_u_small.long()], dim=0)
                label_nf = label_nf.detach().clone().view(-1)  # n

                valid_map = (label_nf != 255)  # filter out ignored pixels
                valid_fts_num = int(valid_map.sum())
                valid_fts = fts_all[valid_map]
                valid_label = label_nf[valid_map]


                sample_num = min(20 * 1024, valid_fts.size(0))
                sample_idx = torch.randperm(valid_fts.size(0))[:sample_num]
                input_nf_sample = valid_fts[sample_idx]  # [sample_num, c]
                label_nf_sample = valid_label[sample_idx]

                # add noise
                input_nf_sample += cfg['flow']['noise'] * torch.randn_like(input_nf_sample)

                z, log_jac_det = nf_model(input_nf_sample)

                nf_loss, ll, prior_ll, sldj = loss_flow(z, sldj=log_jac_det, y=label_nf_sample)

            optimizer_nf.zero_grad()
            scaler_nf.scale(nf_loss).backward()
            scaler_nf.unscale_(optimizer_nf)
            # Clip the gradient
            clip_grad_norm(optimizer_nf, cfg['flow']['grad_clip'])
            scaler_nf.step(optimizer_nf)
            scaler_nf.update()
            nf_model.eval()

        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, target.numpy())
        pixel_accu, iou = metrics[0], metrics[1]

        result = [loss, loss_CE, loss_var, unsup_loss, unsup_loss_pt, nf_loss, pixel_accu, iou]
        results.update(result, input.size(0))


        # compute gradient and do SGD step

        optimizer.zero_grad()
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        model, teacher_model = update_ema_variables(model, teacher_model, epoch=epoch,base_alpha=0.99)

        del input_var, target_var, log_prob_maps, loss_total

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration:[{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_CE {r[1]:.4f}'
                        '\tLoss_var {r[2]:.4f}'
                        '\tunsup_loss {r[3]:.4f}'
                        '\tunsup_loss_pt {r[4]:.4f}'
                        '\tnf_loss {r[5]:.4f}'
                        '\tpixel_accu {r[6]:.4f}'
                        '\tIoU {r[7]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t => Train_Avg:Loss_total {r[0]:.4f}'
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_var {r[2]:.4f}'
                '\tunsup_loss {r[3]:.4f}'
                '\tunsup_loss_pt {r[4]:.4f}'
                '\tnf_loss {r[5]:.4f}'
                '\tpixel_accu {r[6]:.4f}'
                '\tIoU {r[7]:.4f}'.format(epoch, opt.train['num_epochs'], r=results.avg))

    return results.avg





def extract_student_features_by_pseudolabel(
        pseudo_label, confidence, student_feat,
        anchor_class=2, positive_class=1, negative_class=0,
        threshold=0.8, max_points=100
):

    B, C, H, W = student_feat.shape
    student_feat = student_feat.permute(0, 2, 3, 1)  # [B, H, W, C]

    anchor_feat_list, pos_feat_list, neg_feat_list = [], [], []

    for b in range(B):
        label_b = pseudo_label[b]
        conf_b = confidence[b]
        feat_b = student_feat[b]

        mask_anchor = (label_b == anchor_class) & (conf_b > threshold)
        mask_pos = (label_b == positive_class) & (conf_b > threshold)
        mask_neg = (label_b == negative_class) & (conf_b > threshold)

        def get_feat(mask):
            if mask.sum() == 0:
                return None
            feat = feat_b[mask]  # [N, C]
            if feat.shape[0] > max_points:
                idx = torch.randperm(feat.shape[0])[:max_points]
                feat = feat[idx]
            return feat

        anchor_feat = get_feat(mask_anchor)
        pos_feat = get_feat(mask_pos)
        neg_feat = get_feat(mask_neg)

        if anchor_feat is not None:
            anchor_feat_list.append(anchor_feat)
        if pos_feat is not None:
            pos_feat_list.append(pos_feat)
        if neg_feat is not None:
            neg_feat_list.append(neg_feat)

    if anchor_feat_list and pos_feat_list and neg_feat_list:
        return (
            torch.cat(anchor_feat_list, dim=0),
            torch.cat(pos_feat_list, dim=0),
            torch.cat(neg_feat_list, dim=0),
        )
    else:
        return None, None, None


def compute_consistency_loss(out_unlabeled, out_all_unlabeled_pt, mask=None):
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mse = F.mse_loss(out_unlabeled * mask, out_all_unlabeled_pt * mask, reduction='sum')
        norm = mask.sum() * out_unlabeled.shape[1]
        loss = mse / (norm + 1e-6)
    else:
        loss = F.mse_loss(out_unlabeled, out_all_unlabeled_pt)
    return loss



def ramp_up_weight(epoch, max_epoch=1000, max_weight=1.0):
    if epoch < 100:
        return max_weight * (epoch / 100)  # 前100轮线性增长
    else:
        return max_weight

def compute_unsupervised_loss_conf_weight(target, percent, pred_teacher):
    batch_size, num_class, h, w = pred_teacher.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        conf, ps_label = torch.max(prob, dim=1)
        conf = conf.detach()
        conf_thresh = np.percentile(
            conf[target !=0 ].cpu().numpy().flatten(), 100 - percent
        )

        thresh_mask = conf.le(conf_thresh).bool() * (target !=0).bool()

        conf[thresh_mask] = 0
        target[thresh_mask] = 0

        weight = batch_size * h * w / (torch.sum(target !=0 ) + 1e-6)

    loss_ = weight * F.cross_entropy(pred_teacher, target, ignore_index=0, reduction='none')  # [10, 321, 321]
    conf = (conf + 1.0) / (conf + 1.0).sum() * (torch.sum(target !=0 ) + 1e-6)
    loss = torch.mean(conf * loss_)
    return loss

def validate(val_loader, model, criterion):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(3)

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):
        input, weight_map, target = sample
        weight_map = weight_map.float().div(20)
        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        weight_map_var = weight_map.cuda()

        # for b in range(input.size(0)):
        #     utils.show_figures((input[b, 0, :, :].numpy(), target[b,0,:,:].numpy(), weight_map[b, :, :]))

        if torch.max(target) == 255:
            target = target / 255
        if target.dim() == 4:
            target2 = target.squeeze(1)

        target_var = target2.cuda()

        size = opt.train['input_size'][0]
        overlap = opt.train['val_overlap']
        output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])

        output1 = F.softmax(output, dim=1)
        # print(target1.shape)


        log_prob_maps = F.log_softmax(output, dim=1)

        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE = loss_map.mean()

        if opt.train['alpha'] != 0:
            prob_maps = F.softmax(output, dim=1)

            target_labeled = torch.zeros(target2.size()).long()
            for k in range(target2.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target2[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = loss_CE + opt.train['alpha'] * loss_var
        else:
            loss = loss_CE

        # measure accuracy and record loss
        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, target2.numpy())
        pixel_accu = metrics[0]
        iou = metrics[1]

       
        results.update([loss.item(),pixel_accu, iou])

        del output, target_var, log_prob_maps, loss

    logger.info('\t=> Val Avg:   Loss {r[0]:.4f}\tPixel_Acc {r[1]:.4f}'
                '\tIoU {r[2]:.4f}'.format(r=results.avg))



    return results.avg


def save_checkpoint(state, epoch, is_best, save_dir, cp_flag, is_second):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch + 1))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))
    if is_second:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}_demo.pth.tar'.format(cp_dir, epoch + 1))

def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_CE\ttrain_loss_var\ttrain_acc\ttrain_iou\t'
                            'val_loss\tval_acc\tval_iou')

    return logger, logger_results


def update_ema_variables(model, model_teacher, epoch, base_alpha=0.99):
    # 动态调整 alpha
    alpha = min(1.0 - 1.0 / float(epoch + 1), base_alpha)
    for param_t, param in zip(model_teacher.parameters(), model.parameters()):
        param_t.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    return model, model_teacher




def criterion_loss(out_1, out_2, tau_plus, batch_size, beta, estimator):
    # neg score

    out_1 = out_1.view(8,256)
    out_2 = out_2.view(8,256)
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / 0.5)
    old_neg = neg.clone()
    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / 0.5)
    pos = torch.cat([pos, pos], dim=0)

    # negative samples similarity scoring
    if estimator == 'hard':
        N = batch_size * 2 - 2
        imp = (beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / 0.5))
    elif estimator == 'easy':
        Ng = neg.sum(dim=-1)
    else:
        raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
    loss = (- torch.log((pos+1e-12) / (pos + Ng+1e-12))).mean()

    return loss



def compute_unsupervised_loss(predict, target, ignore_mask):

    target[ignore_mask==255] = 255
    loss = F.cross_entropy(predict, target, ignore_index=255)

    return loss

if __name__ == '__main__':
    main()