import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from my_net.UCRNetAB_change import UCRNet13
from my_net.mdunet_de import MDUNET_l1,MDUNET
from my_net.mdunet_de import MDUNET_l1,MDUNET
from UARNet.cenet import CE_Net_
from UARNet.network import AttU_Net
from src.unet import UNet
from src.unet_2plus import  UNetPlusPlus
from src.DCSAU_Net import DCSAUNet
from src.egeunet import  EGEUNet
from src.MSDANet import MSDANet
import imageio
from utils.dataloader import test_dataset
import imageio
import sklearn.metrics as metrics
import sys
import cv2

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    与 accuracy() 中 iou 口径保持一致（分子/分母都加 smooth）
    """
    smooth = kwargs.get('smooth', 1e-3)
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    与 accuracy() 中 dice 口径保持一致（分子/分母都加 smooth）
    """
    smooth = kwargs.get('smooth', 1e-3)
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice

def accuracy(pred_mask, label, smooth: float = 1e-3):
    '''
    与 mean_dice_np/mean_iou_np 统一口径的向量化二值指标
    返回: iou, dice, acc, sen, sp, pre, recall, f1
    '''
    pred = pred_mask.astype(np.uint8)
    gt   = label.astype(np.uint8)

    inter = np.sum(pred * gt)
    sum_  = np.sum(pred) + np.sum(gt)
    union = sum_ - inter

    dice = 2.0 * (inter + smooth) / (sum_ + smooth)
    iou  = (inter + smooth) / (union + smooth)

    TP = inter
    FP = np.sum(pred) - inter
    FN = np.sum(gt)   - inter
    TN = pred.size - TP - FP - FN

    acc    = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)
    sen    = (TP + smooth) / (TP + FN + smooth)
    sp     = (TN + smooth) / (TN + FP + smooth)
    pre    = (TP + smooth) / (TP + FP + smooth)
    recall = sen
    f1     = 2 * pre * recall / (pre + recall + smooth)

    return iou, dice, acc, sen, sp, pre, recall, f1

def calculate_auc_test(prediction, label):
    """
    AUC 仅在 GT 同时包含 0/1 时才可定义；否则返回 None
    prediction: 概率图 (0~1)，不要传二值图
    """
    result_1D = prediction.flatten()
    label_1D = label.flatten().astype(np.uint8)

    # 单类样本无 AUC 定义
    if np.unique(label_1D).size < 2:
        return None

    auc = metrics.roc_auc_score(label_1D, result_1D)
    return auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='isic/isic2017/MDUNET.pth')  # 'snapshots/TransFuse-19_best.pth'
    parser.add_argument('--test_path', type=str,
                        default='data/isic/', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default='isic2018/UNetPlusPlus/', help='path to save inference segmentation')#'result/Trans_unet2/isic'
    parser.add_argument('--thresh', type=float, default=0.5, help='binarization threshold')
    parser.add_argument('--smooth', type=float, default=1e-3, help='smoothing for metrics')

    opt = parser.parse_args()

    model = MDUNET().cuda()
    model.load_state_dict(torch.load(opt.ckpt_path))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    print('evaluating model: ', opt.ckpt_path)

    image_root = '{}/data_test.npy'.format(opt.test_path)
    gt_root = '{}/mask_test.npy'.format(opt.test_path)
    test_loader = test_dataset(image_root, gt_root)

    dice_bank = []
    iou_bank = []
    acc_bank = []
    total_iou = []
    total_dice = []
    total_acc = []
    total_sen = []
    total_sp = []
    total_pre = []
    total_recall = []
    total_f1 = []
    total_auc = []

    for i in range(test_loader.size):
        image, gt = test_loader.load_data()  #
        gt = 1 * (gt > 0.5)
        image = image.cuda()

        with torch.no_grad():
            res = model(image)  # logits
        prob = res.sigmoid().data.cpu().numpy().squeeze()  # 概率

        # ---- AUC：只在有效时加入 ----
        auc = calculate_auc_test(prob, gt)
        if auc is not None:
            total_auc.append(auc)

        # ---- 二值化用于其余指标 ----
        bin_pred = (prob > opt.thresh).astype(np.uint8)

        # if opt.save_path is not None:
        #     cv2.imwrite(os.path.join(opt.save_path, f'{i}_pred.png'), (bin_pred * 255).astype(np.uint8))
        #     cv2.imwrite(os.path.join(opt.save_path, f'{i}_gt.png'),   (gt * 255).astype(np.uint8))

        # 与 accuracy() 一致口径
        dice = mean_dice_np(gt, bin_pred, smooth=opt.smooth)
        iou = mean_iou_np(gt, bin_pred, smooth=opt.smooth)
        acc = np.mean(bin_pred == gt)

        iou1, dice1, acc1, sen, sp, pre, recall, f1 = accuracy(bin_pred, gt, smooth=opt.smooth)

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)

        total_iou.append(iou1)
        total_dice.append(dice1)
        total_acc.append(acc1)
        total_sen.append(sen)
        total_sp.append(sp)
        total_pre.append(pre)
        total_recall.append(recall)
        total_f1.append(f1)

    print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
          format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(np.mean(total_dice), np.std(total_dice))
    print(np.mean(total_iou), np.std(total_iou))
    print(np.mean(total_acc), np.std(total_acc))
    print(np.mean(total_sen), np.std(total_sen))
    print(np.mean(total_sp), np.std(total_sp))
    # AUC 可能为空（全部单类样本），打印时做保护
    if len(total_auc) == 0:
        print("AUC: nan nan  (无有效样本：单图 GT 仅含单一类别)")
    else:
        print(np.mean(total_auc), np.std(total_auc))
    print(np.mean(total_pre), np.std(total_pre))
    print(np.mean(total_recall), np.std(total_recall))
    print(np.mean(total_f1), np.std(total_f1))
    print("############################")
