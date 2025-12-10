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
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice
def get_main_pred(out):
    """从模型输出中拿主预测张量：
    - 张量 -> 直接返回
    - (ds_list, pred) -> 返回 pred
    - (pred1, pred2, ...predK) -> 取最后一个
    """
    import torch
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        # 形如 (ds_list, pred)
        if len(out) == 2 and isinstance(out[0], (list, tuple)):
            if torch.is_tensor(out[1]):
                return out[1]
        # 形如 (pred1, pred2, ...)，取最后一个张量
        flat = list(out)
        while isinstance(flat[-1], (list, tuple)):
            flat = list(flat[-1])
        if torch.is_tensor(flat[-1]):
            return flat[-1]
    raise TypeError(f"Unsupported model output type: {type(out)}")

def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    返回: iou, dice, acc, sen, sp, pre, recall, f1
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_mask[i][j] == 1:
                    FP += 1
                else:
                    TN += 1

    total = TP + FN + TN + FP
    if total == 0:
        return None  # 图像为空，不计算

    eps = 1e-5  # 防止除零
    iou = TP / (TP + FN + FP + eps)
    dice = 2 * TP / (2 * TP + FN + FP + eps)
    acc = (TP + TN) / (total + eps)
    sen = TP / (TP + FN + eps)
    sp = TN / (TN + FP + eps)
    pre = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * pre * recall / (pre + recall + eps)

    return iou, dice, acc, sen, sp, pre, recall, f1
def calculate_auc_test(prediction, label):
    # read images
    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()


    label_1D = label_1D

    auc = metrics.roc_auc_score(label_1D, result_1D)

    # print("AUC={0:.4f}".format(auc))

    return auc
def tensor_to_uint8_img(t, mean=None, std=None):
    """
    t: 形状 [B,C,H,W] 或 [C,H,W] 的 torch.Tensor，值域可能是标准化过的/0~1/任意浮点
    mean, std: 若你的数据做过标准化，可以传入列表/元组（C 通道）。
               例如 ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
               如果不传，会用 min-max 自适应拉伸到 0~255。
    返回: uint8 的 numpy 图像 (H,W) 或 (H,W,3)
    """
    import torch, numpy as np
    if t.dim() == 4:
        t = t[0]  # 取第一个样本 [C,H,W]
    t = t.detach().cpu().float()

    if mean is not None and std is not None:
        # 反标准化
        mean = torch.tensor(mean, dtype=t.dtype)[:, None, None]
        std  = torch.tensor(std, dtype=t.dtype)[:, None, None]
        t = t * std + mean
        t = torch.clamp(t, 0, 1)
        t = t.numpy()
        if t.shape[0] == 1:
            img = (t[0] * 255.0).round().astype(np.uint8)           # (H,W)
        else:
            img = (np.transpose(t, (1,2,0)) * 255.0).round().astype(np.uint8)  # (H,W,C)
    else:
        # 无法确定均值方差，使用 min-max 拉伸到 0~255
        t_np = t.numpy()
        if t_np.shape[0] == 1:
            x = t_np[0]
            mn, mx = x.min(), x.max()
            if mx > mn:
                x = (x - mn) / (mx - mn)
            x = (x * 255.0).round().astype(np.uint8)
            img = x
        else:
            x = np.transpose(t_np, (1,2,0))  # (H,W,C)
            mn, mx = x.min(), x.max()
            if mx > mn:
                x = (x - mn) / (mx - mn)
            x = (x * 255.0).round().astype(np.uint8)
            img = x
    return img
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='isic/isic2018/MDUNET_l1.pth')  # 'snapshots/TransFuse-19_best.pth'
    parser.add_argument('--test_path', type=str,
                        default='data/isic/', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default='isic2018/MDUNET/', help='path to save inference segmentation')#'result/Trans_unet2/isic'

    opt = parser.parse_args()

    model = MDUNET_l1().cuda()
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
        # print(i)
        with torch.no_grad():
            out = model(image)
            res = get_main_pred(out)  # ← 关键：先取主输出张量

        # 如果你的模型 forward 返回 **logits**（推荐做法），再做 sigmoid：
        res = res.sigmoid().data.cpu().numpy().squeeze()
        total_auc.append(calculate_auc_test(res, gt))
        res = 1 * (res > 0.5)

        if opt.save_path is not None:
            os.makedirs(opt.save_path, exist_ok=True)

            # 1) 保存原图
            # 若你有标准化参数，填上：mean=[...], std=[...]
            img_uint8 = tensor_to_uint8_img(image, mean=None, std=None)  # 或者填你的 mean/std
            # cv2 用 BGR；若 img 是 RGB 你想保持看起来正常，可以转一下
            import cv2, numpy as np

            if img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_uint8  # 灰度就不转

            # cv2.imwrite(os.path.join(opt.save_path, f"{i}_img.png"), img_bgr)

            # 2) 保存预测与 GT（保持你原逻辑）
            # cv2.imwrite(os.path.join(opt.save_path, f"{i}_pred.png"), (res * 255).astype(np.uint8))
            # cv2.imwrite(os.path.join(opt.save_path, f"{i}_gt.png"), (gt * 255).astype(np.uint8))

            # 3) 可选：叠加可视化（红色为预测，绿色为GT），只在原图是三通道时演示
            try:
                if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
                    overlay = img_bgr.copy()
                    # 生成彩色掩码
                    pred_color = np.zeros_like(img_bgr);
                    pred_color[:, :, 2] = (res * 255).astype(np.uint8)  # BGR: 红色通道
                    gt_color = np.zeros_like(img_bgr);
                    gt_color[:, :, 1] = (gt * 255).astype(np.uint8)  # 绿色通道
                    alpha = 0.5
                    overlay = cv2.addWeighted(overlay, 1.0, pred_color, alpha, 0)
                    overlay = cv2.addWeighted(overlay, 1.0, gt_color, alpha, 0)
                    cv2.imwrite(os.path.join(opt.save_path, f"{i}_overlay.png"), overlay)
            except Exception as e:
                print(f"[WARN] overlay save failed at {i}: {e}")

        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])
        result = accuracy(res, gt)
        if result is None:
            print("tp 为 0，跳过本次计算")
            continue
        iou1, dice1, acc1, sen, sp, pre, recall, f1 = result
        # total_auc.append(calculate_auc_test(res, gt))

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
    # sys.stdout = open("driveunet20.txt", "a+")
    print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
          format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(np.mean(total_dice), np.std(total_dice))
    print(np.mean(total_iou), np.std(total_iou))
    print(np.mean(total_acc), np.std(total_acc))
    print(np.mean(total_sen), np.std(total_sen))
    print(np.mean(total_sp), np.std(total_sp))
    print(np.mean(total_auc), np.std(total_auc))
    print(np.mean(total_pre), np.std(total_pre))
    print(np.mean(total_recall), np.std(total_recall))
    print(np.mean(total_f1), np.std(total_f1))
    print("############################")
