import os

from torch.autograd import Variable
import time
import glob
from data_load import GeneratorData
# from net3_gaimutil import *
import warnings
import datetime
import logging
from SCNN_xiu import *
# from Net_1_j3gaie4 import *
# from MANet import *
from collections import OrderedDict

# from net_F import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

saveLog = 'logs'
os.makedirs(saveLog, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fileName = datetime.datetime.now().strftime('day' + '%Y-%m-%d-%H')
handler = logging.FileHandler(os.path.join(saveLog, fileName + '.log'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function

def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean()

"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


# bce_loss = nn.BCELoss(size_average=True)
# ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
# iou_loss = pytorch_iou.IOU(size_average=True)
#
# def bce_ssim_loss(pred,target):
#     bce_out = bce_loss(pred,target)
#     ssim_out = 1 - ssim_loss(pred,target)
#     iou_out = iou_loss(pred,target)
#
#     loss = bce_out + ssim_out + iou_out
#
#     return loss
#
# def muti_bce_loss_fusion(d0, d1,  labels_v):
#     loss0 = bce_ssim_loss(d0,labels_v)
#     loss1 = bce_ssim_loss(d1,labels_v)
#     # loss2 = bce_ssim_loss(d2,labels_v)
#     # loss3 = bce_ssim_loss(d3,labels_v)
#     # loss4 = bce_ssim_loss(d4,labels_v)
#
#     loss = loss0 + loss1 # + 5.0*lossa
#     # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f\n" % (
#     #     loss0.data, loss1.data, loss2.data, loss3.data, loss4.data))
#     return loss

class Train(object):
    def __init__(self, data_path, save_path, patch_size=256, learing_rate=1e-4, epoch=100, batchSize=4, ):
        self.data_path = data_path
        self.save_path = save_path
        self.patch_size = patch_size
        self.epoch = epoch
        self.batchSize = batchSize
        self.learing_rate = learing_rate

    def _train2(self, epoch, model, learing_rate, batch_size=4):

        multi_scale = (1.0,)
        # multi_scale = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        train_data_gen = GeneratorData(self.data_path + '/train_data/',
                                       batch_size=batch_size,
                                       multi_scale=multi_scale).generate(val=True)
        val_data_gen = GeneratorData(self.data_path + '/val_data/', batch_size=batch_size).generate(val=True)

        single_image_clip_num = 1  # 一张图裁剪为多少个瓦片
        single_image_clip_num_val = 1
        train_data_nums = len(glob.glob(self.data_path + '/train_data/img/*.tif'))
        iters_per_epoch = train_data_nums // batch_size
        max_iters = epoch * iters_per_epoch
        epoch_size_train = train_data_nums * len(multi_scale) * single_image_clip_num // batch_size
        epoch_size_val = len(
            glob.glob(self.data_path + '/val_data/img/*.tif')) * single_image_clip_num_val // batch_size
        num_flag = epoch_size_train // 100 if epoch_size_train // 100 == 0 else epoch_size_train // 100 + 1

        # loss_func = torch.nn.BCEWithLogitsLoss()
        # loss_func.cuda()
        criteria_loss = BceDiceLoss()
        model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learing_rate)  # 阴影 0.001
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        # optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9, weight_decay=1e-5)

        # ### 更新 学习率
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,0.98)
        ### 预热学习率
        # lr_scheduler = WarmupPolyLR(optimizer,
        #                                   max_iters=max_iters,
        #                                   power=0.9,
        #                                   warmup_factor=1.0 / 3,
        #                                   warmup_iters=0,
        #                                   warmup_method="linear")
        st_val_acc = 0.
        st_panduan = 0.
        # st_val_loss, st_train_loss = 10.0, 10.0

        start_time = time.time()

        loss_draw = []
        acc_draw = []
        t_loss = []
        t_acc = []
        val_iou = []
        train_loss1 = []
        train_loss2 = []
        train_loss3 = []
        train_loss4 = []

        for ep in range(1, epoch + 1):
            logger.info('doing epoch：{}'.format(ep))

            perLoss, perAcc = 0., 0.
            ####______训练_______########

            model.train()
            TP = FP = TN = FN = 0
            for idx in range(epoch_size_train):
                img, label = next(train_data_gen)

                img = Variable(img)
                img = img.cuda()
                output = model(img)
                label = Variable(label)
                label = label.cuda()

                #
                # loss = loss_func(output[0].squeeze(1), label)
                # # loss = loss_func(output.squeeze(1), label)
                # loss1 = loss_func(output[1].squeeze(1), label)

                loss = criteria_loss(torch.sigmoid(output[0]), label.unsqueeze(1))
                loss1 = criteria_loss(torch.sigmoid(output[1]), label.unsqueeze(1))

                # loss2 = loss_func(output[2].squeeze(1), label)
                # loss3 = loss_func(output[3].squeeze(1), label)
                # loss4 = loss_func(output[4].squeeze(1), label)
                loss = loss + loss1
                # loss = 0.6 * loss + (loss1 + loss2 + loss3 + loss4) * 0.4

                perLoss += loss.data.cpu().numpy()
                # output = F.sigmoid(output)

                # print("loss:{}".format(loss))
                # print("loss:{}".format(loss1))
                # output = F.sigmoid(output[0])
                output = F.sigmoid(output[0])
                predict = output.squeeze(1)
                # logger.info(predict.shape, label.shape)
                predict[predict >= 0.5] = 1
                predict[predict < 0.5] = 0
                acc = (predict == label).sum().data.cpu().numpy() / (self.patch_size * self.patch_size * len(label))
                perAcc += acc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                if idx % num_flag == 0:
                    print('Train epoch: {} [{}/{} ({:.2f}%)] || Lr: {:.6f} || Loss:{:.6f} || Acc:{:.3f}'.format(
                        ep, idx + 1, epoch_size_train, 100.0 * (idx + 1) / epoch_size_train,
                        optimizer.param_groups[0]['lr'],
                        loss.data.cpu().numpy(), acc))
                    logger.info('Train epoch: {} [{}/{} ({:.2f}%)]\tLoss:{:.6f}\tAcc:{:.3f}'.format(
                        ep, idx + 1, epoch_size_train, 100.0 * (idx + 1) / epoch_size_train,
                        loss.data.cpu().numpy(), acc))
            t_los_mean = perLoss / epoch_size_train
            t_acc_mean = perAcc / epoch_size_train

            print('Train Epoch: {}, m_Loss: {:.4f}, m_Acc: {:.4f}'.format(ep, t_los_mean, t_acc_mean))
            logger.info('Train Epoch: {}, m_Loss: {:.4f}, m_Acc: {:.4f}'.format(ep, t_los_mean, t_acc_mean))

            ####______验证_______########

            model.eval()
            perValLoss = 0.
            perValAcc = 0.
            print('正在进行验证模型，请稍等...')
            logger.info('正在进行验证模型，请稍等...')
            for idx in range(epoch_size_val):
                img, label = next(val_data_gen)
                with torch.no_grad():
                    img = Variable(img)
                    img = img.cuda()
                    output = model(img)

                    label = Variable(label)
                    label = label.cuda()
                    # metric = SegmentationMetric(2)
                    # metric.update(output[0], label)
                    # pixAcc, mIoU = metric.get()
                    # loss = loss_func(output[0].squeeze(1), label)
                    # # loss = loss_func(output.squeeze(1), label)
                    # loss1 = loss_func(output[1].squeeze(1), label)

                    loss = criteria_loss(torch.sigmoid(output[0]), label.unsqueeze(1))
                    loss1 = criteria_loss(torch.sigmoid(output[1]), label.unsqueeze(1))
                    # loss = bcelos_dice_loss(label, F.sigmoid(output[0].squeeze(1)))
                    # loss1 = bcelos_dice_loss(label, F.sigmoid(output[1].squeeze(1)))
                    # loss2 = loss_func(output[2].squeeze(1), label)
                    # loss3 = loss_func(output[3].squeeze(1), label)
                    # loss4 = loss_func(output[4].squeeze(1), label)
                    # loss = 0.6 * loss + (loss1 + loss2 + loss3 + loss4) * 0.4
                    loss = loss + loss1
                    # loss = loss + loss1/(loss1/loss).detach()+loss2/(loss2/loss).detach()+loss3/(loss3/loss).detach()+loss4/(loss4/loss).detach()

                perValLoss += loss.data.cpu().numpy()

                output = F.sigmoid(output[0])
                # output = F.sigmoid(output)
                predict = output.squeeze(1)
                predict[predict >= 0.5] = 1
                predict[predict < 0.5] = 0

                TP += ((predict == 1) & (label == 1)).cpu().sum().item()
                TN += ((predict == 0) & (label == 0)).cpu().sum().item()
                FN += ((predict == 0) & (label == 1)).cpu().sum().item()
                FP += ((predict == 1) & (label == 0)).cpu().sum().item()

                valacc = (predict == label).sum().data.cpu().numpy() / (self.patch_size * self.patch_size * len(label))
                perValAcc += valacc
            val_los_mean = perValLoss / epoch_size_val
            val_acc_mean = perValAcc / epoch_size_val
            iou = TP / (TP + FP + FN)
            print('Val Loss: {:.6f}, Acc: {:.6f}, IOU: {:.6f}'.format(val_los_mean, val_acc_mean, iou))
            logger.info('Val Loss: {:.6f}, Acc: {:.6f}, IOU: {:.6f}'.format(val_los_mean, val_acc_mean, iou))
            new_panduan = iou
            if new_panduan > st_panduan:
                st_panduan = new_panduan
                # if st_val_acc < val_acc_mean:
                #     st_val_acc = val_acc_mean
                ### 不同的判断
                # if st_train_loss > t_los_mean and st_val_acc < val_acc_mean + 0.02 or st_val_acc < val_acc_mean:
                #     if st_train_loss > t_los_mean: st_train_loss = t_los_mean
                #     if st_val_acc < val_acc_mean: st_val_acc = val_acc_mean

                # 仅保存和加载模型参数
                print('进行权重保存-->>\nEpoch：{}\t\nTrainLoss:{:.4f}\t\nValAcc:{:.4f}\t\nIoU:{:.3f}'
                      ''.format(ep, float(t_los_mean), float(val_acc_mean), float(iou)))
                logger.info('进行权重保存-->>\nEpoch：{}\t\nTrainLoss:{:.4f}\t\nValAcc:{:.4f}\t\nIoU:{:.3f}'
                            ''.format(ep, float(t_los_mean), float(val_acc_mean), float(iou)))
                save_model = self.save_path + 'Epoch_{}_Loss_{:.3f}_Acc_{:.3f}_Iou_{:.3f}.pkl'.format(ep,
                                                                                                      float(t_los_mean),
                                                                                                      float(
                                                                                                          val_acc_mean),
                                                                                                      float(iou))
                torch.save(model.state_dict(), save_model)

            #     no_optim = 0
            # else:
            #     no_optim = no_optim + 1

            # if no_optim > 3:
            #     model.load_state_dict(torch.load(save_model))
            #     scheduler.step()
            #     print('Scheduler step!')
            #     print('Learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
            #     no_optim = 0
            # # if optimizer.state_dict()['param_groups'][0]['lr']<1e-7:
            # #     break
            # if no_optim > 5:
            #     print("应停止训练...")
            #     logger.info("应停止训练...")
            #     break
            duration1 = time.time() - start_time
            # start_time = time.time()
            print('train running time: %.2f(minutes)' % (duration1 / 60))
            logger.info('train running time: %.2f(minutes)' % (duration1 / 60))

            loss_draw.append(val_los_mean)
            acc_draw.append(val_acc_mean)
            t_loss.append(t_los_mean)
            t_acc.append(t_acc_mean)
            val_iou.append(iou)
            # train_loss1.append(loss1.data.cpu().numpy())
            # train_loss2.append(loss2.data.cpu().numpy())
            # train_loss3.append(loss3.data.cpu().numpy())
            # train_loss4.append(loss4.data.cpu().numpy())
            # with open("val_iou.txt", 'w') as f:
            #     for i in range(len(val_iou)):
            #         v = str(val_iou[i])
            #         f.write(v + '\n')
            # with open("train_loss1.txt", 'w') as f:
            #     for i in range(len(train_loss1)):
            #         v = str(train_loss1[i])
            #         f.write(v + '\n')
            # with open("train_loss2.txt", 'w') as f:
            #     for i in range(len(train_loss2)):
            #         v = str(train_loss2[i])
            #         f.write(v + '\n')
            # with open("train_loss3.txt", 'w') as f:
            #     for i in range(len(train_loss3)):
            #         v = str(train_loss3[i])
            #         f.write(v + '\n')
            # with open("train_loss4.txt", 'w') as f:
            #     for i in range(len(train_loss4)):
            #         v = str(train_loss4[i])
            #         f.write(v + '\n')
            with open("val_loss.txt", 'w') as f:
                for i in range(len(loss_draw)):
                    v = str(loss_draw[i])
                    f.write(v + '\n')
            # with open("val_acc.txt",'w') as g:
            #     for o in range(len(acc_draw)):
            #         b = str(acc_draw[o])
            #         g.write(b + '\n')
            #
            with open("train_loss.txt", 'w') as h:
                for p in range(len(t_loss)):
                    c = str(t_loss[p])
                    h.write(c + '\n')
            # with open("train_acc.txt",'w') as j:
            #     for q in range(len(t_acc)):
            #         a = str(t_acc[q])
            #         j.write(a + '\n')

        # draw_fig(loss_draw, "loss", epoch)
        # draw_fig(acc_draw, "acc", epoch)

    def train(self):
        model = SCNN()
        self._train2(self.epoch, model=model, learing_rate=self.learing_rate, batch_size=self.batchSize)


if __name__ == '__main__':
    data_path = './dataset/'
    save_path_name = 'res/SCNN/'
    save_path = data_path + '/' + save_path_name

    os.makedirs(save_path, exist_ok=True)
    T = Train(data_path, save_path,
              patch_size=512, epoch=100, learing_rate=3e-4, batchSize=8, )
    T.train()
    os.system("shutdown")
