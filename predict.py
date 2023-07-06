import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch import nn
import torch
import warnings
from torch.autograd import Variable
import numpy as np
from models.DeepLabV3_plus.deeplabv3_plus import *
import cv2 as cv
from data_load import LoadTest
from glob import glob
from models.semantic.unet import *
from Mynet import *
from SCNN_uncertain import *
from BDRAR.model import *
from PIL import Image
from unet import *
from MSASDNet import *
from torch.nn import functional as F
from mecnet import *
# from baseline.Unet import *
# from models.AttentionUnet.AttUnet import *
# # from models.DeepLabV3_plus.deeplabv3_plus import *
from baseline.FCN import *
from models.danet import *
from models.DeepLabV3.deeplabv3 import *
# # from baseline.res_unet import *
# # from baseline.HRNet import *
# from models.danet import *
# from Net_1_j3gaie4_sota import *
# from models.PSPNet.pspnet import *
# from models.denseASPP import *
from baseline.SegNet import *
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def predictWithOverlapB(model, img, patch_size=256, overlap_rate=0):
    '''

    :param model: a trained model
    :param img: a path for an image
    :param patch_size:
    :param overlap_rate:
    :return:
    '''
    # subsidiary value for the prediction of an image with overlap
    boder_value = int(patch_size * overlap_rate / 2)
    double_bv = boder_value * 2
    stride_value = patch_size - double_bv
    most_value = stride_value + boder_value

    # an image for prediction
    # img = cv.imread(img_path, cv.IMREAD_COLOR)
    m, n, _ = img.shape
    load_data = LoadTest()
    if max(m, n) <= patch_size:
        tmp_img = img
        tmp_img = load_data(tmp_img)
        with torch.no_grad():
            tmp_img = Variable(tmp_img)
            tmp_img = tmp_img.cuda().unsqueeze(0)
            result = model(tmp_img)
        output = result if not isinstance(result, (list, tuple)) else result[0]
        output = F.sigmoid(output)
        # 做个标准化
        output = output[:, 0, :, :]
        output = normPRED(output)
        pred = output.data.cpu().numpy().squeeze(0)  # [0]
        # pred = output.data.cpu().numpy().squeeze(0).squeeze(0)  # [0]
        img_rgb = Image.fromarray(pred * 255).convert('RGB')
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        return pred.astype(np.uint8), img_rgb
        # return pred
    else:
        tmp = (m - double_bv) // stride_value  # 剔除重叠部分相当于无缝裁剪
        new_m = tmp if (m - double_bv) % stride_value == 0 else tmp + 1
        tmp = (n - double_bv) // stride_value
        new_n = tmp if (n - double_bv) % stride_value == 0 else tmp + 1
        FullPredict = np.zeros((m, n), dtype=np.uint8)
        for i in range(new_m):
            for j in range(new_n):
                if i == new_m - 1 and j != new_n - 1:
                    tmp_img = img[
                              -patch_size:,
                              j * stride_value:((j + 1) * stride_value + double_bv), :]
                elif i != new_m - 1 and j == new_n - 1:
                    tmp_img = img[
                              i * stride_value:((i + 1) * stride_value + double_bv),
                              -patch_size:, :]
                elif i == new_m - 1 and j == new_n - 1:
                    tmp_img = img[
                              -patch_size:,
                              -patch_size:, :]
                else:
                    tmp_img = img[
                              i * stride_value:((i + 1) * stride_value + double_bv),
                              j * stride_value:((j + 1) * stride_value + double_bv), :]
                tmp_img = load_data(tmp_img)
                with torch.no_grad():
                    tmp_img = Variable(tmp_img)
                    tmp_img = tmp_img.cuda().unsqueeze(0)
                    result = model(tmp_img)
                output = result if not isinstance(result, (list, tuple)) else result[0]
                output = F.sigmoid(output)
                pred = output.data.cpu().numpy().squeeze(0).squeeze(0)  # [0]

                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0

                if i == 0 and j == 0:  # 左上角
                    FullPredict[0:most_value, 0:most_value] = pred[0:most_value, 0:most_value]
                elif i == 0 and j == new_n - 1:  # 右上角
                    FullPredict[0:most_value, -most_value:] = pred[0:most_value, boder_value:]
                elif i == 0 and j != 0 and j != new_n - 1:  # 第一行
                    FullPredict[0:most_value, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[0:most_value, boder_value:most_value]

                elif i == new_m - 1 and j == 0:  # 左下角
                    FullPredict[-most_value:, 0:most_value] = pred[boder_value:, :-boder_value]
                elif i == new_m - 1 and j == new_n - 1:  # 右下角
                    FullPredict[-most_value:, -most_value:] = pred[boder_value:, boder_value:]
                elif i == new_m - 1 and j != 0 and j != new_n - 1:  # 最后一行
                    FullPredict[-most_value:, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[boder_value:, boder_value:-boder_value]

                elif j == 0 and i != 0 and i != new_m - 1:  # 第一列
                    FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, 0:most_value] = \
                        pred[boder_value:-boder_value, 0:-boder_value]
                elif j == new_n - 1 and i != 0 and i != new_m - 1:  # 最后一列
                    FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, -most_value:] = \
                        pred[boder_value:-boder_value, boder_value:]
                else:  # 中间情况
                    FullPredict[
                    boder_value + i * stride_value:boder_value + (i + 1) * stride_value,
                    boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[boder_value:-boder_value, boder_value:-boder_value]
        return FullPredict


class Test(object):
    def __init__(self, save_path, save_path_rgb, imgpath, weight_path, ):
        self.imgpath = imgpath
        self.weight_path = weight_path
        self.save_path = save_path
        self.save_path_rgb = save_path_rgb
        os.makedirs(self.save_path, exist_ok=True)

    def predict(self, rgb=False):
        '''
        :return:
        '''
        img_pathes = glob(self.imgpath + '/*.png')
        # model = SCNN()
        # model =DeepLabv3_plus(in_channels=3, num_classes=1, backend='resnet101', os=16)
        # model = MECNet()
        # model = FCN32s(VGGNet(requires_grad=True),1)
        # model = UNet(in_channels=3, num_classes=1, filter_scale=2)
        # model = MSASDNet()
        # model = MSASDNet()
        # model = DeepLabV3(1)
        # model = PSPNet(in_channels=3, num_classes=1, backend='resnet101', pool_scales=(1, 2, 3, 6))
        # model = U_Net()
        model = BDRAR()
        # model = HighResolutionNet(1)
        # model = SegNet(1)
        # model = DANet()
        # model  = ResUnet(3)
        # model = denseASPP121(1)
        # model = FMNet()
        model.load_state_dict(torch.load(self.weight_path))
        model.cuda()
        model.eval()
        for i, path in enumerate(img_pathes):
            basename = os.path.basename(path)
            print('正在预测:%s, 已完成:(%d/%d)' % (basename, i + 1, len(img_pathes)))
            img = cv.imread(path, cv.IMREAD_COLOR)
            pred, img_rgb = predictWithOverlapB(model, img, patch_size=512)
            cv.imwrite(os.path.join(self.save_path, basename), pred)
            # 保存rgb预测图
            if rgb:
                img_rgb.save(os.path.join(self.save_path_rgb, basename))
        print('预测完毕!')


if __name__ == '__main__':
    root = r"F:\ww\dataset_gao\res\bdrar"
    save_path =  r'F:\ww\dataset_gao\save_pre/'
    save_path_rgb = r'F:\ww\dataset_gao\save_pre/'
    model_name = 'new_oursnodf2m'
    model_name_rgb = 'SCNN_rgb'
    save_path = os.path.join(save_path, model_name)
    save_path_rgb = os.path.join(save_path_rgb, model_name_rgb)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_rgb, exist_ok=True)
    img_path = r"F:\ww\dataset_gao\new\predict_image_2m"

    # weight_path = root + '/Epoch_20_Loss_0.064_Acc_0.989_Iou_0.905(nobri).pkl'
    # weight_path = root + '/Epoch_39_Loss_0.044_Acc_0.990_Iou_0.909(noglobal).pkl'
    weight_path = root + '/Epoch_10_Loss_3.892_Acc_0.967_Iou_0.862.pkl'
    # weight_path = root + '/Epoch_26_Loss_0.287_Acc_0.966_Iou_0.856.pkl'
    # weight_path = root + '/Epoch_27_Loss_0.059_Acc_0.990_Iou_0.911.pkl'
    # weight_path = root + '/Epoch_12_Loss_0.032_Acc_0.988_Iou_0.898(noall).pkl'

    predict_fuc = Test(save_path, save_path_rgb, img_path, weight_path)
    predict_fuc.predict(rgb=False)
