import torch
import torchvision
from torch import nn
import torch.nn.functional as F
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        """ 通道注意力机制 同最大池化和平均池化两路分别提取信息，后共用一个多层感知机mlp,再将二者结合

        :param in_channel: 输入通道
        :param ratio: 通道降低倍率
        """
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通道先降维后恢复到原来的维数
        self.fc1 = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        # return x*self.sigmoid(out)

        # 平均池化一支 (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,8,8) -> (2,512,1,1)
        avg = self.avg_pool(x)
        # 多层感知机mlp (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,1,1) -> (2,512/ratio,1,1)
        avg = self.fc1(avg)
        avg = self.relu1(avg)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        avg_out = self.fc2(avg)

        # 最大池化一支
        # (2,512,8,8) -> (2,512,1,1)
        max = self.max_pool(x)
        # 多层感知机
        # (2,512,1,1) -> (2,512/ratio,1,1)
        max = self.fc1(max)
        max = self.relu1(max)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        max_out = self.fc2(max)

        # (2,512,1,1) + (2,512,1,1) -> (2,512,1,1)
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """ 空间注意力机制 将通道维度通过最大池化和平均池化进行压缩，然后合并，再经过卷积和激活函数，结果和输入特征图点乘

        :param kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x shape', x.shape)
        # (2,512,8,8) -> (2,1,8,8)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # (2,512,8,8) -> (2,1,8,8)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # (2,1,8,8) + (2,1,8,8) -> (2,2,8,8)
        cat = torch.cat([avg_out, max_out], dim=1)
        # (2,2,8,8) -> (2,1,8,8)
        out = self.conv1(cat)
        return x * self.sigmoid(out)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BR_ref(nn.Module):
    def __init__(self, in_channels):
        super(BR_ref, self).__init__()

        # self.fuse_conv = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
        #                                nn.BatchNorm2d(in_channels),
        #                                nn.ReLU(inplace=True),
        #                                )

        # # self.conv_ori = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        # #                               nn.BatchNorm2d(in_channels),
        # #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(in_channels, 2, kernel_size=1))
        self.conv_ori = nn.Sequential(nn.Conv2d(in_channels, 2, kernel_size=1))
        self.pred_conv = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1))
        self.br_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(True),
                                     # nn.Dropout2d()
                                     )

        self.loc_conv = nn.Sequential(nn.Conv2d(in_channels + 2, in_channels, kernel_size=3, padding=1),
                                      )
        self.BN_RELU = nn.Sequential(nn.BatchNorm2d(in_channels),
                                     nn.ReLU(True),
                                     # nn.Dropout2d()
                                     )
        self.avg = nn.AvgPool2d(3, stride=1, padding=1)
        # self.r_conv = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
        #                             nn.BatchNorm2d(in_channels),
        #                             # nn.ReLU(True)
        #                             )
        # self.res_Conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1),
        #                             nn.BatchNorm2d(in_channels),
        #                             nn.ReLU(True)
        #                             )
        # self.res_Conv = BasicBlock(in_channels, in_channels)
        self.down_Conv = nn.Sequential(nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU(True),
                                       # nn.Dropout2d()
                                       )
        # self.certan_conv = nn.Sequential(
        #     nn.MaxPool2d(3, stride=1, padding=1)
        # )
        self.loc_conv_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                        )
        self.ca = ChannelAttention(in_channels * 3)
        self.sa = SpatialAttention()
        self.enc_Conv = nn.Sequential(nn.Conv2d(64 + 128, in_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(in_channels),
                                      nn.ReLU(True)
                                      )

    def forward(self, x, df):
        # N1, _, H1, W1 = x.shape

        pred_score = self.pred_conv(x)

        pred_score = torch.sigmoid(pred_score)

        # boundary
        dist = torch.abs(pred_score - 0.5)
        boundary_att = 1 - (dist / 0.5)
        # br_enc = torch.cat([h1,
        #                     F.interpolate(h2, scale_factor=2, mode='bilinear')], 1)
        # br_enc = self.enc_Conv(br_enc)
        boundary_x = x * boundary_att

        # boundary_x = F.interpolate(h2, scale_factor=2, mode='bilinear') \
        #              * boundary_att  # 128 256 256
        # boundary_x = x \
        #              * boundary_att  # 128 256 256
        opt = self.br_conv(boundary_x)

        # certainty
        ori_map = self.conv_ori(x)
        score = F.softmax(ori_map, dim=1)
        score_top, _ = score.topk(k=2, dim=1)
        uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)  # bs, h, w
        uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)  # bs, 1, h, w
        un = uncertainty.clone()
        certain = 1 - uncertainty
        rough = certain * torch.cat([x, score], 1)
        rough1 = self.loc_conv(rough).clone()
        rough = self.BN_RELU(self.loc_conv(rough) / self.avg(certain))
        # norm = torch.tensor([[[[256, 256]]]]).type_as(x).to(x.device)
        # w = torch.linspace(-1.0, 1.0, 256).view(-1, 1).repeat(1, 256)
        # h = torch.linspace(-1.0, 1.0, 256).repeat(256, 1)
        # grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        # grid = grid.repeat(N1, 1, 1, 1).type_as(x).to(x.device)
        # grid = grid + df.permute(0, 2, 3, 1) / norm
        # refine_inp = F.grid_sample(x, grid)


        # certain = self.certan_conv(certain)
        # rough = self.BN_RELU(self.loc_conv_1(rough) / self.avg(certain))
        # certain = self.certan_conv(certain)
        # rough = self.BN_RELU(self.loc_conv_1(rough) / self.avg(certain))
        # certain = self.certan_conv(certain)
        # rough = self.BN_RELU(self.loc_conv_1(rough) / self.avg(certain))
        # certain = self.certan_conv(certain)
        # rough = self.BN_RELU(self.loc_conv_1(rough) / self.avg(certain))

        # rough = self.BN_RELU(rough / self.avg(certain))
        # print(rough.shape)
        # ref = torch.cat([rough, x], 1)
        # ref = self.r_conv(ref)
        # ref = self.res_Conv(ref)
        # print(ref.shape)
        # print(x.shape)
        N, _, H, W = df.shape

        '''算距离 相当于一个方向'''
        # distance_better = torch.sqrt(torch.sum(df ** 2, dim=1)) > 0.5  # 算距离 （1 ，256， 256）和distance_transform_edt一样
        # distance_better = distance_better.unsqueeze(1)
        # distance_better = torch.cat([distance_better, distance_better], 1)  # 增加维度变成(1，2，256，256)
        #
        # df[~distance_better] = 0  # 背景置为0 (1，2，256，256)
        mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        greater_mask = mag > 0.5
        greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        df[~greater_mask] = 0

        '''生成坐标grid 方便后续对应查找坐标'''


        # x0, y0 = torch.meshgrid(torch.arange(H), torch.arange(W))  # 相当于生成坐标网格(2，256，256)
        # grid = torch.stack([x0, y0], 0)
        #
        # grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()  # 增加维度(1, 2，256，256)
        #
        # trans_feature = grid + df
        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)
        grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()

        grid = grid + 1.0 * df

        # grid = trans_feature.permute(0, 2, 3, 1).transpose(1, 2)  # 增加维度(1, 256，256,2),grid的最后一维是2，分别对应x,y
        grid = grid.permute(0, 2, 3, 1).transpose(1, 2)

        final_grid = grid + 0.

        # 归一化到[-1, 1]
        grid[..., 0] = 2 * final_grid[..., 0] / (H - 1) - 1
        grid[..., 1] = 2 * final_grid[..., 1] / (W - 1) - 1
        # print(x.shape[0])
        final_grid = grid
        # print(final_grid)
        '''
        grid中最后一个维度的2表示在input中的相对索引位置（offset），函数的内部主要执行几件事：
            1.遍历output图像的所有像素坐标
            2.比如现在要求output中的（5, 5）坐标的特征向量，若通过查找grid中（5, 5）位置中的offset值为（0.1, 0.2）
            3.根据(0.1*W_in， 0.2*H_in)得到对应input图像上的位置坐标
            4.通过双线性插值得到该点的特征向量。
            5.将该特征向量copy到output图像的(5, 5)位置
            '''
        # 迭代3步（这个参数可以到时候调一下）border 超界的用边界像素值补充
        refine_inp = F.grid_sample(x, final_grid, )
        # refine_inp = F.grid_sample(refine_inp, final_grid,)
        # refine_inp = F.grid_sample(refine_inp, final_grid, )

        # refine_inp = F.grid_sample(refine_inp, final_grid, mode='bilinear', padding_mode='border')
        # side_out = self.side_conv(x)
        # refine_out = self.fuse_conv(torch.cat([x, refine_inp], dim=1))

        f_out = torch.cat([opt, rough, refine_inp], 1)

        f_out = self.ca(f_out)
        f_out = self.sa(f_out)
        f_out = self.down_Conv(f_out)

        # f = torch.cat([opt,rough],1)
        # f = self.ca(f)
        # f = self.sa(f)
        # df = self.down_Conv(f)
        #
        # mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        # greater_mask = mag > 0.5
        # greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        # df[~greater_mask] = 0
        #
        # grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)
        # grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()
        #
        # grid = grid + 1.0 * df
        #
        #
        # grid = grid.permute(0, 2, 3, 1).transpose(1, 2)
        #
        # final_grid = grid + 0.
        #
        # # 归一化到[-1, 1]
        # grid[..., 0] = 2 * final_grid[..., 0] / (H - 1) - 1
        # grid[..., 1] = 2 * final_grid[..., 1] / (W - 1) - 1
        # # print(x.shape[0])
        # final_grid = grid
        #
        # refine_inp = F.grid_sample(x, final_grid, mode='bilinear', padding_mode='border')

        # f_out = ref + refine_out
        return f_out + x,un