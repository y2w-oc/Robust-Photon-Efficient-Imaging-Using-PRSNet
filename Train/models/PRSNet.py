import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.io as scio
import numpy as np

# the feature extraction block used in the encoder (dense dilated fusion strategy)
class DDFS(nn.Module):
    def __init__(self, in_channels):
        outchannel_MS = 2
        super(DDFS, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1, 1, 1), padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=False))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu');
        init.constant_(self.conv1[0].bias, 0.0)

        self.dalited_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1, 1, 1), padding=2, dilation=2, bias=True),
            nn.ReLU(inplace=False))
        init.kaiming_normal_(self.dalited_conv1[0].weight, 0, 'fan_in', 'relu');
        init.constant_(self.dalited_conv1[0].bias, 0.0)

        self.conv2 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=1, dilation=1, bias=True),
                                   nn.ReLU(inplace=False))
        init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu');
        init.constant_(self.conv2[0].bias, 0.0)

        self.dalited_conv2 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=False))
        init.kaiming_normal_(self.dalited_conv2[0].weight, 0, 'fan_in', 'relu');
        init.constant_(self.dalited_conv2[0].bias, 0.0)

    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out2 = self.dalited_conv1(inputs)
        out3 = self.conv2(out2)
        out4 = self.dalited_conv2(out1)
        return torch.cat((out1, out2, out3, out4), 1)


class PixelWiseResShrinkBlock(nn.Module):
    def __init__(self, in_channels=32):
        super(PixelWiseResShrinkBlock, self).__init__()

        # residual learning
        ####################################################################################
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.ReLU(inplace=False))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.conv1[0].bias, 0.0)
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
        init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.conv2[0].bias, 0.0)
        ####################################################################################

        # scales parameters learning
        ####################################################################################
        self.scales_conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True),
                                   nn.ReLU(inplace=False))
        init.kaiming_normal_(self.scales_conv1[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.scales_conv1[0].bias, 0.0)

        self.scales_conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True),
                                          nn.ReLU(inplace=False))
        init.kaiming_normal_(self.scales_conv2[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.scales_conv2[0].bias, 0.0)

        self.scales_conv3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                          nn.Sigmoid())
        init.kaiming_normal_(self.scales_conv3[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.scales_conv3[0].bias, 0.0)
        ####################################################################################

        self.final_relu = nn.Sequential(nn.ReLU(inplace=False))

    def forward(self, inputs):
        B, C, T, height, width = inputs.size()

        # residual learning
        ####################################################################################
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        raw = conv2.view(-1, T, height, width)
        ####################################################################################

        conv_abs = torch.abs(raw)
        average = conv_abs.mean(dim=1)

        # pixel-wise thresholds generation
        ####################################################################################
        # min-max normlization
        conv_max, _ = conv_abs.max(dim=1, keepdim=True)
        conv_norm = conv_abs / (conv_max + 1e-7)

        scales = self.scales_conv1(conv_norm)
        scales = self.scales_conv2(scales)
        scales = self.scales_conv3(scales)
        scales = scales.squeeze()

        thres = torch.mul(average, scales)
        thres = thres.unsqueeze(1)
        ####################################################################################

        # soft thresholding
        ####################################################################################
        sub = conv_abs - thres
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        residual = torch.mul(torch.sign(raw), n_sub)
        ####################################################################################

        residual = residual.view(-1, C, T, height, width)

        out = self.final_relu(inputs + residual)

        return out


class PixelWiseResShrinkNet(nn.Module):
    def __init__(self, in_channels=1):
        super(PixelWiseResShrinkNet, self).__init__()

        # temporal window strategy
        ###################################################################################
        self.window = nn.Sequential(nn.Conv3d(1, 1, kernel_size=(5,1,1), stride=(1, 1, 1), padding=(2, 0, 0), bias=False), nn.ReLU(inplace=True))
        init.constant_(self.window[0].weight, 1)
        self.window[0].weight.requires_grad_(False)
        ###################################################################################

        # encoder
        ###################################################################################
        self.feat_extraction = DDFS(in_channels)
        self.feat_conv = nn.Sequential(nn.Conv3d(8, 2, kernel_size=1, stride=(1, 1, 1), bias=True), nn.ReLU(inplace=False))
        init.kaiming_normal_(self.feat_conv[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.feat_conv[0].bias, 0.0)

        self.dowmsapling1 = nn.Sequential(nn.Conv3d(2, 4, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),
                                 nn.ReLU(inplace=False))
        init.kaiming_normal_(self.dowmsapling1[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.dowmsapling1[0].bias, 0.0)
        self.dowmsapling2 = nn.Sequential(nn.Conv3d(4, 8, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),
                                 nn.ReLU(inplace=False))
        init.kaiming_normal_(self.dowmsapling2[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.dowmsapling2[0].bias, 0.0)
        self.dowmsapling3 = nn.Sequential(nn.Conv3d(8, 16, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),
                                 nn.ReLU(inplace=False))
        init.kaiming_normal_(self.dowmsapling3[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.dowmsapling3[0].bias, 0.0)
        self.dowmsapling4 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),
                                 nn.ReLU(inplace=False))
        init.kaiming_normal_(self.dowmsapling4[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.dowmsapling4[0].bias, 0.0)
        ###################################################################################

        # pixel-wise residual shrinkage blocks
        ###################################################################################
        self.prsblk0 = PixelWiseResShrinkBlock(32)
        self.prsblk1 = PixelWiseResShrinkBlock(32)
        self.prsblk2 = PixelWiseResShrinkBlock(32)
        self.prsblk3 = PixelWiseResShrinkBlock(32)
        self.prsblk4 = PixelWiseResShrinkBlock(32)
        self.prsblk5 = PixelWiseResShrinkBlock(32)
        self.prsblk6 = PixelWiseResShrinkBlock(32)
        self.prsblk7 = PixelWiseResShrinkBlock(32)
        self.prsblk8 = PixelWiseResShrinkBlock(32)
        self.prsblk9 = PixelWiseResShrinkBlock(32)


        # decoder
        ###################################################################################
        self.trans_convs = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
            nn.ReLU(inplace=False),  # 64 32
            nn.ConvTranspose3d(32, 16, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
            nn.ReLU(inplace=False),  # 32 28
            nn.ConvTranspose3d(16, 16, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(16, 4, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False))
        init.kaiming_normal_(self.trans_convs[0].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.trans_convs[2].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.trans_convs[4].weight, 0, 'fan_in', 'relu')
        init.normal_(self.trans_convs[6].weight, mean=0.0, std=0.001)

        self.final_conv = nn.Sequential(nn.Conv3d(4, 1, kernel_size=1, stride=(1, 1, 1), bias=True), nn.ReLU(inplace=False))
        init.kaiming_normal_(self.final_conv[0].weight, 0, 'fan_in', 'relu');
        init.constant_(self.final_conv[0].bias, 0.0)
        ###################################################################################

    def forward(self, inputs):
        # photon cluster generation
        ####################################################################################
        inputs = self.window(inputs)
        ####################################################################################

        # encoding
        ####################################################################################
        shallow_feat = self.feat_extraction(inputs)
        feat_conv_out = self.feat_conv(shallow_feat)
        dsfeat1 = self.dowmsapling1(feat_conv_out)
        dsfeat2 = self.dowmsapling2(dsfeat1)
        dsfeat3 = self.dowmsapling3(dsfeat2)
        dsfeat4 = self.dowmsapling4(dsfeat3)
        ####################################################################################

        # boost the signal with pixel-wise residual shrinkage blocks
        ####################################################################################
        prsb0 = self.prsblk0(dsfeat4)
        prsb1 = self.prsblk1(prsb0)
        prsb2 = self.prsblk2(prsb1)
        prsb3 = self.prsblk3(prsb2)
        prsb4 = self.prsblk4(prsb3)
        prsb5 = self.prsblk5(prsb4)
        prsb6 = self.prsblk6(prsb5)
        prsb7 = self.prsblk7(prsb6)
        prsb8 = self.prsblk8(prsb7)
        prsb9 = self.prsblk9(prsb8)
        ####################################################################################

        # decoding
        ####################################################################################
        trans_out = self.trans_convs(prsb9)
        out_cube = self.final_conv(trans_out)
        out_cube = torch.squeeze(out_cube, 1)
        ####################################################################################

        # soft_argmax calculation
        ####################################################################################
        smax = torch.nn.Softmax2d()
        weights = Variable(
            torch.linspace(1, 1024, 1024).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax(out_cube)
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)
        ####################################################################################

        return out_cube, soft_argmax
