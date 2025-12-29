
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.transforms import Resize, Compose, RandomCrop, ToTensor, ToPILImage, Normalize
import cv2
from DefEDNetmain.DefEDNet import SeparableConv2d
from FullNet import DoubleConv
from MyUnet import DoubleConvn
from RFBmodel import BasicRFB_a
from backbones.resnet.resnet_factory import get_resnet_backbone

from functools import partial

from backbones.scale_attention_layer import scale_atten_convblock, conv3x3, conv1x1
from models import DeepLab
from network import deeplabv3plus_resnet101
from smatunetmodels.layers import DepthwiseSeparableConv, CBAM, ChannelAttention, SpatialAttention
from smatunetmodels.unet_parts_depthwise_separable import DoubleConvDS
from einops import rearrange
nonlinearity = partial(F.elu, inplace=True)

# from attack_Score import attack
# from attack_GMM import attack
# from attack_Diff import attack_diffusion
from attack import attack

class MixSyncBatchNorm(nn.SyncBatchNorm):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MixSyncBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.aux_bn = nn.SyncBatchNorm(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats)
        self.batch_type = 'clean'

    def forward(self, input):
        if self.batch_type == 'adv':
            input = self.aux_bn(input)
        elif self.batch_type == 'clean':
            input = super(MixSyncBatchNorm, self).forward(input)
        elif self.batch_type == 'warm_up':
            batch_size = input.shape[0]
            input0 = super(MixSyncBatchNorm, self).forward(input[:batch_size//2])
            input1 = self.aux_bn(input[batch_size//2:])
            input = torch.cat((input0, input1), 0)
        else:
            # In setting of tri, we have labeled features, strong aug features, pt features, three sets
            assert self.batch_type == 'mix'
            batch_size = input.shape[0]
            clean_bd = batch_size // 3 * 2
            input0 = super(MixSyncBatchNorm, self).forward(input[:clean_bd])
            input1 = self.aux_bn(input[clean_bd:])
            input = torch.cat((input0, input1), 0)

        return input



class CAFM(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=(3,3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1,1,1), bias=bias)
        self.fc = nn.Conv3d(3*self.num_heads, 9, kernel_size=(1,1,1), bias=True)

        self.dep_conv = nn.Conv3d(9*dim//self.num_heads, dim, kernel_size=(3,3,3), bias=True, groups=dim//self.num_heads, padding=1)


    def forward(self, x):
        b,c,h,w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0,2,3,1) 
        f_all = qkv.reshape(f_conv.shape[0], h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3) 
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        #local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[1]//self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv) # B, C, H, W
        out_conv = out_conv.squeeze(2)


        # global SA
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output =  out + out_conv

        return output



class Denis_Net_(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(Denis_Net_, self).__init__()
        # print("构造Denis_CE_Net_")
        filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
      

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.cafm1= CAFM(dim=256,num_heads=8)
        self.cafm2= CAFM(dim=128,num_heads=8)
        self.cafm3= CAFM(dim=64,num_heads=8)
        self.conv1x1 = nn.Conv2d(128, 3, kernel_size=1, dilation=1, padding=0)
	

        self.decoder4 = DecoderBlock(512, filters[2])

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(128, filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.dsv1 = UnetDsv3(256, out_size=32, scale_factor=(512,512))
        self.dsv2 = UnetDsv3(128, out_size=32, scale_factor=(512, 512))
        self.dsv3 = UnetDsv3(64, out_size=32, scale_factor=(512, 512))
        

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
		
    def forward(self, x,x1=None):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
       
        # Decoder

        d4 = self.decoder4(e4) + e3
        d4 = self.cafm1(d4)
        d3 = self.decoder3(d4) + e2
        d3 = self.cafm2(d3)
        d2 = self.decoder2(d3) + e1
        d2 = self.cafm3(d2)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out1 = self.finalrelu2(out)
        out2 = self.finalconv3(out1)
        out3 = out2+self.conv1x1(torch.cat([out1,self.dsv1(d4),self.dsv2(d3),self.dsv3(d2)],dim=1))
        return out2, out3





class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(F_g,F_int,1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.w_g(g)
        # 上采样的 l 卷积
        x1 = self.w_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi




class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock_without_atrous(nn.Module):
    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out


class DACblock_with_inception(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = x + dilate3_out
        return out


class DACblock_with_inception_blocks(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv1 = DepthwiseSeparableConv(in_channels,in_channels//4,1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        # self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.conv3 = DepthwiseSeparableConv(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class my_up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=2):
        super(my_up,self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        self.conv1 = nn.Conv2d(in_channels,in_channels//2,1,padding=0)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x1_ = self.conv1(x1)
        x3 = x2*x1_
        x = torch.cat([x3, x1_], dim=1)
        return self.conv(x)+self.conv(x1)
class my_up1(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=2):
        super(my_up1,self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,padding=0)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x3 = x2*x1
        x = torch.cat([x3, x1], dim=1)
        return self.conv(x)+x1


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)


class New_Semic_Seg_Diff(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(New_Semic_Seg_Diff, self).__init__()
        resnet = get_resnet_backbone('resnet101')(pretrain=True)
        print("__init__, New_Semic_Seg")
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            DACblock(512),
            SPPblock(512)
        )

        self.representation = nn.Sequential(DecoderBlock(516, 256))

        self.decoder0 = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()

        )
        self.decoder0_1 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x_l, x_u=None, label=None, nf_model=None, cfg=None, eps=0, adv=False):
        if x_u is None:
            x_l1 = self.encoder(x_l)
            res_head_l1 = self.representation(x_l1)
            out_fin_labeled = self.decoder0(res_head_l1)
            out_labeled = self.decoder0_1(out_fin_labeled)
            return out_labeled, res_head_l1
        else:
            if adv:
                # 有标签预测
                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

                # 无标签预测
                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)

                # 无标签扰动增强预测
                # x_u1_pt = self.encoder(x_u.clone()).float()
                # pt = attack(x_u1_pt, label, self.representation, self.decoder0_1, nf_model, cfg, eps)
                x_u1_pt = self.encoder(x_u.clone()).float()
                res_head_u1 = self.representation(x_u1_pt)
                pt = attack_diffusion(res_head_u1, label, nf_model, cfg, eps)  # [B,256,Hf,Wf]
                fts_half_pt = res_head_u1 + pt
                # out_fts_half_pt = self.representation(fts_half_pt)
                out_fin_unlabeled_pt = self.decoder0(fts_half_pt)
                out_all_unlabeled_pt = self.decoder0_1(out_fin_unlabeled_pt)

                return out_labeled, out_unlabeled, out_all_unlabeled_pt, res_head_u1

            else:
                # 有标签预测
                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

                # 无标签预测
                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)
                res_head = torch.cat([res_head_l1, res_head_u1], dim=0)

                return out_labeled, out_unlabeled, res_head


class New_Semic_Seg_Score(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(New_Semic_Seg_Score, self).__init__()
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        print("__init__, New_Semic_Seg")
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            DACblock(512),
            SPPblock(512)
        )

        self.representation = nn.Sequential(DecoderBlock(516, 256))

        self.decoder0 = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()

        )
        self.decoder0_1 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x_l, x_u=None, label=None, nf_model=None, cfg=None, eps=0, adv=False):
        if x_u is None:
            x_l1 = self.encoder(x_l)
            res_head_l1 = self.representation(x_l1)
            out_fin_labeled = self.decoder0(res_head_l1)
            out_labeled = self.decoder0_1(out_fin_labeled)
            return out_labeled, res_head_l1
        else:
            if adv:
                # 有标签预测
                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

                # 无标签预测
                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)

                # 无标签扰动增强预测
                # x_u1_pt = self.encoder(x_u.clone()).float()
                # pt = attack(x_u1_pt, label, self.representation, self.decoder0_1, nf_model, cfg, eps)
                x_u1_rep = self.representation(self.encoder(x_u.clone()).float())  # [B,256,13,13]
                # pt = attack(x_u1_rep, label, None, None, nf_model, cfg, eps)  # attack 在 256×13×13 空间
                pt = attack(x_u1_rep, label, nf_model, cfg, eps)
                
                # mean = x_u1_pt.mean(dim=[2, 3], keepdim=True)
                # std = x_u1_pt.std(dim=[2, 3], keepdim=True) + 1e-6
                # x_u1_pt_norm = (x_u1_pt - mean) / std

                fts_half_pt = x_u1_rep + pt
                # out_fts_half_pt = self.representation(fts_half_pt)
                out_fin_unlabeled_pt = self.decoder0(fts_half_pt)
                out_all_unlabeled_pt = self.decoder0_1(out_fin_unlabeled_pt)

                return out_labeled, out_unlabeled, out_all_unlabeled_pt, res_head_u1

            else:
                # 有标签预测
                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

                # 无标签预测
                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)
                res_head = torch.cat([res_head_l1, res_head_u1], dim=0)

                return out_labeled, out_unlabeled, res_head




class New_Semic_Seg_GMM(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(New_Semic_Seg_GMM, self).__init__()
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        print("__init__, New_Semic_Seg")
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            DACblock(512),
            SPPblock(512)
        )
      

        self.representation = nn.Sequential(DecoderBlock(516, 256))

        self.decoder0 = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()

        )
        self.decoder0_1 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x_l, x_u=None, label=None, cfg=None, eps=0, adv=False):
        if x_u is None:
            x_l1 = self.encoder(x_l)
            res_head_l1 = self.representation(x_l1)
            out_fin_labeled = self.decoder0(res_head_l1)
            out_labeled = self.decoder0_1(out_fin_labeled)
            return out_labeled, res_head_l1
        else:
            if adv:
                # 有标签预测
                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

                # 无标签预测
                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)

                # 无标签扰动增强预测
                x_u1_pt = self.encoder(x_u.clone()).float()
                pt = attack(x_u1_pt, label, self.representation, self.decoder0_1, cfg, eps)
                mean = x_u1_pt.mean(dim=[2, 3], keepdim=True)
                std = x_u1_pt.std(dim=[2, 3], keepdim=True) + 1e-6
                x_u1_pt_norm = (x_u1_pt - mean) / std

                fts_half_pt = x_u1_pt_norm + pt
                out_fts_half_pt = self.representation(fts_half_pt)
                out_fin_unlabeled_pt = self.decoder0(out_fts_half_pt)
                out_all_unlabeled_pt = self.decoder0_1(out_fin_unlabeled_pt)

                return out_labeled, out_unlabeled, out_all_unlabeled_pt, res_head_u1

            else:
                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

                # 无标签预测
                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)

                res_head = torch.cat([res_head_l1, res_head_u1], dim=0)

                return out_labeled, out_unlabeled, res_head

class New_Semic_Seg(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(New_Semic_Seg, self).__init__()
        resnet = get_resnet_backbone('resnet101')(pretrain=True)
        print("__init__, New_Semic_Seg")
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            DACblock(512),
            SPPblock(512)
        )


        self.representation = nn.Sequential(DecoderBlock(516, 256))
   
        self.decoder0 = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()

        )
        self.decoder0_1 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1)
        )
            
    def forward(self, x_l, x_u=None,label=None, nf_model=None, loss_flow=None, cfg=None, eps=0, adv=False):
        if x_u==None:
            x_l1 = self.encoder(x_l)
            res_head_l1 = self.representation(x_l1)
            out_fin_labeled = self.decoder0(res_head_l1)
            out_labeled = self.decoder0_1(out_fin_labeled)

            return out_labeled, res_head_l1
        else:
            if adv:

                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)



                x_u1_pt = self.encoder(x_u.clone()).float()
                pt = attack(x_u1_pt, label, self.representation, self.decoder0_1, nf_model, loss_flow, cfg, eps)

                if torch.isnan(pt).any() or torch.isinf(pt).any():
                    print(">>> pt non-finite!", pt.min().item(), pt.max().item())


                pt = torch.nan_to_num(pt, nan=0.0, posinf=0.0, neginf=0.0)


                if eps is not None and eps > 0:
                    pt = pt.clamp(min=-eps, max=eps)

                mean = x_u1_pt.mean(dim=[2, 3], keepdim=True)
                std = x_u1_pt.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                x_u1_pt_norm = (x_u1_pt - mean) / std

                fts_half_pt = x_u1_pt_norm + pt
                
                

                fts_half_pt = x_u1_pt_norm + pt
                out_fts_half_pt = self.representation(fts_half_pt)
                out_fin_unlabeled_pt = self.decoder0(out_fts_half_pt)
                out_all_unlabeled_pt = self.decoder0_1(out_fin_unlabeled_pt)

                if torch.isnan(out_all_unlabeled_pt).any() or torch.isinf(out_all_unlabeled_pt).any():
                    print(">>> out_all_unlabeled_pt non-finite!")

                return out_labeled,out_unlabeled, out_all_unlabeled_pt, res_head_u1

            else:
            
                x_l1 = self.encoder(x_l)
                res_head_l1 = self.representation(x_l1)
                out_fin_labeled = self.decoder0(res_head_l1)
                out_labeled = self.decoder0_1(out_fin_labeled)

              
                x_u1 = self.encoder(x_u)
                res_head_u1 = self.representation(x_u1)
                out_fin_unlabeled = self.decoder0(res_head_u1)
                out_unlabeled = self.decoder0_1(out_fin_unlabeled)
                res_head = torch.cat([res_head_l1,res_head_u1],dim=0)
                
                return out_labeled, out_unlabeled,res_head



class Our_Semic_Seg(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(Our_Semic_Seg, self).__init__()
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        print("__init__, Our_Semic_Seg")
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            DACblock(512),
            SPPblock(512)
        )
        self.representation = nn.Sequential(
            nn.Conv2d(516, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.decoder0 = nn.Sequential(
            DecoderBlock(516, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()

        )
        self.decoder0_1 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1)
        )
        self.decoder0_2 = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
        )
        self.decoder1 = nn.Sequential(
            DecoderBlock(516, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.decoder1_1 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1)
        )
        self.decoder1_2 = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, x_0=None):
        if x_0 == None:
            x1 = self.encoder(x)
            res_head = self.representation(x1)
            out = self.decoder0(x1)
            out1 = self.decoder0_1(out)
            out_sdm = self.decoder0_2(out)

            return out1,out_sdm, res_head
        else:
            x1 = self.encoder(x)
            x2 = self.encoder(x_0)
            res_head = self.representation(x1)
            out = self.decoder0(x1)
            out1 = self.decoder0_1(out)
            out_sdm = self.decoder0_2(out)

            out_x2 = self.decoder1(x2)
            out2 = self.decoder1_1(out_x2)
            out_sdm2 = self.decoder1_2(out_x2)

            return out1, out_sdm, res_head,out2,out_sdm2




#        return output,out_sdm
class CE_Net_(nn.Module):
    def __init__(self, num_classes=3, num_channels=3):
        super(CE_Net_, self).__init__()
        filters = [64, 128, 256, 512]
        # resnet = models.resnet34(pretrained=True)
        resnet = get_resnet_backbone('resnet34')(pretrain=True)
        # self.corp = Compose([
        #     Resize(26),
        #     ToTensor(),
        #     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])


        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])



        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(128, filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.dsv1 = UnetDsv3(128, out_size=3, scale_factor=(512, 512))
        self.dsv2 = UnetDsv3(64, out_size=3, scale_factor=(512, 512))
        
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        # self.finalconv4 = nn.Conv2d(32, 1, 3, padding=1)
        # self.sigmoid = nn.Sigmoid()
		
    def forward(self, x, x_0=None):
        if x_0 == None:
            # Encoder
            x = self.firstconv(x)
            x = self.firstbn(x)
            x = self.firstrelu(x)
            x = self.firstmaxpool(x)
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)

            # Center
            e4 = self.dblock(e4)
            e4 = self.spp(e4)

            # Decoder


            d4 = self.decoder4(e4) + e3
            d3 = self.decoder3(d4) + e2
            out2 = self.dsv1(d3)
            d2 = self.decoder2(d3) + e1
            d1 = self.decoder1(d2)
            out1 = self.dsv2(d1)

            out = self.finaldeconv1(d1)
            out = self.finalrelu1(out)
            out = self.finalconv2(out)
            out1 = self.finalrelu2(out)
            out = self.finalconv3(out1)

            # out_dis = self.finalconv4(out1)
            # out_dis = self.sigmoid(out_dis)
            # out_p = out_dis.clone()
            # out_p[out_p>=0.5]=1
            # out_p[out_p<0.5]=0
            # out = out_p*out+out


            # return out,out_dis,out1,out2
            return out
        else:
            x = self.firstconv(x)
            x = self.firstbn(x)
            x = self.firstrelu(x)
            x = self.firstmaxpool(x)
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)

            # Center
            e4 = self.dblock(e4)
            e4 = self.spp(e4)

            # Decoder

            d4 = self.decoder4(e4) + e3
            d3 = self.decoder3(d4) + e2
            out2 = self.dsv1(d3)
            d2 = self.decoder2(d3) + e1
            d1 = self.decoder1(d2)
            out1 = self.dsv2(d1)

            out = self.finaldeconv1(d1)
            out = self.finalrelu1(out)
            out = self.finalconv2(out)
            out1 = self.finalrelu2(out)
            out = self.finalconv3(out1)


            x_0 = self.firstconv(x_0)
            x_0 = self.firstbn(x_0)
            x_0 = self.firstrelu(x_0)
            x_0 = self.firstmaxpool(x_0)
            e_1 = self.encoder1(x_0)
            e_2 = self.encoder2(e_1)
            e_3 = self.encoder3(e_2)
            e_4 = self.encoder4(e_3)

            # Center
            e_4 = self.dblock(e_4)
            e_4 = self.spp(e_4)

            # Decoder

            d_4 = self.decoder4(e_4) + e_3
            d_3 = self.decoder3(d_4) + e_2
            out2 = self.dsv1(d_3)
            d_2 = self.decoder2(d_3) + e_1
            d_1 = self.decoder1(d_2)
            out_1 = self.dsv2(d_1)

            out_0 = self.finaldeconv1(d_1)
            out_0 = self.finalrelu1(out_0)
            out_0 = self.finalconv2(out_0)
            out_1 = self.finalrelu2(out_0)
            out_u = self.finalconv3(out_1)


            return out,out_u




class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = self.relu(x)
        return x


class My_CE_Net_(nn.Module):
    def __init__(self, num_classes=3):
        super(My_CE_Net_, self).__init__()
        print("Construct My_CE_Net_+ Mutil loss(0.6 ,0.3,0.1 ")
        filters = [64, 128, 256, 512]
        resnet = get_resnet_backbone('resnet34')(pretrain=True)

        self.deeplabv3 = DeepLab()

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.block1 = ASPP(64,atrous_rates=[1,12, 24, 36])
        self.block2 = ASPP(256,atrous_rates=[1,12, 24, 36])
        self.block3 = ASPP(512,atrous_rates=[1,12, 24, 36])

        self.cbam1 = CBAM(48, reduction_ratio=16)

        self.cbam2 = CBAM(256)

        self.cbam3 = CBAM(96)
        self.cbam4 = CBAM(144)
        self.dsv1 = UnetDsv3(96, out_size=3, scale_factor=(416,416))
        self.dsv2 = UnetDsv3(144, out_size=3, scale_factor=(416, 416))


        self.blockconv = nn.Sequential(
            nn.Conv2d(256,48,1,bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True)
        )

        self.decoderconv = nn.Sequential(
            nn.Conv2d(48,48,3,padding=1,stride=1),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True))


        # self.finaldeconv1 = nn.ConvTranspose2d(144, 32, 4, 2, 1)
        self.finaldeconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(144,32,3,padding=1)
        )
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):

        # Encoder
        high_level_features, low_level_features = self.deeplabv3(x)
        # print("low_level_features:",low_level_features.shape)
        # print("high_level_features:", high_level_features.shape)

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        d1 = self.block1(e1)
        d2 = self.block2(e3)
        d3 = self.block3(e4)

        d1 = self.blockconv(d1)
        d1 = low_level_features+d1
        d1 = d1 + self.cbam1(d1)


        d2 = self.blockconv(d2)
        d3 = F.interpolate(d3, size=high_level_features.size()[2:],mode='bilinear', align_corners=True)
        d3 = high_level_features+d3
        d3 = d3 + self.cbam2(d3)
        d3 = self.blockconv(d3)



        p1 = self.decoderconv(d3)
        p2 = F.interpolate(d3, size=d2.size()[2:], mode='bilinear', align_corners=True)
        p3 = p2*d2;
        p3 = F.interpolate(p3, size=d1.size()[2:],mode='bilinear', align_corners=True)

        p4 = F.interpolate(p1,size=d1.size()[2:],mode='bilinear', align_corners=True)

        p5 = torch.cat([p3,p4],dim=1)
        p5 = p5 + self.cbam3(p5)

        out1 = self.dsv1(p5)


        s = self.decoderconv(d2)
        s1 =  F.interpolate(s,size=d1.size()[2:],mode='bilinear', align_corners=True)

        s2 = s1*d1
        s3 = torch.cat([s2,p5],dim=1)
        s3 = s3 + self.cbam4(s3)
        out2 = self.dsv2(s3)
        # Decoder

        out = self.finaldeconv1(s3)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out = F.interpolate(out, size=(416,416), mode='bilinear', align_corners=True)
        return out

class Our_Net_V4_(nn.Module):
    def __init__(self, num_classes=3):
        super(Our_Net_V4_, self).__init__()
        print("Construct My_CE_Net_+ Mutil loss(0.6 ,0.3,0.1) + Boundary_Attention ")
        filters = [64, 128, 256, 512]
        resnet = get_resnet_backbone('resnet34')(pretrain=True)

        # self.deeplabv3 = DeepLab()
        # self.mynet = My_CE_Net_()

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.block3 = MY_ASPP(512)

        self.bat1 = Boundary_Attention(64)
        self.bat2 = Boundary_Attention(128)
        self.bat3 = Boundary_Attention(256)
        self.bat4 = Boundary_Attention(512)



        self.up1 = up(768, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.decoder = DecoderBlock(64,64)
        # self.up4 = up(128, 32)

        self.dsv1 = UnetDsv3(256, out_size=3, scale_factor=(416, 416))
        self.dsv2 = UnetDsv3(128, out_size=3, scale_factor=(416, 416))
        self.dsv3 = UnetDsv3(64, out_size=3, scale_factor=(416, 416))
        self.dsv4 = UnetDsv3(64, out_size=3, scale_factor=(416, 416))
        self.scale = my_scale_atten_convblock(in_size=12, out_size=3)

        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.convlast = conv1x1(6, 3, 1)

    def forward(self, x):

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e1att = self.bat1(e1)
        e2 = self.encoder2(e1)
        e2att = self.bat2(e2)
        e3 = self.encoder3(e2)
        e3att = self.bat3(e3)

        e4 = self.encoder4(e3)
        e4 = self.block3(e4)
        e4att = self.bat4(e4)

        out = self.up1(e4att,e3att)
        out1 = self.dsv1(out)
        out = self.up2(out,e2att)
        out2 = self.dsv2(out)
        out = self.up3(out,e1att)
        out3 = self.dsv3(out)
        out = self.decoder(out)
        out4 = self.dsv4(out)



        # Decoder



        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out_Muti = torch.cat([out1, out2, out3,out4], dim=1)
        out_Muti = self.scale(out_Muti)
        out = self.convlast(torch.cat([out, out_Muti], dim=1))
        return [out,out1,out2,out3,out4]



class my_scale_atten_convblock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, drop_out=False):
        super(my_scale_atten_convblock, self).__init__()
        # if stride != 1 or in_size != out_size:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(in_size, out_size,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_size),
        #     )
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out
        # self.cbam = CBAM(4)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ELU(inplace=True)

        self.conv1 = conv1x1(in_size,out_size)
        self.conv3 = conv3x3(in_size, out_size)

        self.bn3 = nn.BatchNorm2d(out_size)

        # self.conv_gpb = SeparableConv2d(in_size, 256, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(out_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        # residual = self.conv1(x)
        x0 = self.max_pool(x)
        x0 = self.conv1(x0)
        x0 = self.bn_gpb(x0)
        x1 = self.avg_pool(x)
        x1 = self.conv1(x1)
        x1 = self.bn_gpb(x1)
        x2 = self.relu(self.conv1(x) * self.sg(x1) + self.conv1(x) * self.sg(x0))

        # out = self.relu(x)
        # s = self.sa(x)
        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)
        # print(out.shape)
        # print(self.sa(out).shape)
        # out =  out*self.ca(out)*s + residual
        out = out + x2

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


class MY_ASPP(nn.Module):
    def __init__(self, channel):
        super(MY_ASPP, self).__init__()
        self.dilate1 = SeparableConv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = SeparableConv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = SeparableConv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate4 = SeparableConv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.bn = nn.BatchNorm2d(channel)
        self.drop = nn.Dropout2d(0.5)
        self.sg = nn.Sigmoid()

        self.cbam = CBAM(channel)
        self.finalchannel = channel

        self.conv1x1_1 = SeparableConv2d(channel * 5, channel, kernel_size=1, dilation=1, padding=0)
        # self.conv1x1_2 = SeparableConv2d(channel * 3, channel * 2, kernel_size=1, dilation=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        # Master branch
        self.conv_master = SeparableConv2d(channel, channel, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channel)
        self.conv1x1 = SeparableConv2d(channel,channel,kernel_size=1)
        # self.conv1x2 = SeparableConv2d(256, channel, kernel_size=1)
        # Global pooling branch
        self.conv_gpb = SeparableConv2d(channel, channel, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_gpb = self.avg_pool(x)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)
        x_gpb = self.sg(x_gpb)


        x1 = self.conv1x1(x)
        x_se = x_gpb * x1

        # first block rate1
        d1 = self.dilate1(x)
        d1 = self.bn(d1)
        d1 = self.relu(d1)


        # second block rate3
        # d2 = torch.cat([d1, x], 1)
        # print("d1:",d1.shape)
        # print("x:",x1.shape)
        # d2 = self.conv1x2(d1) + x
        d2 = self.dilate2(x)
        d2 = self.bn(d2)
        d2 = self.relu(d2)
        d2_ = d2 + d1

        # third block rate5
        # d3 = torch.cat([d1, d2, x], 1)
        # d3 = self.conv1x2(d2) + x
        d3 = self.dilate3(x)
        d3 = self.bn(d3)
        d3 = self.relu(d3)
        d3_ = d3+d2

        # last block rate7
        # d4 = torch.cat([d1, d2, d3, x], 1)
        # d4 = self.conv1x2(d3) + x
        d4 = self.dilate4(x)
        d4 = self.bn(d4)
        d4 = self.relu(d4)
        d4_ = d3 + d4

        out = torch.cat([d1, d2_, d3_, d4_, x_se], 1)
        # out = out*self.ca(out)
        out = self.drop(out)
        out = self.conv1x1_1(out)
        out,_ = self.cbam(out)
        out = self.bn1(out)
        out = self.dropout(self.relu(out) + x1)
        return out

class Boundary_Attention(nn.Module):
    def __init__(self,channels):
        super(Boundary_Attention, self).__init__()
        self.cbam = CBAM(channels)
        self.conv3 = nn.Conv2d(2, 1, 1, padding=0)
        self.conv4 = nn.Conv2d(2*channels, channels, 1, padding=0)
        self.bn = nn.BatchNorm2d(1)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x1,x2 = self.cbam(x)
        x2 = torch.sigmoid(x2)
        threshold = 0.5
        p = x2.clone()
        p[p<threshold]=0
        p[p>=threshold]=1


        x3 = torch.cat([p,x2],dim=1)

        x4 = self.conv3(x3)
        x5 = self.bn(x4)
        x5 = torch.sigmoid(x5)
        mb = x5*x1
        f = self.drop(self.conv4(torch.cat([x1,mb],dim=1)) + x)

        return f

class SAD(nn.Module):
    def __init__(self, channels):
        super(SAD, self).__init__()
        self.conv2 = nn.Conv2d(channels,channels,3,padding=1)
        self.conv3 = nn.Conv2d(channels,channels,kernel_size=(7,1),padding=(3,0))
        self.conv4 = nn.Conv2d(channels,channels,kernel_size=(1,7),padding=(0,3))
        self.cbam = CBAM(channels)
        self.conv5 = nn.Conv2d(2, 1, 1, padding=0)
        self.conv6 = nn.Conv2d(2 * channels, channels, 1, padding=0)
        self.bn = nn.BatchNorm2d(1)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x0 = self.conv2(x)
        x1 = self.conv6(torch.cat([self.conv3(self.conv4(x0)),self.conv4(self.conv3(x0))],dim=1))
        x1 = torch.sigmoid(x1)
        x_ = x*x1
        x6, x2 = self.cbam(x_)
        x2 = torch.sigmoid(x2)
        threshold = 0.5
        p = x2.clone()
        p[p < threshold] = 0
        p[p >= threshold] = 1

        x3 = torch.cat([p, x2], dim=1)

        x4 = self.conv5(x3)
        x5 = self.bn(x4)
        x5 = torch.sigmoid(x5)
        mb = x5 * x1
        f = self.drop(self.conv6(torch.cat([x6, mb], dim=1)) + x_)

        return f














