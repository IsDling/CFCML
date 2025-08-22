import os
import copy
import clip
import timm
import torch
import argparse
import numpy as np
from math import sqrt
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, input_channel=3):
    """3x3 convolution with padding."""
    if input_channel == 3:
        conv_op = nn.Conv3d
    elif input_channel == 2:
        conv_op = nn.Conv2d
    elif input_channel == 1:
        conv_op = nn.Conv1d

    return conv_op(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, input_channel=3):
    """1x1 convolution."""
    if input_channel == 3:
        conv_op = nn.Conv3d
    elif input_channel == 2:
        conv_op = nn.Conv2d
    elif input_channel == 1:
        conv_op = nn.Conv1d

    return conv_op(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, mamba_layer=None, input_channel=3):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if input_channel == 3:
            batch_norm = nn.BatchNorm3d
        elif input_channel == 2:
            batch_norm = nn.BatchNorm2d
        elif input_channel == 1:
            batch_norm = nn.BatchNorm1d

        self.conv1 = conv3x3(inplanes, planes, stride, input_channel=input_channel)
        self.bn1 = batch_norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, input_channel=input_channel)
        self.bn2 = batch_norm(planes)
        self.mamba_layer = mamba_layer
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.mamba_layer is not None:
            global_att = self.mamba_layer(x)
            out += global_att

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1, mamba_layer=None, input_channel=3):
    if input_channel == 3:
        batch_norm = nn.BatchNorm3d
    if input_channel == 2:
        batch_norm = nn.BatchNorm2d
    elif input_channel == 1:
        batch_norm = nn.BatchNorm1d

    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride, input_channel=input_channel),
        batch_norm(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample, input_channel=input_channel))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes, mamba_layer=mamba_layer, input_channel=input_channel))

    return nn.Sequential(*layers)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, bimamba_type="none", input_channel=3):
        super().__init__()
        if input_channel == 3:
            batch_norm = nn.BatchNorm3d
        elif input_channel == 2:
            batch_norm = nn.BatchNorm2d
        elif input_channel == 1:
            batch_norm = nn.BatchNorm1d

        self.dim = dim
        self.nin = conv1x1(dim, dim, input_channel=input_channel)
        self.nin2 = conv1x1(dim, dim, input_channel=input_channel)
        self.norm2 = batch_norm(dim) # LayerNorm
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.norm = batch_norm(dim) # LayerNorm
        self.relu = nn.ReLU(inplace=True)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type=bimamba_type
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        act_x = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        # x_mamba = self.mamba(x_flat)

        x_flip_l = torch.flip(x_flat, dims=[2])
        x_flip_c = torch.flip(x_flat, dims=[1])
        x_flip_lc = torch.flip(x_flat, dims=[1,2])
        x_ori = self.mamba(x_flat)
        x_mamba_l = self.mamba(x_flip_l)
        x_mamba_c = self.mamba(x_flip_c)
        x_mamba_lc = self.mamba(x_flip_lc)
        x_ori_l = torch.flip(x_mamba_l, dims=[2])
        x_ori_c = torch.flip(x_mamba_c, dims=[1])
        x_ori_lc = torch.flip(x_mamba_lc, dims=[1,2])
        x_mamba = (x_ori+x_ori_l+x_ori_c+x_ori_lc)/4

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        # act_x = self.relu3(x)
        out += act_x
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, input_channel=3):
        super(DoubleConv, self).__init__()
        if input_channel == 3:
            conv_op = nn.Conv3d
            batch_norm = nn.BatchNorm3d
        elif input_channel == 2:
            conv_op = nn.Conv2d
            batch_norm = nn.BatchNorm2d

        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            batch_norm(out_ch),
            nn.ReLU(inplace=True),
            conv_op(out_ch, out_ch, 3, padding=1, dilation=1),
            batch_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class CrossAttention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.dim_q = self.dim_k = self.dim_v = input_dim

        # 定义线性变换函数
        self.linear_q = nn.Linear(input_dim, self.dim_k, bias=False).cuda()
        self.linear_k = nn.Linear(input_dim, self.dim_k, bias=False).cuda()
        self.linear_v = nn.Linear(input_dim, self.dim_v, bias=False).cuda()
        self._norm_fact = 1 / sqrt(self.dim_k)

    def forward(self, x1, x2):
        # x: batch_size, seq_len, input_dim

        q = self.linear_q(x1)  # batch_size, seq_len, dim_k
        k = self.linear_k(x2)  # batch_size, seq_len, dim_k
        v = self.linear_v(x2)  # batch_size, seq_len, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch_size, seq_len, seq_len
        # 归一化获得attention的相关系数  对每个字求sofmax 也就是每一行
        dist = torch.softmax(dist, dim=-1)  # batch_size, seq_len, seq_len
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)

        return att


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1, x2, attn_mask=None):
        batch_size = x1.size(0)

        # Linear projections
        query = self.q_linear(x1)  # (batch_size, seq_len_q, embed_dim)
        key = self.k_linear(x2)      # (batch_size, seq_len_k, embed_dim)
        value = self.v_linear(x2)  # (batch_size, seq_len_v, embed_dim)

        # Split into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)        # (batch_size, num_heads, seq_len_k, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_v, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        if attn_mask is not None:
            scores += attn_mask

        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Attention output
        attn_output = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_len_q, head_dim)

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (batch_size, seq_len_q, embed_dim)
        output = self.out_linear(attn_output)  # (batch_size, seq_len_q, embed_dim)

        return output


class CFCML(nn.Module):
    def __init__(self, num_classes=2, args=None, criterion=None):
        super(CFCML, self).__init__()

        self.num_classes = num_classes
        self.args = args
        self.criterion = criterion
        self.ifclinical = args.ifclinical
        self.multigranularity_num = 4
        self.eps = torch.tensor(0.00000001, dtype=torch.float32, requires_grad=False)
        self.mse_loss = nn.MSELoss(reduction='mean')

        # parameter setting
        if args.dataset_name == 'men':
            self.input_channel = 3
            in_ch = 1

            if self.ifclinical:
                self.modal_num = 4
            else:
                self.modal_num = 3
            self.image_modal_num = 3

            multigranularity_channels = [[64 * 64 * 12, 64 * 64 * 12, 64 * 64 * 12, None],
                                         [32 * 32 * 6, 32 * 32 * 6, 32 * 32 * 6, None],
                                         [16 * 16 * 3, 16 * 16 * 3, 16 * 16 * 3, None],
                                         [8 * 8 * 2, 8 * 8 * 2, 8 * 8 * 2, 6]]
        elif args.dataset_name == 'derm7pt':
            self.input_channel = 2
            in_ch = 3

            if self.ifclinical:
                self.modal_num = 3
            else:
                self.modal_num = 2
            self.image_modal_num = 2

            multigranularity_channels = [[56 * 56, 56 * 56, None],
                                          [28 * 28, 28 * 28, None],
                                          [14 * 14, 14 * 14, None],
                                          [7 * 7, 7 * 7, 5]]

        # image encoder
        if args.image_encoder == 'swin_t':
            self.swin_t = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, pretrained_cfg_overlay=dict(file='/media/ExtHDD02/ltl/project/crossmodal/crossmodalFusion/models/checkpoints/swin_tiny_patch4_window7_224.pth'))
            # 去掉最后的分类层
            swin_t_fe = nn.Sequential(*list(self.swin_t.children())[:-1])
            # image encoder
            if self.args.ifshared_image_encoder:
                self.image_encoder = swin_t_fe
            else:
                self.image_encoder = nn.ModuleList([])
                for i in range(self.image_modal_num):
                    self.image_encoder.append(copy.deepcopy(swin_t_fe))

            swin_t_channel_list = [96, 192, 384, 768]
            self.linear_image = nn.ModuleList([])

            for i in range(self.image_modal_num):
                self.linear_image.append(nn.Sequential(nn.Linear(swin_t_channel_list[-1], args.channels * 8),
                                                       nn.ReLU()))
        elif args.image_encoder == 'res_mamba':
            self.in_conv = nn.ModuleList([])
            self.layer1 = nn.ModuleList([])
            self.layer2 = nn.ModuleList([])
            self.layer3 = nn.ModuleList([])
            for i in range(self.image_modal_num):
                self.in_conv.append(DoubleConv(in_ch, args.channels, stride=2, kernel_size=3, input_channel=self.input_channel))
                self.layer1.append(make_res_layer(args.channels, args.channels * 2, args.blocks, stride=2,
                                                  mamba_layer=MambaLayer(args.channels * 2, input_channel=self.input_channel),
                                                  input_channel=self.input_channel))
                self.layer2.append(make_res_layer(args.channels * 2, args.channels * 4, args.blocks, stride=2,
                                                  mamba_layer=MambaLayer(args.channels * 4, input_channel=self.input_channel),
                                                  input_channel=self.input_channel))
                self.layer3.append(make_res_layer(args.channels * 4, args.channels * 8, args.blocks, stride=2,
                                                  mamba_layer=MambaLayer(args.channels * 8, input_channel=self.input_channel),
                                                  input_channel=self.input_channel))

        # clinical encoder
        if self.ifclinical:
            if args.clinical_encoder == 'clip':
                self.encoder_clinic, _ = clip.load("ViT-B/32")
                for param in self.encoder_clinic.parameters():
                    param.requires_grad = False
                self.encoder_clinic.eval()  # 设置为评估模式（冻结参数）
                self.linear_clinical = nn.Sequential(nn.Linear(512, self.args.channels*8),
                                                     nn.ReLU())

        # token compression
        self.crossmodal_conv_layers = nn.ModuleList([])
        if args.ifmultigranularity:
            for i in range(len(multigranularity_channels)):
                if i < len(multigranularity_channels) - 1:
                    crossmodal_conv_layer = nn.ModuleList([])
                    for j in range(self.image_modal_num):
                        crossmodal_conv_layer.append(nn.Conv1d(multigranularity_channels[i][j], self.args.num_latents_img, kernel_size=3, padding=1))
                    self.crossmodal_conv_layers.append(crossmodal_conv_layer)

        # last layer conv: image and clinical
        crossmodal_conv_layer = nn.ModuleList([])
        for j in range(len(multigranularity_channels[-1])):
            if args.ifdif_token_num and args.ifclinical:
                if j == len(multigranularity_channels[-1]) - 1:
                    num_latents = self.args.num_latents_clinical
                else:
                    num_latents = self.args.num_latents_img
            else:
                num_latents = self.args.num_latents

            crossmodal_conv_layer.append(nn.Conv1d(multigranularity_channels[-1][j], num_latents, kernel_size=3, padding=1))
        self.crossmodal_conv_layers.append(crossmodal_conv_layer)

        # multigranularity conv and upsample linear
        if args.ifmultigranularity:
            # multigranularity conv
            crossmodal_conv_layer = nn.ModuleList([])
            for j in range(self.modal_num):
                if args.ifdif_token_num and args.ifclinical:
                    if j == self.modal_num - 1:
                        num_latents = self.args.num_latents_clinical
                    else:
                        num_latents = self.args.num_latents_img
                else:
                    num_latents = self.args.num_latents

                if self.args.completion_type == 'concat' or self.args.completion_type == 'tc_att':
                    input_num_latents = num_latents*self.multigranularity_num*2
                elif self.args.completion_type == 'add':
                    input_num_latents = num_latents * self.multigranularity_num
                crossmodal_conv_layer.append(nn.Conv1d(input_num_latents, num_latents, kernel_size=3, padding=1))
            self.crossmodal_conv_layers.append(crossmodal_conv_layer)

            # multigranularity upsample linear, 将不同stage的channel upsample到 128
            self.crossmodal_linear_layers = nn.ModuleList([])
            for i in range(self.multigranularity_num):
                crossmodal_linear_layer = nn.ModuleList([])
                for j in range(self.image_modal_num):
                    if args.dataset_name == 'men' or args.dataset_name == 'abide':
                        crossmodal_linear_layer.append(nn.Linear(args.channels*pow(2, i),args.channels*8))
                    elif args.dataset_name == 'derm7pt':
                        crossmodal_linear_layer.append(nn.Linear(swin_t_channel_list[i], args.channels * 8))
                self.crossmodal_linear_layers.append(crossmodal_linear_layer)


        # CIE
        if self.args.ifcie:
            self.cm_atts = nn.ModuleList([]) #

            if self.args.ifmultigranularity:
                for i in range(self.multigranularity_num):
                    cm_att = nn.ModuleList([])
                    for j in range(self.modal_num):
                        if self.args.cie_att_type == 'ca':
                            cm_att.append(CrossAttention(self.args.channels * 8).cuda())
                        elif self.args.cie_att_type == 'mhca':
                            cm_att.append(MultiHeadCrossAttention(self.args.channels * 8, self.args.num_heads).cuda())
                    self.cm_atts.append(cm_att)
            else:
                for i in range(self.modal_num):
                    if self.args.cie_att_type == 'ca':
                        self.cm_atts.append(CrossAttention(self.args.channels * 8).cuda())
                    elif self.args.cie_att_type == 'mhca':
                        self.cm_atts.append(MultiHeadCrossAttention(self.args.channels * 8, self.args.num_heads).cuda())

        self.flatten = nn.Flatten(start_dim=1)
        self.pooling3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.pooling2d = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling1d = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=-1)

        # classifier
        self.linear_concat = nn.Sequential(nn.Linear(args.channels * 8 * self.modal_num, args.channels * 8),
                                           nn.ReLU(),
                                           nn.Dropout(p=self.args.cls_drop))

        self.linear_cls = nn.Sequential(nn.Linear(args.channels * 8, args.channels),
                                            nn.ReLU(),
                                            nn.Dropout(p=args.cls_drop),
                                            nn.Linear(args.channels, self.num_classes))

    def concat_features(self, input_list, dim=-1, ifpooling=0):
        for i in range(len(input_list)):
            if i == 0:
                if ifpooling:
                    output = self.flatten(self.pooling1d(input_list[i]))
                else:
                    output = input_list[i]
            else:
                if ifpooling:
                    output = torch.cat((output, self.flatten(self.pooling1d(input_list[i]))), dim=dim)
                else:
                    output = torch.cat((output, input_list[i]), dim=dim)
        return output

    def rearrange_operation(self, input_list):
        out_list = []
        if self.input_channel == 3:
            rearrange_info = 'b c h w d -> b c (h w d)'
        elif self.input_channel == 2:
            rearrange_info = 'b c h w -> b c (h w)'

        for i in range(len(input_list)):
            if i < self.image_modal_num:
                out_list.append(rearrange(input_list[i], rearrange_info))
            else:
                out_list.append(input_list[i])
        return out_list

    def get_multigranularity_shape(self, list):
        for i in range(len(list)):
            print('c{}'.format(str(i)))
            for j in range(len(list[i])):
                print(list[i][j].shape)

    def calc_prototype(self, features_list=None, features=None, labels=None, run_type='train'):
        protos = []
        if features_list is not None:
            for features in features_list:
                proto = []
                fea_norm = F.normalize(features, p=2, dim=1)  # L2 norm
                assert features.shape[0] == labels.shape[0]
                for k in range(self.num_classes):
                    mask_k = labels == k
                    emb_k = fea_norm[mask_k, :]
                    num = emb_k.shape[0]
                    if num == 0 and run_type == 'train':
                        print('exist 0 in class {} in a batch'.format(str(k)))
                    proto_k = F.normalize(torch.mean(emb_k, dim=0), p=2, dim=-1)
                    proto.append(proto_k)
                protos.append(proto)
        else:
            fea_norm = F.normalize(features, p=2, dim=1)  # L2 norm
            assert features.shape[0] == labels.shape[0]
            for k in range(self.num_classes):
                mask_k = labels == k
                emb_k = fea_norm[mask_k, :]
                num = emb_k.shape[0]
                if num == 0 and run_type == 'train':
                    print('exist 0 in class {} in a batch'.format(str(k)))
                proto_k = F.normalize(torch.mean(emb_k, dim=0), p=2, dim=-1)
                protos.append(proto_k)
        return {'protos': protos}

    def calc_contrastive_loss(self, anchor, pos=None, pos_list=None, neg_list=None):
        cs_anchor_neg = torch.tensor(0., dtype=torch.float32).cuda()

        if pos_list is None:
            cs_anchor_pos = torch.exp((torch.cosine_similarity(anchor, pos, dim=-1) + 1) / self.args.tau)
        else:
            cs_anchor_pos = torch.tensor(0., dtype=torch.float32).cuda()
            for pos in pos_list:
                cs_anchor_pos += torch.exp((torch.cosine_similarity(anchor, pos,
                                                                    dim=-1) + 1) / self.args.tau)  # +1是因为cosine similarity取值是[-1,1]

        for neg in neg_list:
            cs_anchor_neg += torch.exp((torch.cosine_similarity(anchor, neg, dim=-1) + 1) / self.args.tau)
        loss = -torch.log(cs_anchor_pos / (cs_anchor_pos + cs_anchor_neg) + self.eps)
        return loss

    def center_learning(self, features_list, labels, run_type='train', epoch=None):
        if self.args.CCRM_type == 'proto':
            # singlemodal
            singlemodal_centers = self.calc_prototype(features_list=features_list, labels=labels, run_type=run_type)['protos']
            # crossmodal
            crossmodal_centers = self.calc_prototype(features=torch.cat(features_list, dim=0), labels=labels.repeat(self.modal_num), run_type=run_type)['protos']
        
        if self.args.dataset_name == 'men':
            list_indices_0 = [index for index, element in enumerate(labels) if element == 0]
            list_indices_1 = [index for index, element in enumerate(labels) if element == 1]
            list_indices_2 = [index for index, element in enumerate(labels) if element == 2]

            # sample
            loss_sample = torch.tensor(0., dtype=torch.float32).cuda()
            for i in range(len(features_list)):
                features = features_list[i]
                ## label0
                features_label0 = features[list_indices_0]
                # pos_list0
                pos_list0 = []
                pos_list0.append(crossmodal_centers[0])
                for k in range(len(singlemodal_centers)):
                    pos_list0.append(singlemodal_centers[k][0])
                # calc loss label0
                for j in range(features_label0.shape[0]):
                    loss_sample += self.calc_contrastive_loss(anchor=features_label0[j], pos_list=pos_list0, neg_list=[features[list_indices_1 + list_indices_2][i] for i in range(len(list_indices_1 + list_indices_2))]) / features_label0.shape[0]

                ## label1
                features_label1 = features[list_indices_1]
                # pos_list1
                pos_list1 = []
                pos_list1.append(crossmodal_centers[1])
                for k in range(len(singlemodal_centers)):
                    pos_list1.append(singlemodal_centers[k][1])
                # calc loss label1
                for j in range(features_label1.shape[0]):
                    loss_sample += self.calc_contrastive_loss(anchor=features_label1[j], pos_list=pos_list1, neg_list=[features[list_indices_0 + list_indices_2][i] for i in range(len(list_indices_0 + list_indices_2))]) / features_label1.shape[0]

                ## label2
                features_label2 = features[list_indices_2]
                # pos_list2
                pos_list2 = []
                pos_list2.append(crossmodal_centers[2])
                for k in range(len(singlemodal_centers)):
                    pos_list2.append(singlemodal_centers[k][2])
                # calc loss label2
                for j in range(features_label2.shape[0]):
                    loss_sample += self.calc_contrastive_loss(anchor=features_label2[j], pos_list=pos_list2, neg_list=[features[list_indices_0 + list_indices_1][i] for i in range(len(list_indices_0 + list_indices_1))]) / features_label2.shape[0]
        elif self.args.dataset_name == 'derm7pt':
            list_indices_0 = [index for index, element in enumerate(labels) if element == 0]
            list_indices_1 = [index for index, element in enumerate(labels) if element == 1]

            # sample
            loss_sample = torch.tensor(0., dtype=torch.float32).cuda()
            for i in range(len(features_list)):
                features = features_list[i]
                ## label0
                features_label0 = features[list_indices_0]
                # pos_list0
                pos_list0 = []
                pos_list0.append(crossmodal_centers[0])
                for k in range(len(singlemodal_centers)):
                    pos_list0.append(singlemodal_centers[k][0])
                # calc loss label0
                for j in range(features_label0.shape[0]):
                    loss_sample += self.calc_contrastive_loss(anchor=features_label0[j], pos_list=pos_list0, neg_list=[features[list_indices_1][i] for i in range(len(list_indices_1))]) / features_label0.shape[0]

                ## label1
                features_label1 = features[list_indices_1]
                # pos_list1
                pos_list1 = []
                pos_list1.append(crossmodal_centers[1])
                for k in range(len(singlemodal_centers)):
                    pos_list1.append(singlemodal_centers[k][1])
                # calc loss label1
                for j in range(features_label1.shape[0]):
                    loss_sample += self.calc_contrastive_loss(anchor=features_label1[j], pos_list=pos_list1, neg_list=[features[list_indices_0][i] for i in range(len(list_indices_0))]) / features_label1.shape[0]

        # singlecenter
        loss_singlecenter = self.eps.cuda()
        for i in range(len(singlemodal_centers)):
            singlemodal_center = singlemodal_centers[i]
            for j in range(len(singlemodal_center)):
                anchor = singlemodal_center[j]
                pos = crossmodal_centers[j]

                neg_list = []
                for k in range(len(singlemodal_centers)):
                    for l in range(len(singlemodal_centers[k])):
                        if j != l:
                            neg_list.append(singlemodal_centers[k][l])

                loss_singlecenter = loss_singlecenter + self.calc_contrastive_loss(anchor, pos=pos, neg_list=neg_list) / self.modal_num

        # crossmodal center
        loss_crosscenter = self.eps.cuda()
        for i in range(len(crossmodal_centers)):
            anchor = crossmodal_centers[i]
            pos_list = [singlemodal_centers[x][i] for x in range(len(singlemodal_centers))]
            neg_list = []
            for j in range(len(crossmodal_centers)):
                if i != j:
                    neg_list.append(crossmodal_centers[j])

            loss_crosscenter = loss_crosscenter + self.calc_contrastive_loss(anchor, pos_list=pos_list, neg_list=neg_list) / self.num_classes

        return {'loss_sam': loss_sample,
                'loss_up': loss_singlecenter,
                'loss_cp': loss_crosscenter,
                'crossmodal_centers': crossmodal_centers,
                'singlemodal_centers': singlemodal_centers}

    def remove_list(self, list, remove_index):
        new_list = []
        for i in range(len(list)):
            if not i == remove_index:
                new_list.append(list[i])
        return new_list

    def CIE_fusion(self, att, list, index):
        ori_m = list[index]
        other_m = self.concat_features(self.remove_list(list, index), dim=1)
        sup_m = att[index](ori_m, other_m)
        if self.args.completion_type == 'concat':
            completed_m = torch.cat((ori_m, sup_m), dim=1)
        return completed_m

    def get_crossmodal_intermediate_layer_output(self, input, run_type):
        # 定义一个钩子函数来获取中间层的输出
        def get_intermediate_layer_output(layer_name):
            def hook(module, input, output):
                intermediate_outputs[layer_name] = output.detach()
                # intermediate_outputs[layer_name] = output
            return hook

        # 创建一个字典来存储中间层的输出
        intermediate_outputs = {}

        # 创建一个字典来存储 Hook 的引用
        hooks = {}
        # 注册钩子到模型的不同阶段
        # # Patch embedding
        # self.swin_t.patch_embed.register_forward_hook(get_intermediate_layer_output('stage0_output'))
        # Stage 1
        hooks['stage1'] = self.swin_t.layers[0].register_forward_hook(get_intermediate_layer_output('stage1_output'))
        # Stage 2
        hooks['stage2'] = self.swin_t.layers[1].register_forward_hook(get_intermediate_layer_output('stage2_output'))
        # Stage 3
        hooks['stage3'] = self.swin_t.layers[2].register_forward_hook(get_intermediate_layer_output('stage3_output'))

        # 前向传播
        with torch.no_grad():
            self.swin_t(input)
        # self.swin_t(input)

        # 获取中间层的输出
        stage1_output = intermediate_outputs['stage1_output']
        stage2_output = intermediate_outputs['stage2_output']
        stage3_output = intermediate_outputs['stage3_output']

        # 移除 Hook
        if not run_type=='train':
            for hook_value in hooks.values():
                hook_value.remove()

        return{'stage1_output':stage1_output,
               'stage2_output':stage2_output,
               'stage3_output':stage3_output}

    def crossmodal_tokenlinear_layers_operation(self, operation_layer, input):
        if self.args.image_encoder == 'swin_t' and self.args.dataset_name == 'derm7pt':
            # input: [bs, token, dim]
            return operation_layer(input.permute(0,2,1)).permute(0,2,1)
        elif self.args.image_encoder == 'res_mamba' and self.args.dataset_name == 'men':
            # input: [bs, dim, token]
            return operation_layer(input).permute(0, 2, 1)

    def forward(self, x, clinical, labels=None, run_type='train', epoch=None):
        if self.args.image_encoder == 'swin_t':
            if self.args.ifmultigranularity:
                # 这部分不将dim降维到128，在后面进行降维
                c1 = []
                c2 = []
                c3 = []
                for i in range(x.shape[1]):
                    multigranularity_hook_out = self.get_crossmodal_intermediate_layer_output(x[:, i, ...], run_type)
                    c1.append(rearrange(multigranularity_hook_out['stage1_output'], 'b w1 w2 c -> b (w1 w2) c'))
                    c2.append(rearrange(multigranularity_hook_out['stage2_output'], 'b w1 w2 c -> b (w1 w2) c'))
                    c3.append(rearrange(multigranularity_hook_out['stage3_output'], 'b w1 w2 c -> b (w1 w2) c'))
                c4 = [rearrange(self.image_encoder(x[:, i, ...]), 'b w1 w2 c -> b (w1 w2) c') for i in range(self.image_modal_num)]
            else:
                c4 = [self.linear_image[i](rearrange(self.image_encoder(x[:, i, ...]), 'b w1 w2 c -> b (w1 w2) c')) for i in range(self.image_modal_num)]


            if self.args.clinical_encoder == 'clip':
                branch_num = self.image_modal_num
        elif self.args.image_encoder == 'res_mamba':
            if self.ifclinical:
                c1 = [self.in_conv[i](torch.unsqueeze(x[:, i, ...], 1)) for i in range(self.image_modal_num)]
            else:
                c1 = [self.in_conv[i](torch.unsqueeze(x[:, i, ...], 1)) for i in range(self.modal_num)]
            if self.ifclinical:
                if self.args.clinical_encoder == 'clip':
                    branch_num = self.image_modal_num

            c2 = [self.layer1[i](c1[i]) for i in range(branch_num)]
            c3 = [self.layer2[i](c2[i]) for i in range(branch_num)]
            c4 = [self.layer3[i](c3[i]) for i in range(branch_num)]

        loss_sam = torch.tensor(0., dtype=torch.float32).cuda()
        loss_up = torch.tensor(0., dtype=torch.float32).cuda()
        loss_cp = torch.tensor(0., dtype=torch.float32).cuda()

        if self.args.ifmultigranularity:
            if self.args.image_encoder == 'swin_t':
                p1 = [self.crossmodal_linear_layers[0][i](self.crossmodal_conv_layers[0][i](c1[i])) for i in range(branch_num)]
                p2 = [self.crossmodal_linear_layers[1][i](self.crossmodal_conv_layers[1][i](c2[i])) for i in range(branch_num)]
                p3 = [self.crossmodal_linear_layers[2][i](self.crossmodal_conv_layers[2][i](c3[i])) for i in range(branch_num)]
                p4 = [self.crossmodal_linear_layers[3][i](self.crossmodal_conv_layers[3][i](c4[i])) for i in range(branch_num)]
            elif self.args.image_encoder == 'res_mamba':
                p1 = [self.crossmodal_linear_layers[0][i](self.crossmodal_conv_layers[0][i](self.rearrange_operation(c1)[i].permute(0,2,1))) for i in range(branch_num)]  # rearrange out: [batch, dim, token]
                p2 = [self.crossmodal_linear_layers[1][i](self.crossmodal_conv_layers[1][i](self.rearrange_operation(c2)[i].permute(0,2,1))) for i in range(branch_num)]
                p3 = [self.crossmodal_linear_layers[2][i](self.crossmodal_conv_layers[2][i](self.rearrange_operation(c3)[i].permute(0,2,1))) for i in range(branch_num)]
                p4 = [self.crossmodal_conv_layers[3][i](self.rearrange_operation(c4)[i].permute(0,2,1)) for i in range(branch_num)]

            if self.ifclinical and self.args.clinical_encoder == 'clip':
                with torch.no_grad():
                    for i in range(clinical.shape[0]):
                        if i == 0:
                            x_clinical = torch.unsqueeze(self.encoder_clinic.encode_text(clinical[i]),dim=0)
                        else:
                            x_clinical = torch.cat((x_clinical,torch.unsqueeze(self.encoder_clinic.encode_text(clinical[i]),dim=0)),dim=0)
                    x_clinical = x_clinical.type(torch.FloatTensor).cuda() # 转为float, x_clinical: [batch_size, token, dim]

                p1.append(self.crossmodal_conv_layers[3][-1](self.linear_clinical(x_clinical)))
                p2.append(self.crossmodal_conv_layers[3][-1](self.linear_clinical(x_clinical)))
                p3.append(self.crossmodal_conv_layers[3][-1](self.linear_clinical(x_clinical)))
                p4.append(self.crossmodal_conv_layers[3][-1](self.linear_clinical(x_clinical)))

            if self.args.ifcie:
                tc1 = [self.CIE_fusion(self.cm_atts[0], p1, i) for i in range(self.modal_num)]
                tc2 = [self.CIE_fusion(self.cm_atts[1], p2, i) for i in range(self.modal_num)]
                tc3 = [self.CIE_fusion(self.cm_atts[2], p3, i) for i in range(self.modal_num)]
                tc4 = [self.CIE_fusion(self.cm_atts[3], p4, i) for i in range(self.modal_num)]
                tc_out = [self.flatten(self.pooling1d(self.crossmodal_conv_layers[4][i](torch.cat((tc1[i],tc2[i],tc3[i],tc4[i]),dim=1)).permute(0,2,1))) for i in range(self.modal_num)]
        else:
            if self.args.image_encoder == 'swin_t':
                p4 = [self.crossmodal_conv_layers[0][i](c4[i]) for i in range(self.image_modal_num)] # c4[i]:(batch, token, dim)
            elif self.args.image_encoder == 'res_mamba':
                p4 = [self.crossmodal_conv_layers[0][i](self.rearrange_operation(c4)[i].permute(0,2,1)) for i in range(branch_num)]

            if self.ifclinical and self.args.clinical_encoder == 'clip':
                with torch.no_grad():
                    for i in range(clinical.shape[0]):
                        if i == 0:
                            x_clinical = torch.unsqueeze(self.encoder_clinic.encode_text(clinical[i]),dim=0)
                        else:
                            x_clinical = torch.cat((x_clinical,torch.unsqueeze(self.encoder_clinic.encode_text(clinical[i]),dim=0)),dim=0)
                    x_clinical = x_clinical.type(torch.FloatTensor).cuda() # 转为float
                p4.append(self.crossmodal_conv_layers[0][-1](self.linear_clinical(x_clinical)))

            if self.args.ifcie:
                tc_out = [self.flatten(self.pooling1d(self.CIE_fusion(self.cm_atts, p4, i).permute(0, 2, 1))) for i in range(self.modal_num)]

        if run_type == 'train':
            cl_out = self.center_learning(tc_out, labels, run_type, epoch)

            loss_sam = cl_out['loss_sam'] * self.args.w_loss_sam
            loss_up = cl_out['loss_up'] * self.args.w_loss_up
            loss_cp = cl_out['loss_cp'] * self.args.w_loss_cp

        concated_features = self.concat_features(tc_out)
        f_out = self.linear_concat(concated_features)
        out_prediction = self.linear_cls(f_out)

        if not run_type == 'train':
            return {'out_prediction': out_prediction}

        main_cls_loss = torch.mean(self.criterion(out_prediction, labels)) * self.args.w_main_cls

        loss = main_cls_loss + loss_sam + loss_up + loss_cp

        return {'out_prediction': out_prediction,
                'loss': loss,
                'main_cls_loss': main_cls_loss,
                'loss_sam': loss_sam,
                'loss_up': loss_up,
                'loss_cp': loss_cp}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--method_name', type=str, default='CFCML')
    parser.add_argument('--multi_gpu', type=int, default=0)
    parser.add_argument('--ifbalanceloader', type=int, default=0)
    parser.add_argument('--ifbalanceloader2', type=int, default=0) # Dataloader 一次性load batch_size个数据, 只对训练时有效
    parser.add_argument('--ifbalanceloader3', type=int, default=0) # Dataloader 一次性load num_classes个数据, 只对训练时有效
    parser.add_argument('--ifbalanceloader4', type=int, default=1) # Dataloader 一次性load num_classes个数据, 只对训练时有效, 并进行online aug
    parser.add_argument('--ifbalanceloader5', type=int, default=0)
    parser.add_argument('--ifoffline_data_aug', type=int, default=0)
    parser.add_argument('--ifonline_data_aug', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default='men')
    parser.add_argument('--img_size_men', type=int, default=128)
    parser.add_argument('--fold_train', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--train_epochs_abide', type=int, default=800)
    parser.add_argument('--early', type=int, default=400) # early stop
    parser.add_argument('--ifwarmup', type=bool, default=True)
    parser.add_argument('--run_type', type=str, default='train') # train, test, vis, heatmap, vis_space, heatmap_moe_weights, wilcoxon_test_get_pred, get_param
    parser.add_argument('--loss_type', type=str, default='ce')
    parser.add_argument('--testTraindata', type=int, default=0) # test_traindata
    parser.add_argument('--test_type', type=str, default='ori') # ori, ori_noise
    parser.add_argument('--batch_size_men', type=int, default=36)
    parser.add_argument('--batch_size_derm7pt', type=int, default=64)
    parser.add_argument('--train_lr_men', type=float, default=0.0005)
    parser.add_argument('--train_lr_derm7pt', type=float, default=0.0001)
    parser.add_argument('--cls_drop', type=float, default=0.5)
    parser.add_argument('--weight_decay_value', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=5, help='学习率下降策略，每隔多少个epoch对学习率进行下降')
    parser.add_argument('--lr_gamma', type=float, default=0.8, help='学习率下降策略，每隔step_size个epoch将学习率下降为之前的lr_gamma')
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--save_epoch_start', type=int, default=5)
    parser.add_argument('--ifclinical', type=int, default=1)
    parser.add_argument('--image_encoder', type=str, default='res_mamba')
    parser.add_argument('--clinical_encoder', type=str, default='clip')
    parser.add_argument('--ifshared_image_encoder', type=int, default=1) # 不同的模态图像使用共享权重的encoder提取特征，在image_encoder为swin_t时起作用
    parser.add_argument('--ifcie', type=int, default=1) # Crossmodal information enhancement
    parser.add_argument('--cie_att_type', type=str, default='mhca')
    parser.add_argument('--completion_type', type=str, default='concat')
    parser.add_argument('--num_latents_img', type=int, default=32)
    parser.add_argument('--num_latents_clinical', type=int, default=16)
    parser.add_argument('--ifdif_token_num', type=int, default=1)
    #model
    parser.add_argument('--mamba_drop', type=float, default=0.5)
    parser.add_argument('--mamba_drop_path', type=float, default=0.5)
    parser.add_argument('--if_cls_token', type=float, default=0)
    parser.add_argument('--final_pool_type', type=str, default='mean')
    parser.add_argument('--multi_gpus', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--finetuning', type=int, default=0)
    parser.add_argument('--ft_rate', type=float, default=0.1)
    parser.add_argument('--w_main_cls', type=float, default=1)
    parser.add_argument('--w_loss_sam', type=float, default=0.03)
    parser.add_argument('--w_loss_up', type=float, default=1)
    parser.add_argument('--w_loss_cp', type=float, default=1)
    parser.add_argument('--CCRM_diff_loss_type', type=str, default='cs')
    parser.add_argument('--ifmultigranularity', type=int, default=1) # multi-granularity
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--CCRM_type', type=str, default='proto') # proto, targeted_proto, targeted_proxy
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--blocks', type=int, default=3) #很占内存空间
    #mamba
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    batch_size = 24

    if args.dataset_name == 'men':
        args.image_encoder = 'res_mamba'
        input = torch.zeros((batch_size, 3, 24, 128, 128)).cuda()  # men
        if args.clinical_encoder == 'clip':
            clinical = torch.zeros((batch_size, 6, 77)).type(torch.LongTensor).cuda()
        else:
            clinical = torch.zeros((batch_size, 9)).cuda()  # men
        num_classes = 3
    elif args.dataset_name == 'derm7pt':
        args.image_encoder = 'swin_t'
        input = torch.zeros((batch_size, 2, 3, 224, 224)).cuda()  # derm7pt
        if args.clinical_encoder == 'clip':
            clinical = torch.zeros((batch_size, 5, 77)).type(torch.LongTensor).cuda()
        else:
            clinical = torch.zeros((batch_size, 20)).cuda()
        num_classes = 2
        args.image_encoder = 'swin_t'

    label = torch.tensor(np.random.randint(num_classes, size=batch_size)).cuda()

    model = CFCML(num_classes=num_classes, args=args, criterion=nn.CrossEntropyLoss()).cuda()

    out = model(input, clinical=clinical, labels=label)
    print(out['out_prediction'].shape)