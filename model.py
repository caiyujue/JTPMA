import os
import scipy.io
import numpy as np
from collections import OrderedDict
from glob import glob

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, Tensor

from typing import Optional, List
import copy
import math

from utils import *
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

from PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


def append_params1(params, module, prefix):
    for j,child in enumerate(module.named_children()):
        for k,p in child[1]._parameters.items():
            if p is None: continue
            name = prefix+'_'+str(j)+'_'+k
            
            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=0.0001, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(2.0).pow(self.beta)
        x = x.div(div)
        return x

class MDNet(nn.Module):
    def __init__(self, model_path = None, model_path1=None, restore_path=None, 
                load_pretrain = False,mode='normal',memory_sz=5, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.mode = mode
        reduce_dim = 1024

        self.RGB_layers = nn.Sequential(OrderedDict([
                ('conv1_RGB', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        )),
                ('conv2_RGB', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2,dilation=1),
                                        nn.ReLU(),
                                        LRN(),
                                        )),
                ('conv3_RGB', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,dilation=3),
                                        nn.ReLU(),
                                        ))
        ]))

        self.T_layers = nn.Sequential(OrderedDict([
                ('conv1_T', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        )),
                ('conv2_T', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2,dilation=1),
                                        nn.ReLU(),
                                        LRN(),
                                        )),
                ('conv3_T', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,dilation=3),
                                        nn.ReLU(),
                                        ))
        ]))

        self.redection_layers = nn.Sequential(OrderedDict([
                ('re_down', nn.Sequential(
                                    nn.Conv2d(2048, reduce_dim, kernel_size=1, padding=0, bias=False),
                                )
                ),
                ('re_cascade', nn.Sequential(
                                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.GroupNorm(32,reduce_dim),    
                                    
                                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.GroupNorm(32,reduce_dim),
                                    
                                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.GroupNorm(32,reduce_dim),
                                )  
                ),
           
        ]))

        self.linear_layers =  nn.Sequential(OrderedDict([       
                ('cls_fc4',   nn.Sequential(
                                        nn.Linear(1024 * 3 * 3, 512),
                                        nn.ReLU()
                                        )),
                ('cls_fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()
                                        )),
                ]))

        self.shared_linear =  nn.Sequential(OrderedDict([       
                ('shared_fc4',   nn.Sequential(
                                        nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU()
                                        )),
                ('shared_fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()
                                        )),
                ]))

        self.fuse_layers =  nn.Sequential(OrderedDict([
                ('fuse_fc4',   nn.Sequential(
                                        nn.Linear(1024* 3 * 3, 512),
                                        nn.ReLU()
                                        )),
                ('fuse_fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()
                                        )),
                ]))
               
        self.cls_branches = nn.ModuleList([nn.Sequential(
                                                         nn.Dropout(0.5),
                                                         nn.Linear(512, 2)) for _ in range(K)])

        self.match_branches = nn.ModuleList([nn.Sequential(
                                                         nn.Dropout(0.5),
                                                         nn.Linear(512, 2)) for _ in range(K)])
        
        self.roi_align_model = PrRoIPool2D(3, 3, 1./8).cuda()
        
        self.receptive_field = 75.   # r[n]=r[n-1] + d(k[n] -1 )*(s[1]*...*s[n-1])

        self.build_param_dict()
        self.zero_init_offset = True

        if load_pretrain:
            #self.restore_model(model_path)
            self.load_model_tracking(model_path)
            #self.load_model_part(model_path1)

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.RGB_layers.named_modules():
            append_params1(self.params, module, name)

        for name, module in self.T_layers.named_modules():
            append_params1(self.params, module, name)

        for name, module in self.redection_layers.named_modules():
            append_params1(self.params, module, name)

        for name, module in self.shared_linear.named_modules():
            append_params1(self.params, module, name)

        for name, module in self.linear_layers.named_modules():
            append_params1(self.params, module, name)

        for name, module in self.fuse_layers.named_modules():
            append_params1(self.params, module, name)

        ####################################################
        for k, module in enumerate(self.cls_branches):
            append_params1(self.params, module, 'cls_fc6%d'%(k))

        for k, module in enumerate(self.match_branches):
            append_params1(self.params, module, 'match_%d'%(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
 
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def get_backbone_feat(self,xR=None, xT=None,out_layer='conv3'):
        ## LAYER1
        featR = self.RGB_layers.conv1_RGB(xR)
        featT = self.T_layers.conv1_T(xT)
        ## LAYER2  
        featR1 = self.RGB_layers.conv2_RGB(featR)
        featT1 = self.T_layers.conv2_T(featT)
        ## LAYER3  
        featR2 = self.RGB_layers.conv3_RGB(featR1)
        featT2 = self.T_layers.conv3_T(featT1)
        if out_layer == 'conv3':
            return featR2,featT2

    def get_target_feat(self, RGB_target, T_target):
        RGB_target = self.RGB_layers(RGB_target)
        T_target = self.T_layers(T_target)
        RT_target = torch.cat([RGB_target, T_target], 1)#1,1024,3,3
        self.feat_target_RGBT = RT_target[0:1,...]#.view(RT_target.shape[0],-1)
        self.feat_target_RGB = RGB_target.view(RGB_target.shape[0],-1)
        self.feat_target_T = T_target.view(T_target.shape[0],-1)

    def Weighted_GAP(self,supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask.float(), (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
        return supp_feat

    def similarity_func(self,feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) #* 10.0
        return out

    def masked_average_pooling(self,feature, mask):
        mask = F.interpolate(mask, size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    def SSP_func(self,feature_q, out):
        bs = feature_q.shape[0]
        C = feature_q.shape[1]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)#b,2,hw
        pred_fg = pred_1[:, 1]#b,hw
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7 #0.9 #0.6
            bg_thres = 0.6 #0.6
            cur_feat = feature_q[epi].view(C, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))#1,C
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(C, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(C, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def get_query_mask(self,feat_map,FP, BP):
        out_0 = self.similarity_func(feat_map, FP, BP)
        
        ##################### Self-Support Prototype (SSP) #####################
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feat_map, out_0)
        
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7

        out_1 = self.similarity_func(feat_map, FP_1, BP_1)
       
        SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feat_map, out_1)

        FP_2 = FP * 0.5 + SSFP_2 * 0.5
        BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7

        FP_2 = FP * 0.5 + FP_1 * 0.2 + FP_2 * 0.3
        BP_2 = BP * 0.5 + BP_1 * 0.2 + BP_2 * 0.3

        out_2 = self.similarity_func(feat_map, FP_2, BP_2)
        out_2 = out_2 * 0.7 + out_1 * 0.3

        return [out_0, out_2]#out_2

    def get_supp_prototype(self, RGB_target, T_target, supp_mask): 
        feature_fg_list = []
        feature_bg_list = [] 
        feat_map = torch.cat((RGB_target,T_target),1)  

        feature_fg = self.masked_average_pooling(feat_map,
                                            (supp_mask == 1).float().cuda())[None, :]
        feature_bg = self.masked_average_pooling(feat_map,
                                            (supp_mask == 0).float().cuda())[None, :]
        feature_fg_list.append(feature_fg)
        feature_bg_list.append(feature_bg)
        self.FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        self.BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        feature_fg = self.masked_average_pooling(RGB_target,
                                            (supp_mask == 1).float().cuda())[None, :]
        feature_bg = self.masked_average_pooling(RGB_target,
                                            (supp_mask == 0).float().cuda())[None, :]
        self.R_FP =feature_fg[0].unsqueeze(-1).unsqueeze(-1)
        self.R_BP =feature_bg[0].unsqueeze(-1).unsqueeze(-1)

        feature_fg = self.masked_average_pooling(T_target,
                                            (supp_mask == 1).float().cuda())[None, :]
        feature_bg = self.masked_average_pooling(T_target,
                                            (supp_mask == 0).float().cuda())[None, :]
        self.T_FP =feature_fg[0].unsqueeze(-1).unsqueeze(-1)
        self.T_BP =feature_bg[0].unsqueeze(-1).unsqueeze(-1)

    def forward(self,xR=None,xT=None,feat=None,r_anchor=None,t_anchor=None,cls_anchor=None,
                k=0,in_layer='conv1', out_layer='fc6'):
        
        if in_layer=='conv1':
            featR,featT = self.get_backbone_feat(xR,xT,out_layer='conv3')
            feat_RGBT = torch.cat((featR,featT),1)
            
            if feat is not None:
                sim_rt = self.get_query_mask(feat_RGBT,self.FP,self.BP)
            else:
                sim_rt = None
            
            if r_anchor is not None:
                
                sim_r = self.get_query_mask(featR,self.R_FP,self.R_BP)
            else:
                sim_r = None

            if t_anchor is not None: 
                
                sim_t = self.get_query_mask(featT,self.T_FP,self.T_BP)
            else:
                sim_t = None

            if out_layer == 'conv4':
                return featR,featT,feat_RGBT#,sim_rt#,sim_r,sim_t

        elif in_layer=='conv3':
            featR = xR
            featT = xT
            feat_RGBT = torch.cat((featR,featT),1)
            supp_similarity_fg = F.cosine_similarity(feat_RGBT, self.FP, dim=1)
            supp_similarity_bg = F.cosine_similarity(feat_RGBT, self.BP, dim=1)
            supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) #* 10.0

            supp_similarity_fg1 = F.cosine_similarity(featR, self.R_FP, dim=1)
            supp_similarity_bg1 = F.cosine_similarity(featR, self.R_BP, dim=1)
            supp_out1 = torch.cat((supp_similarity_bg1[:, None, ...], supp_similarity_fg1[:, None, ...]), dim=1) #* 10.0

            supp_similarity_fg2 = F.cosine_similarity(featT, self.T_FP, dim=1)
            supp_similarity_bg2 = F.cosine_similarity(featT, self.T_BP, dim=1)
            supp_out2 = torch.cat((supp_similarity_bg2[:, None, ...], supp_similarity_fg2[:, None, ...]), dim=1) #* 10.0

            if out_layer == 'sim':
                return supp_out#,supp_out1,supp_out2

        elif in_layer=='fc4':
            # 对特征进行降维，实质上是对任务有用信息的提取
            feat_rsh=self.shared_linear(xR)
            feat_tsh=self.shared_linear(xT)
            feat_rt =self.linear_layers(feat)

            feat_fuse = feat_rt+feat_rsh+feat_tsh

            clsiou_feat = self.cls_branches[k](feat_fuse)#b,2
            clsiou_out = clsiou_feat.view(clsiou_feat.size(0), -1)
            clsiou_feat1 = self.cls_branches[k](feat_rsh)#b,2
            clsiou_out1 = clsiou_feat1.view(clsiou_feat1.size(0), -1)
            clsiou_feat2 = self.cls_branches[k](feat_tsh)#b,2
            clsiou_out2 = clsiou_feat2.view(clsiou_feat2.size(0), -1)
          
            if out_layer=='fc6':
                return clsiou_out#feat_rsh,feat_tsh,feat_fuse,\
                       #clsiou_out,clsiou_out1,clsiou_out2
            elif out_layer=='fc45':
                return clsiou_out,clsiou_out1,clsiou_out2
            elif out_layer=='fc5':
                return feat_rsh,feat_tsh,feat_fuse
        
        elif in_layer=='match':
            if xR is not None:
                n_pos = xR.shape[0]
                n_neg = xT.shape[0]
                ff_pos = feat.repeat(n_pos,1,1,1)
                ff_neg = feat.repeat(n_neg,1,1,1)

                fuse_pos = torch.cat((ff_pos,xR),dim=1)#ff_pos+xR#
                fuse_neg = torch.cat((ff_neg,xT),dim=1)#ff_neg+xT#
                
                fuse_pos = self.redection_layers.re_down(fuse_pos)
                fuse_neg = self.redection_layers.re_down(fuse_neg)

                fuse_pos = self.redection_layers.re_cascade(fuse_pos)
                fuse_neg = self.redection_layers.re_cascade(fuse_neg)

                fuse_pos = fuse_pos.view(n_pos,-1)#b,1024*3*3
                fuse_neg = fuse_neg.view(n_neg,-1)
                
                feat_pos = self.fuse_layers(fuse_pos)
                feat_neg = self.fuse_layers(fuse_neg)
                
                match_posfeat = self.match_branches[k](feat_pos)
                match_pos = match_posfeat.view(match_posfeat.size(0), -1)
                match_negfeat = self.match_branches[k](feat_neg)
                match_neg = match_negfeat.view(match_negfeat.size(0), -1)
                return match_pos,match_neg
            else:
                fuse_pos = self.redection_layers.re_down(feat)
                fuse_pos = self.redection_layers.re_cascade(fuse_pos)
                fuse_pos = fuse_pos.view(fuse_pos.shape[0],-1)#b,1024*3*3
                feat_pos = self.fuse_layers(fuse_pos)
                match_posfeat = self.match_branches[k](feat_pos)
                match_pos = match_posfeat.view(match_posfeat.size(0), -1)
                return match_pos

     
    def load_model_tracking(self, model_path):
        states = torch.load(model_path)
        RGB_layers = states['RGB_layers']
        self.RGB_layers.load_state_dict(RGB_layers)

        T_layers = states['T_layers']
        self.T_layers.load_state_dict(T_layers)

        redection_layers= states['redection_layers']
        self.redection_layers.load_state_dict(redection_layers)

        linear_layers = states['linear_layers']
        self.linear_layers.load_state_dict(linear_layers)

        shared_linear = states['shared_linear']
        self.shared_linear.load_state_dict(shared_linear)

        fuse_layers = states['fuse_layers']
        self.fuse_layers.load_state_dict(fuse_layers)

        print('load pth Done!')

    def restore_model(self, model_path):
        states = torch.load(model_path)

        RGB_layers = states['RGB_layers']
        self.RGB_layers.load_state_dict(RGB_layers)

        T_layers = states['T_layers']
        self.T_layers.load_state_dict(T_layers)

        redection_layers= states['redection_layers']
        self.redection_layers.load_state_dict(redection_layers)

        linear_layers = states['linear_layers']
        self.linear_layers.load_state_dict(linear_layers)

        shared_linear = states['shared_linear']
        self.shared_linear.load_state_dict(shared_linear)

        fuse_layers = states['fuse_layers']
        self.fuse_layers.load_state_dict(fuse_layers)

        cls_branches = states['cls_branches']
        self.cls_branches.load_state_dict(cls_branches)

        match_branches = states['match_branches']
        self.match_branches.load_state_dict(match_branches)
        print('restore pth Done!')

    def load_model_part(self, model_path):
        states = torch.load(model_path)
        states = states['shared_layers']
        conv1_w = states['conv1.0.weight']
        conv1_b = states['conv1.0.bias']
        conv2_w = states['conv2.0.weight']
        conv2_b = states['conv2.0.bias']
        conv3_w = states['conv3.0.weight']
        conv3_b = states['conv3.0.bias']

        self.RGB_layers[0][0].weight.data = conv1_w
        self.RGB_layers[0][0].bias.data = conv1_b
        self.RGB_layers[1][0].weight.data = conv2_w
        self.RGB_layers[1][0].bias.data = conv2_b
        self.RGB_layers[2][0].weight.data = conv3_w
        self.RGB_layers[2][0].bias.data = conv3_b

        self.T_layers[0][0].weight.data = conv1_w
        self.T_layers[0][0].bias.data = conv1_b
        self.T_layers[1][0].weight.data = conv2_w
        self.T_layers[1][0].bias.data = conv2_b
        self.T_layers[2][0].weight.data = conv3_w
        self.T_layers[2][0].bias.data = conv3_b

        print('load rt_pth finish!')

#######################   loss  function  ###############################################
class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:,1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:,0]
        loss = (pos_loss.sum() + neg_loss.sum()) / (pos_loss.size(0) + neg_loss.size(0))
        return loss

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss

#####################################################################################
class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]  
        TP = (topk < pos_score.size(0)).float().sum()
        prec = TP / (pos_score.size(0)+1e-8)
        return prec.item()

def get_cid_prec(q,k,neg):
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    neg_embbding = F.normalize(neg, dim=1)    

    l_pos = torch.mm(q, k.transpose(1, 0))
    l_neg = torch.mm(q, neg_embbding.transpose(1, 0))
    out = torch.cat((l_pos, l_neg), dim=-1)
    prec_batch = np.zeros(out.shape[0])
    
    for bb in range(out.shape[0]):
        topk = torch.topk(out[bb], k.shape[0])[1]  
        prec = (topk <  k.shape[0]).float().sum() / (k.shape[0]+1e-8)
        prec_batch[bb] = prec.item() 
    
    return prec_batch

if __name__ == '__main__':
    model = MDNet()
    model.set_learnable_params(['conv',])
    params = model.get_learnable_params()
    for k, p in params.items():
        print(k,p.shape)
