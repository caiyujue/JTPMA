import os
import sys
import pickle
import time
from skimage import color

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import datetime
import visdom
import pandas as pd

from data_prov import *
from model_vit import *
from options import *


os.environ["CUDA_VISIBLE_DEVICES"] ="0"
torch.backends.cudnn.benchmark = False

class Memory(nn.Module):
    def __init__(self, device='cuda', size=50, opts=opts,
                 emb_dim=512, emb_dim1=512, m=0.999, T=0.07,):
        super(Memory, self).__init__()
        self.pos_num = opts['batch_frames']*opts['batch_pos']
        self.neg_num = opts['batch_frames']*opts['batch_neg']
        self.emb_dim = emb_dim
        self.emb_dim1 = emb_dim1
        self.K = opts['batch_frames']*opts['batch_neg']

        ##### 类间样本保存 #########
        self.r_pos_memory = np.zeros((size,opts['batch_frames'],opts['batch_pos'],emb_dim))
        self.r_neg_memory = np.zeros((size,opts['batch_frames'],opts['batch_neg'],emb_dim))
        self.t_pos_memory = np.zeros((size,opts['batch_frames'],opts['batch_pos'],emb_dim))
        self.t_neg_memory = np.zeros((size,opts['batch_frames'],opts['batch_neg'],emb_dim))
        self.frame_id_save = np.zeros((size,opts['batch_frames']))
        
        self.m = m
        self.device = device
        self.size = size
      
    def check_num(self,embbding,total_num):
        if embbding.shape[0] < total_num:
            diff = total_num-embbding.shape[0]
            embbding=torch.cat((embbding,embbding[-diff:]))
        if embbding.shape[0] > total_num:
            embbding = embbding[0:total_num]
        # embbding = F.normalize(embbding, dim=1)
        return embbding

    @torch.no_grad()
    def return_self(self, index, modal='rgb',mode='inter'):
        if mode =='fuse':
            self_hard_samples = torch.Tensor(self.hard_memory[index])
            self_samples = self_hard_samples
        elif mode =='inter':
            if modal=='rgb':
                self_pos_samples = torch.Tensor(self.r_pos_memory[index])
                self_neg_samples = torch.Tensor(self.r_neg_memory[index])
            elif modal=='t':
                self_pos_samples = torch.Tensor(self.t_pos_memory[index])
                self_neg_samples = torch.Tensor(self.t_neg_memory[index])
            self_samples = self_pos_samples.view(-1,self_pos_samples.shape[-1])
        elif mode =='intra':
            if modal=='rgb':
                self_pos_samples = torch.Tensor(self.r_pos_memory1[index])
                self_neg_samples = torch.Tensor(self.r_neg_memory1[index])
            elif modal=='t':
                self_pos_samples = torch.Tensor(self.t_pos_memory1[index])
                self_neg_samples = torch.Tensor(self.t_neg_memory1[index])
            self_samples = self_pos_samples.view(-1,self_pos_samples.shape[-1])#torch.cat((random_pos_samples,random_neg_samples),dim=1)
        return self_samples
    
    @torch.no_grad()
    def return_random(self, size, index, modal='rgbt',mode='intra'):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        allowed = [x for x in range(index[0])] + [x for x in range(index[0] + 1, self.size)]
        index = random.sample(allowed, size)
        
        if mode =='fuse':
            random_hard_samples = torch.Tensor(self.hard_memory[index])
            
            random_samples = random_hard_samples.view(-1,self.emb_dim)#
            #random_samples = torch.cat((random_pos_samples,random_neg_samples),dim=1).view(-1,self.emb_dim)   
        elif mode =='inter':
            if modal=='rgb':
                random_pos_samples = torch.Tensor(self.r_pos_memory[index])
                random_neg_samples = torch.Tensor(self.r_neg_memory[index])
            elif modal=='t':
                random_pos_samples = torch.Tensor(self.t_pos_memory[index])
                random_neg_samples = torch.Tensor(self.t_neg_memory[index])
            random_samples = random_pos_samples.view(-1,self.emb_dim)
            #random_samples = random_neg_samples.view(-1,self.emb_dim)
            #random_samples = torch.cat((random_pos_samples,random_neg_samples),dim=2).view(-1,self.emb_dim)  
        elif mode =='intra':
            if modal=='rgb':
                random_pos_samples = torch.Tensor(self.r_pos_memory1[index])
                random_neg_samples = torch.Tensor(self.r_neg_memory1[index])
            elif modal=='t':
                random_pos_samples = torch.Tensor(self.t_pos_memory1[index])
                random_neg_samples = torch.Tensor(self.t_neg_memory1[index])
            #random_samples = random_pos_samples.view(-1,self.emb_dim1)#torch.cat((random_pos_samples,random_neg_samples),dim=1)    
            random_samples = torch.cat((random_pos_samples,random_neg_samples),dim=1).view(-1,self.emb_dim1)  
        return random_samples,index

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, index, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[index])
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[index,ptr:ptr + batch_size, : ] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[index] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue(self, pos_keys, neg_keys, index, modal='rgb',mode='intra'): #'inter'  # 出队和入队操作
        pos_keys = self.check_num(pos_keys,self.pos_num)
        neg_keys = self.check_num(neg_keys,self.neg_num)

        if mode =='fuse':
            self.hard_memory[index] = pos_keys.view(-1,opts['batch_pos'],self.emb_dim).cpu().numpy()

        elif mode =='intra':#模态内
            if modal=='rgb':
                self.r_pos_memory1[index] = pos_keys.view(-1,opts['batch_pos'],self.emb_dim1).cpu().numpy()
                self.r_neg_memory1[index] = neg_keys.view(-1,opts['batch_neg'],self.emb_dim1).cpu().numpy()
            elif modal=='t':
                self.t_pos_memory1[index] = pos_keys.view(-1,opts['batch_pos'],self.emb_dim1).cpu().numpy()
                self.t_neg_memory1[index] = neg_keys.view(-1,opts['batch_neg'],self.emb_dim1).cpu().numpy()
        elif mode =='inter':#模态间
            if modal=='rgb':
                self.r_pos_memory[index] = pos_keys.view(-1,opts['batch_pos'],self.emb_dim).cpu().numpy()
                self.r_neg_memory[index] = neg_keys.view(-1,opts['batch_neg'],self.emb_dim).cpu().numpy()
            elif modal=='t':
                self.t_pos_memory[index] = pos_keys.view(-1,opts['batch_pos'],self.emb_dim).cpu().numpy()
                self.t_neg_memory[index] = neg_keys.view(-1,opts['batch_neg'],self.emb_dim).cpu().numpy()

    def forward(self,pos_save1,pos_save2,
                neg_save1,neg_save2,
                index,frame_id,intilize=True):
        # update memory
        def Dynamic_update(x,y,ind,fram_id=None):
            x_emb_dim = x.shape[-1]
            y_emb_dim = y.shape[-1]
            if x.dim() == 2:
                x = x.view(-1,opts['batch_pos'],x_emb_dim)#b,32,512
            if y.dim() == 2:
                y = y.view(-1,opts['batch_pos'],y_emb_dim)#b,32,512
            y=y.cuda()
            x=x.cuda()

            for j in range(y.shape[0]):
                anchor = y[j]
                sim_list = np.zeros(x.shape[0])
                for i in range(x.shape[0]):
                    pos_sim = torch.mm(anchor, x[i].transpose(1, 0))
                    sim_list[i] = pos_sim.mean().item()
                sim = torch.Tensor(sim_list)
                topk = torch.topk(sim, 1)[1]
                x[topk]=anchor.unsqueeze(0) 
                # if fram_id is not None:
                #     self.frame_id_save[ind][topk] = fram_id[j]
            return x

        if intilize:
            pos_save1 = pos_save1.view(-1,self.emb_dim)
            pos_save2 = pos_save2.view(-1,self.emb_dim)
            #self.frame_id_save[index] = np.array(frame_id)
            
        else:
            # pos_save1 = Dynamic_update(self.return_self(index,modal='rgb',mode='inter'),pos_save1,index,frame_id)
            # #pos_save1 = Dynamic_update1(self.return_self(index,modal='rgb',mode='inter'),pos_save1)
            # pos_save2 = Dynamic_update(self.return_self(index,modal='t',mode='inter'),pos_save2,index,fram_id=None)

            pos_save1 = pos_save1.view(-1,self.emb_dim)
            pos_save2 = pos_save2.view(-1,self.emb_dim)

        self._dequeue_and_enqueue(pos_save1,neg_save1,index,modal='rgb',mode='inter')
        self._dequeue_and_enqueue(pos_save2,neg_save2,index,modal='t',mode='inter')
    
#########################################################################################################

def set_optimizer(model, lr_base, layers = opts['ft_layers'], 
                  lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    
    optimizer = optim.SGD([ 
                            # VGG-M##                           
                            {'params': model.RGB_layers.parameters(),'lr': 1e-4},
                            #{'params': model.T_layers.parameters(),'lr': 1e-4},
                            
                            ## For semantic match #####
                            {'params': model.redection_layers.parameters(),'lr': 1e-4},
                            {'params': model.fuse_layers.parameters(),'lr': 1e-3},
                            {'params': model.match_branches.parameters(),'lr': 1e-3},

                            ## For cls ##
                            {'params': model.linear_layers.parameters(),'lr': 1e-3},
                            {'params': model.shared_linear.parameters(),'lr': 1e-3},
                            {'params': model.cls_branches.parameters(),'lr': 1e-3},

                             ], lr=1e-4,momentum=momentum, weight_decay=w_decay)
    return optimizer

def get_contrastive_loss1(anchor_embbding,pos_embbding,
                         neg_embbding,criterion_ct,T = 0.07,weight = None):
    '''
    anchor_embbding:  8,512
    pos_embbding:  8*32,512
    neg_embbding:  8*96,512
    '''
    q = anchor_embbding
    q = F.normalize(q, dim=1)
    k = pos_embbding
    k = F.normalize(k, dim=1)
    anchor_count = q.shape[0]
    pos_count = k.shape[0]
    neg_embbding = F.normalize(neg_embbding, dim=1)

    # pos logit
    l_pos = torch.mm(k, q.transpose(1, 0))
    l_pos = l_pos.transpose(0, 1).unsqueeze(-1)
    # neg logit
    l_neg = torch.mm(neg_embbding, q.transpose(1, 0))
    l_neg = l_neg.transpose(0, 1).unsqueeze(1).repeat(1,pos_count,1)

    out = torch.cat((l_pos, l_neg), dim=-1)#8,256,768+1 --> 8,256,769
    out = torch.div(out, T)

    out1 = out.clone()
    out1 = out1.mean(dim=0).view(-1,out1.shape[-1]).contiguous()
    if weight is not None:
        loss_con = 0.
        for i in range(out.shape[0]):# 8 
            bsz = out[i].shape[0]#8*32=256  pos_num
            x = out[i].squeeze()
            label = torch.zeros([bsz]).cuda().long()
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(x, label) * weight[i][0]
            loss_con = loss_con + loss.mean()
    else:
        out = out.view(-1,out.shape[-1]).contiguous()
        bsz = out.shape[0]#8*32=256  pos_num
        label = torch.zeros([bsz]).cuda().long()
        criterion_ct = nn.CrossEntropyLoss()
        loss_con = criterion_ct(out, label) 
        #print("613",i,loss.mean().item())
    #loss_con = criterion_ct(out)
    return loss_con,out1[:,0:]#,losz1_id[sample_num:]

def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1
    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )
    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )
    # remove the center pixels dim=2维度，共ｓｉｚｅ张图，第ｉ张图是原ｘ上元素的８邻域的第ｉ个元素
    # 中间那张刚好就是原图
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)
    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images

    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks, kernel_size=kernel_size,
        dilation=dilation
    )
    
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]#邻域内有１，即选择框内的像素
    return similarity * unfolded_weights

class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
        return img_lab

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def compute_project_term(mask_scores, gt_bitmasks):
    assert mask_scores.dim() == 4
    fg_mask_score = mask_scores[:,1:2,...]
    ly = gt_bitmasks.max(dim=2, keepdim=True)[0]
    lx = gt_bitmasks.max(dim=3, keepdim=True)[0]
    
    mask_losses_y = dice_coefficient(
        fg_mask_score.max(dim=2, keepdim=True)[0],
        ly
    )
    mask_losses_x = dice_coefficient(
        fg_mask_score.max(dim=3, keepdim=True)[0],
        lx
    )
    dice_loss = (mask_losses_x + mask_losses_y).mean()
    return dice_loss

def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4
    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)
    
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold
    # 比较tensor1 和tensor2 中的元素，返回较大的那个值
    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)) + max_
    # loss = -log(prob)
    return -log_same_prob[:, 0]#.squeeze(1)

def compute_pairwise_term1(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    loss_fg = compute_pairwise_term(mask_logits[:,1:2,:,:], pairwise_size, pairwise_dilation)
    loss_bg = compute_pairwise_term(mask_logits[:,0:1,:,:], pairwise_size, pairwise_dilation)
    
    return loss_fg,loss_bg

def draw_axis(ax, img, title, show_minmax=False):
    ax.imshow(img)
    if show_minmax:
        minval_, maxval_, _, _ = cv2.minMaxLoc(img)
        title = '%s \n min=%.2f max=%.2f' % (title, minval_, maxval_)
    ax.set_title(title, fontsize=9)

def save_debug(epoch,r_template,r_search,soft_r,k,k1):

    dir_path = '/home/caiyujue/MANet/workspace/images'
    soft_r[0] = F.interpolate(soft_r[0], size=(r_search[k1].shape[2:]), mode='bilinear', align_corners=True)
    soft_r[1] = F.interpolate(soft_r[1], size=(r_search[k1].shape[2:]), mode='bilinear', align_corners=True)
    soft_r[2] = F.interpolate(soft_r[2], size=(r_search[k1].shape[2:]), mode='bilinear', align_corners=True)

    template1 = np.asarray(r_template[0].permute(1,2,0).cpu())
    search1 = np.asarray(r_search[k1][0].permute(1,2,0).cpu())
    search1 = search1 + 128.

    search4 = np.asarray(soft_r[0][0,1:2,...].permute(1,2,0).detach().cpu())
    search5 = np.asarray(soft_r[1][0,1:2,...].permute(1,2,0).detach().cpu())
    search6 = np.asarray(soft_r[2][0,1:2,...].permute(1,2,0).detach().cpu())

    search41 = np.asarray((soft_r[0][0,1:2,...]>0.65).float().permute(1,2,0).detach().cpu())
    search51 = np.asarray((soft_r[1][0,1:2,...]>0.65).float().permute(1,2,0).detach().cpu())
    search61 = np.asarray((soft_r[2][0,1:2,...]>0.65).float().permute(1,2,0).detach().cpu())

    mask = (search41)[:,:,0]
    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
    frame = cv2.addWeighted(search1, 0.77, mask, 0.23, -1,dtype=cv2.CV_8U)
    mask = (search51)[:,:,0]
    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
    frame1 = cv2.addWeighted(search1, 0.77, mask, 0.23, -1,dtype=cv2.CV_8U)
    mask = (search61)[:,:,0]
    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
    frame2 = cv2.addWeighted(search1, 0.77, mask, 0.23, -1,dtype=cv2.CV_8U)
    
    t = (template1).astype(np.uint8)
    s = (search1).astype(np.uint8)
       
    s3 = (search4)
    s4 = (search5)
    s5 = (search6)

    f, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax12)) = plt.subplots(3, 4, figsize=(12, 9))
    draw_axis(ax1, t, 'template')
    draw_axis(ax2, s, 'r_searchs')
    draw_axis(ax3, t, 'r_corr')
    draw_axis(ax4, s, 'rt_sim')

    draw_axis(ax5, s3, 'r_sim', show_minmax=True)
    draw_axis(ax6, s4, 't_sim', show_minmax=True)
    draw_axis(ax7, s5, 'rt_sim1', show_minmax=True)
   # draw_axis(ax8, frame, 'rt_feat1')

    draw_axis(ax9, frame, 'r_sim')
    draw_axis(ax10, frame1, 't_sim')
    draw_axis(ax11, frame2, 'rt_sim1')
   # draw_axis(ax12, s7, 'rt_feat1', show_minmax=True)

    dir_path = '/home/caiyujue/MANet/workspace/images'
    save_path = os.path.join(dir_path, '%03d-%04d.png' % (epoch,k+k1))
    plt.savefig(save_path)
    plt.close(f)

def save_debug2(epoch,r_template,r_search,k):

    dir_path = '/home/caiyujue/MANet/workspace/images'
    template1 = np.asarray(r_template[0].permute(1,2,0).cpu())
    search1 = np.asarray(r_search[1][0].permute(1,2,0).cpu())
    search2 = np.asarray(r_search[2][0].permute(1,2,0).cpu())
    search3 = np.asarray(r_search[3][0].permute(1,2,0).cpu())
    search4 = np.asarray(r_search[4][0].permute(1,2,0).cpu())
    search5 = np.asarray(r_search[5][0].permute(1,2,0).cpu())
    search6 = np.asarray(r_search[6][0].permute(1,2,0).cpu())
    search7 = np.asarray(r_search[7][0].permute(1,2,0).cpu())
    search8 = np.asarray(r_search[8][0].permute(1,2,0).cpu())

    search1 = search1 + 128.
    search2 = search2 + 128.
    search3 = search3 + 128.
    search4 = search4 + 128.
    search5 = search5 + 128.
    search6 = search6 + 128.
    search7 = search7 + 128.
    search8 = search8 + 128.

    t = (template1).astype(np.uint8)
    s = (search1).astype(np.uint8)
    s1 = (search2).astype(np.uint8)
    s2 = (search3).astype(np.uint8)
    s3 = (search4).astype(np.uint8)
    s4 = (search5).astype(np.uint8)
    s5 = (search6).astype(np.uint8)
    s6 = (search7).astype(np.uint8)
    s7 = (search8).astype(np.uint8)

    f, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3, 3, figsize=(9, 9))
    draw_axis(ax1, t, 'template')
    draw_axis(ax2, s,'r_searchs')
    draw_axis(ax3, s1, 't_search')
    draw_axis(ax4, s2, 'r_search1')
    draw_axis(ax5, s3, 't1')
    draw_axis(ax6, s4, 'fuse_0')
    draw_axis(ax7, s5, 'template')
    draw_axis(ax8, s6,'r_searchs')
    draw_axis(ax9, s7, 't_search')

    dir_path = '/home/caiyujue/MANet/workspace/images'
    save_path = os.path.join(dir_path, '%03d-%04d.png' % (epoch,k))
    plt.savefig(save_path)
    plt.close(f)

def save_debug3(epoch,r_template,r_search,k):

    dir_path = '/home/caiyujue/MANet/workspace/images'
    template1 = np.asarray(r_template[0].permute(1,2,0).cpu())
    search1 = np.asarray(r_search[0].permute(1,2,0).cpu())
    search2 = np.asarray(r_search[1].permute(1,2,0).cpu())
    search3 = np.asarray(r_search[2].permute(1,2,0).cpu())
    search4 = np.asarray(r_search[3].permute(1,2,0).cpu())
    search5 = np.asarray(r_search[4].permute(1,2,0).cpu())
    search6 = np.asarray(r_search[5].permute(1,2,0).cpu())
    search7 = np.asarray(r_search[6].permute(1,2,0).cpu())
    search8 = np.asarray(r_search[7].permute(1,2,0).cpu())

    t = (template1).astype(np.uint8)
    s = (search1)
    s1 = (search2)
    s2 = (search3)
    s3 = (search4)
    s4 = (search5)
    s5 = (search6)
    s6 = (search7)
    s7 = (search8)

    f, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3, 3, figsize=(9, 9))
    draw_axis(ax1, t, 'template')
    draw_axis(ax2, s,'r_searchs')
    draw_axis(ax3, s1, 't_search')
    draw_axis(ax4, s2, 'r_search1')
    draw_axis(ax5, s3, 't1')
    draw_axis(ax6, s4, 'fuse_0')
    draw_axis(ax7, s5, 'template')
    draw_axis(ax8, s6,'r_searchs')
    draw_axis(ax9, s7, 't_search')

    dir_path = '/home/caiyujue/MANet/workspace/images'
    save_path = os.path.join(dir_path, '%03d-%04d.png' % (epoch,k))
    plt.savefig(save_path)
    plt.close(f)

##############################################################################
#create the vis,with initialized 0
def create_vis_plot(viz,_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, len(_legend))).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )
#update visdom plot
def update_vis_plot(viz, iteration,l1,l2,l3,l4,l5,l6,window,update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1,6)).cpu() * iteration,
        Y=torch.Tensor([l1,l2,l3,l4,l5,l6]).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )

def update_vis_plot1(viz, iteration,l1,l2,l3,window,update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1,3)).cpu() * iteration,
        Y=torch.Tensor([l1,l2,l3]).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )

def update_vis_plot2(viz, iteration,l1,l2,l3,l4,window,update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1,4)).cpu() * iteration,
        Y=torch.Tensor([l1,l2,l3,l4]).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )

def train_mdnet(dataset='rgbt234'):

    #torch.backends.cudnn.benchmark = True
    #********************************************set dataset path ********************************************
    #********************************************set seq list .pkl file path  ********************************
    if dataset == 'rgbt234':
        img_home = "/home/caiyujue/baidunetdiskdownload/RGBT234/RGB_T234/"
        data_path1 = '/home/caiyujue/MANet/data/rgbt234.pkl'   
    elif dataset == 'gtot':
        img_home = "/home/caiyujue/baidunetdiskdownload/GTOT/"
        data_path1 = '/home/caiyujue/MANet/data/gtot.pkl'
        img_home1 = "/home/caiyujue/baidunetdiskdownload/GTOT/"
    elif dataset == 'vot2019':
        img_home = "/home/caiyujue/d3s/dataset/vot2019rgbt/"
        img_home1 = "/home/caiyujue/baidunetdiskdownload/VOT_OCC/"
        data_path1 = '/home/caiyujue/MANet/data/vot2019.pkl'

    #*********************************************************************************************************
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    ## Init dataset ##
    with open(data_path1, 'rb') as fp1:
        data1 = pickle.load(fp1)
   
    K1 = len(data1)
    dataset1 = [None]*K1
    dataset2 = [None]*K1
    seqnames = []
    seqlength= []
    for k, ([seqname, seq]) in enumerate(sorted(data1.items())):
        print(k,seqname)
        seqnames.append(seqname)
        rgbimg_list = seq['rgb_images']
        timg_list = seq['t_images']      
        gt = seq['gt']
        seqlength.append(len(rgbimg_list))
        
        if dataset == 'rgbt234':
            rgbimg_dir = os.path.join(img_home,seqname+'/visible')
            timg_dir = os.path.join(img_home,seqname+'/infrared')
        elif dataset == 'gtot':
            rgbimg_dir = os.path.join(img_home,seqname+'/v')
            timg_dir = os.path.join(img_home,seqname+'/i')
        elif dataset == 'vot2019':
            rgbimg_dir = os.path.join(img_home,seqname+'/color')
            timg_dir = os.path.join(img_home,seqname+'/ir')

        print("read_data_from",rgbimg_dir,len(rgbimg_list))
        print("read_data_from",timg_dir,len(timg_list))

        dataset1[k] = RegionDataset(rgbimg_dir, rgbimg_list, timg_dir, timg_list ,
                                    gt, opts)

    seqlength = torch.Tensor(seqlength)

    sample_gap = 20
    batch_sz = 8
    seqlength = ((seqlength/sample_gap)/batch_sz).ceil()
    test_num = []
    for seqi,seqlen in enumerate(seqlength):
        if seqlen <=1:
            test_num.append(batch_sz)
        else:
            test_num.append((seqlen*batch_sz).int().item())
    
    for k, ([seqname, seq]) in enumerate(sorted(data1.items())):
        rgbimg_list = seq['rgb_images']
        timg_list = seq['t_images']
        gt = seq['gt']
        sample_num = test_num[k]
        if dataset == 'rgbt234':
            rgbimg_dir = os.path.join(img_home,seqname+'/visible')
            timg_dir = os.path.join(img_home,seqname+'/infrared')
        elif dataset == 'gtot':
            rgbimg_dir = os.path.join(img_home,seqname+'/v')
            timg_dir = os.path.join(img_home,seqname+'/i')
        elif dataset == 'vot2019':
            rgbimg_dir = os.path.join(img_home,seqname+'/color')
            timg_dir = os.path.join(img_home,seqname+'/ir')

        dataset2[k] = RegionDataset1(rgbimg_dir,rgbimg_list,timg_dir,timg_list,gt,sample_num,opts)

    ## Init model ## 
    model = MDNet(model_path='/home/caiyujue/JTPMA/models/MANet-2IC.pth',
                  model_path1='/home/caiyujue/JTPMA/rt-mdnet.pth',#rt-mdnet.pth  #ADRNet_GTOT.pth
                  load_pretrain=True,K=K1)
    memory = Memory(size=K1,opts=opts)

    if opts['use_gpu']:
        model = model.cuda()
        
    ##### 设置可训练的参数############
    model.set_learnable_params(opts['ft_layers'])
   
    if opts['adaptive_align']:
        align_h = model.roi_align_model.pooled_height
        align_w = model.roi_align_model.pooled_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = PrRoIPool2D(align_h, align_w, spatial_s)
        
    model.train()
    ########### Init criterion and optimizer #################
    criterion = BinaryLoss() 
    evaluator = Precision()
    interDomainCriterion = nn.CrossEntropyLoss(reduction='mean')

    ##########################################################
    optimizer_cls = set_optimizer(model,opts['lr'],opts['ft_layers'])
    #########################################################
    ### python -m visdom.server
    viz = visdom.Visdom()
    vis_title = 'SGD.PyTorch on ' + dataset
    vis_legend = ['Cls Loss','Inter Loss','con_loss','match_loss','rank_loss','Total Loss']
    vis_legend1 = ['prec','prec_rsh','prec_tsh']
    vis_legend2 = ['prec_match','prec_r2tsh','prec_t2rsh','prec_fuse']
    
    iter_plot = create_vis_plot(viz,'Iteration', 'Loss', vis_title, vis_legend)
    epoch_plot = create_vis_plot(viz,'Epoch', 'Loss', vis_title, vis_legend)
    epoch_plot1 = create_vis_plot(viz,'Epoch', 'Prec', vis_title, vis_legend1)
    epoch_plot2 = create_vis_plot(viz,'Epoch', 'Prec1', vis_title, vis_legend2)
    ##########################################################
    
    best_prec = 0.
    batch_cur_idx = 0
    val_precs = []
    prec_save = np.zeros([opts['n_cycles'],K1])
    prec_save1 = np.zeros([opts['n_cycles'],K1])
    prec_save2 = np.zeros([opts['n_cycles'],K1])

    ##########################################################
    bottom_pixels_removed = 10
    pairwise_size = 3
    pairwise_dilation = 2
    pairwise_color_thresh = 0.3

    for i in range(opts['n_cycles']):
        print("==== Start Cycle %d ====" % (i))
        k_list = np.random.permutation(K1)
        
        # for instance classifier
        prec_r = np.zeros(K1)
        prec_t = np.zeros(K1)
        prec = np.zeros(K1)

        prec_match= np.zeros(K1)
        
        # for corss-modal 
        prec_r2tsh= np.zeros(K1)
        prec_t2rsh= np.zeros(K1)
        prec_r2t= np.zeros(K1)
        prec_t2r= np.zeros(K1)
        prec_fuse= np.zeros(K1)
        # loss show
        epoch_loss=np.zeros(K1)
        epoch_clsloss=np.zeros(K1)
        epoch_interloss=np.zeros(K1)
        epoch_matchloss=np.zeros(K1)
        epoch_crossloss=np.zeros(K1)
        epoch_segloss=np.zeros(K1)

        for j,k in enumerate(k_list):
            print(j,k)
            #k=0
            tic = time.time()
            rgb_scenes, t_scenes,mask_labels, pos_rois, neg_rois,idx,anchor_rois,pos_ious, neg_ious,\
                anchor_boxes,init_RGB_targets,init_T_targets = dataset1[k].next(opts)
            batch_size = len(rgb_scenes)
            print("736",np.array(idx))
            
            init_RGB_targets = torch.stack(init_RGB_targets,dim=0).squeeze(1).cuda()
            print("749",init_RGB_targets.shape)
            init_T_targets = torch.stack(init_T_targets,dim=0).squeeze(1).cuda()
            ## model.get_target_feat(init_RGB_targets, init_T_targets)

            x=1
            assert x==0

            def get_roi_feats(cur_feat_map,anchor_roi,pos_rois,neg_rois):
                cur_anchor_feat = model.roi_align_model(cur_feat_map, anchor_roi)
                cur_anchor_feat = cur_anchor_feat.view(cur_anchor_feat.size(0), -1)
                
                cur_pos_feats = model.roi_align_model(cur_feat_map, pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                
                cur_neg_feats = model.roi_align_model(cur_feat_map, neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)
                return cur_anchor_feat,cur_pos_feats,cur_neg_feats
            
            ####################### support image the initial frame #############################################
            sidx=0
            if opts['use_gpu'] & torch.cuda.is_available():
                cur_rgbscene = rgb_scenes[sidx].cuda()
                cur_tscene = t_scenes[sidx].cuda()
                cur_anchor_roi = anchor_rois[sidx].cuda()
            
            # only need feature map
            r_map,t_map,cls_map,_,_,_= model(cur_rgbscene,cur_tscene,k=k,out_layer='conv4')
            
            # crop target feat
            init_anchor = model.roi_align_model(cls_map, cur_anchor_roi)
            init_r_anchor = model.roi_align_model(r_map, cur_anchor_roi)
            init_t_anchor = model.roi_align_model(t_map, cur_anchor_roi)

            model.get_supp_prototype(r_map,t_map,mask_labels[sidx].cuda())

            # get support seg
            init_sim_rt,init_sim_r,init_sim_t = model(r_map,t_map,\
                                                      k=k,in_layer='conv3',out_layer='sim')
            
            ################################################################################
            # detail: first compute sim,then interpolate it
            if opts['seg_task']:
                color_transfer = RGB2Lab()
                lab_scenes = []
                col_sim = []
                original_image_masks = [torch.ones_like(x, dtype=torch.float32) for x in rgb_scenes]
                
                #cur_rgbscene = F.interpolate(cur_rgbscene, size=(cls_map.shape[2:]), mode='bilinear', align_corners=True)
                images_lab = color_transfer((cur_rgbscene[0]+128.).permute(1, 2, 0).cpu().numpy())
                images_lab = torch.as_tensor(images_lab, device=rgb_scenes[sidx].device, dtype=torch.float32)
                images_lab = images_lab.permute(2, 0, 1)[None]
                lab_scenes.append(images_lab)
                
                #original_image_masks[sidx]= F.interpolate(original_image_masks[sidx], size=(cls_map.shape[2:]), mode='bilinear', align_corners=True)
                images_color_similarity = get_images_color_similarity(
                                            images_lab.cuda(), original_image_masks[sidx].cuda(),
                                            pairwise_size, pairwise_dilation)
                images_color_similarity = F.interpolate(images_color_similarity, size=(cls_map.shape[2:]), mode='bilinear', align_corners=True)
                
                images_color_similarity1 = get_images_color_similarity(
                                            (cur_tscene+128.), original_image_masks[sidx].cuda(),
                                            pairwise_size, pairwise_dilation)
                images_color_similarity1 = F.interpolate(images_color_similarity1, size=(cls_map.shape[2:]), mode='bilinear', align_corners=True)

                images_color_similarity2 = (images_color_similarity+images_color_similarity1)/2.
                # # images_color_similarity = images_color_similarity.squeeze(0).unsqueeze(1).unsqueeze(1)
                
                if opts['sim_mode']=='RGB':
                    col_sim.append(images_color_similarity)
                elif opts['sim_mode']=='T':
                    col_sim.append(images_color_similarity1)
                elif opts['sim_mode']=='fuse':
                    col_sim.append(images_color_similarity2)
            
                #################################################################################
                gt_bitmasks = F.interpolate(mask_labels[sidx], size=(init_sim_rt.shape[2:]), mode='bilinear', align_corners=True)
                gt_bitmasks = gt_bitmasks.cuda()

                loss_prj_term = compute_project_term(init_sim_rt, gt_bitmasks.cuda())

                pairwise_lossfg,pairwise_lossbg = compute_pairwise_term1(
                    init_sim_rt, pairwise_size,
                    pairwise_dilation
                )

                weights = ((col_sim[sidx] >= pairwise_color_thresh).float() * gt_bitmasks.float()).cuda()

                loss_pairwise1 = (pairwise_lossfg * weights).sum() / weights.sum().clamp(min=1.0)
                
                loss_supp_seg = (loss_prj_term+loss_pairwise1)
            else:
                loss_supp_seg = 0.
            
            ##############################################################################################

            def set_grl_lambda(cur_epoch,max_epoch,gamma=10):
                lambda_p = 2/(1+torch.exp(-torch.Tensor([gamma])*(cur_epoch/max_epoch)))-1
                return lambda_p.item()
            max_epoch = opts['n_cycles']*K1
            cur_epoch = (i)*K1+j
            cur_lambda = set_grl_lambda(cur_epoch,max_epoch)
            
            ########################################################################################
            sim_rts=[]
            sim_rs=[]
            sim_ts=[]
            
            map_rt=[]
            map_r1=[]
            map_t1=[]
            for sidx in range(1, batch_size):
                if opts['use_gpu'] & torch.cuda.is_available():
                    cur_rgbscene = rgb_scenes[sidx].cuda()
                    cur_tscene = t_scenes[sidx].cuda()
                    cur_anchor_roi = anchor_rois[sidx].cuda()
                    cur_pos_rois = pos_rois[sidx].cuda()
                    cur_neg_rois = neg_rois[sidx].cuda()
                    
                if cur_pos_rois.shape[0] == 0:
                    continue
                
                r_sp_map,t_sp_map,cls_map,sim_rt,sim_r,sim_t = model(cur_rgbscene,cur_tscene,\
                                                    init_anchor,init_r_anchor,init_t_anchor,\
                                                    k=k,out_layer='conv4')
                # print(len(sim_rt))
                # save_debug(i,init_RGB_targets[0:1],rgb_scenes,sim_rt,j+4,sidx)
                
                ############################################################
                if opts['seg_task']:
                    #cur_rgbscene = F.interpolate(cur_rgbscene, size=(sim_rt.shape[2:]), mode='bilinear', align_corners=True)
                    images_lab = color_transfer((cur_rgbscene[0]+128.).permute(1, 2, 0).cpu().numpy())
                    images_lab = torch.as_tensor(images_lab, device=rgb_scenes[sidx].device, dtype=torch.float32)
                    images_lab = images_lab.permute(2, 0, 1)[None]
                    lab_scenes.append(images_lab)
                    
                    #original_image_masks[sidx]= F.interpolate(original_image_masks[sidx], size=(cls_map.shape[2:]), mode='bilinear', align_corners=True)
                    images_color_similarity = get_images_color_similarity(
                                        images_lab.cuda(), original_image_masks[sidx].cuda(),
                                        pairwise_size, pairwise_dilation)
                    images_color_similarity = F.interpolate(images_color_similarity, size=(cls_map.shape[2:]), mode='bilinear', align_corners=True)
                    
                    images_color_similarity1 = get_images_color_similarity(
                                            (cur_tscene+128.), original_image_masks[sidx].cuda(),
                                            pairwise_size, pairwise_dilation)
                    images_color_similarity1 = F.interpolate(images_color_similarity1, size=(cls_map.shape[2:]), mode='bilinear', align_corners=True)

                    images_color_similarity2 = (images_color_similarity+images_color_similarity1)/2.
                    
                    if opts['sim_mode']=='RGB':
                        col_sim.append(images_color_similarity)
                    elif opts['sim_mode']=='T':
                        col_sim.append(images_color_similarity1)
                    elif opts['sim_mode']=='fuse':
                        col_sim.append(images_color_similarity2)

                ##############################################################
                sim_rts.append(sim_rt[-1])

                anchor_r = model.Weighted_GAP(r_sp_map,sim_rt[-1][:,1:2,...]).squeeze()
                anchor_t = model.Weighted_GAP(t_sp_map,sim_rt[-1][:,1:2,...]).squeeze()
                # mask avg pool
                map_r1.append(anchor_r)
                map_t1.append(anchor_t)
                
                r_sp_anchor,r_sp_pos,r_sp_neg = get_roi_feats(r_sp_map,cur_anchor_roi,cur_pos_rois,cur_neg_rois)
                t_sp_anchor,t_sp_pos,t_sp_neg = get_roi_feats(t_sp_map,cur_anchor_roi,cur_pos_rois,cur_neg_rois)
                cls_anchor,cls_pos,cls_neg = get_roi_feats(cls_map,cur_anchor_roi,cur_pos_rois,cur_neg_rois)

                if sidx == 1:
                    r_anchor_feat = r_sp_anchor
                    t_anchor_feat = t_sp_anchor
                    cls_anchor_feat = cls_anchor
                    
                    pos_feats1 = r_sp_pos
                    neg_feats1 = r_sp_neg
                    pos_feats2 = t_sp_pos
                    neg_feats2 = t_sp_neg
                    pos_feats3 = cls_pos
                    neg_feats3 = cls_neg
                else:
                    r_anchor_feat=torch.cat((r_anchor_feat,r_sp_anchor),dim=0)
                    t_anchor_feat=torch.cat((t_anchor_feat,t_sp_anchor),dim=0)
                    cls_anchor_feat=torch.cat((cls_anchor_feat,cls_anchor),dim=0)

                    pos_feats1=torch.cat((pos_feats1,r_sp_pos),dim=0)
                    neg_feats1=torch.cat((neg_feats1,r_sp_neg),dim=0)
                    pos_feats2=torch.cat((pos_feats2,t_sp_pos),dim=0)
                    neg_feats2=torch.cat((neg_feats2,t_sp_neg),dim=0)
                    pos_feats3=torch.cat((pos_feats3,cls_pos),dim=0)
                    neg_feats3=torch.cat((neg_feats3,cls_neg),dim=0)
            
            # feat for contrast and score for cls task
            r_pos_sh,t_pos_sh,fuse_pos,pos_score,pos_score1,pos_score2,\
                = model(pos_feats1,pos_feats2,pos_feats3,k=k,in_layer='fc4',out_layer='fc6')
            r_neg_sh,t_neg_sh,fuse_neg,neg_score,neg_score1,neg_score2,\
                = model(neg_feats1,neg_feats2,neg_feats3,k=k,in_layer='fc4',out_layer='fc6')
            r_anchor,t_anchor,fuse_anchor\
                = model(r_anchor_feat,t_anchor_feat,cls_anchor_feat,in_layer='fc4',out_layer='fc5')

            anchor_rs = torch.stack(map_r1,dim=0)
            anchor_ts = torch.stack(map_t1,dim=0)#8,1024

            #########################################################################################
            #########################################################################
            if opts['match_task']:
                cls_ff_feat = init_anchor.view(-1,1024,3,3)
                cls_det_pos = pos_feats3.view(-1,1024,3,3)
                cls_det_neg = neg_feats3.view(-1,1024,3,3)
                
                match_pos,match_neg = model(xR=cls_det_pos,xT=cls_det_neg,feat=cls_ff_feat,k=k,in_layer='match') 

                criterion = BinaryLoss() 
                loss_itm = criterion(match_pos,match_neg)
                prec_match[k] = evaluator(match_pos,match_neg)
            else:
                loss_itm = 0.
                prec_match[k] = 0.
        
            ################  利用对比学习的特征表示还有检索任务的ｐｒｅｃ来确定噪声帧######################
            fuse_con_prec = get_cid_prec(fuse_anchor,fuse_pos,fuse_neg)
            r_con_prec = get_cid_prec(r_anchor,r_pos_sh,r_neg_sh)
            t_con_prec = get_cid_prec(t_anchor,t_pos_sh,t_neg_sh)

            b1 = 0.7
            R_label1 = (r_con_prec<(r_con_prec.mean()*b1)).astype(float)
            T_label1 = (t_con_prec<(t_con_prec.mean()*b1)).astype(float)
            RT_label1 = (fuse_con_prec<(fuse_con_prec.mean()*b1)).astype(float)

            beta = 10.
            chall_label1 = 1-(RT_label1)*beta*cur_lambda
            weight_RT = F.softmax(torch.Tensor(chall_label1),dim=0).unsqueeze(dim=1)
            weight1 = ((weight_RT.repeat(1,128))/128).view(-1)
            weight11 = ((weight_RT.repeat(1,32))/32).view(-1)
            
            chall_label1 = 1-(R_label1)*beta*cur_lambda
            weight_R = F.softmax(torch.Tensor(chall_label1),dim=0).unsqueeze(dim=1)
            weight2 = ((weight_R.repeat(1,128))/128).view(-1)
            weight22 = ((weight_R.repeat(1,32))/32).view(-1)

            chall_label1 = 1-(T_label1)*beta*cur_lambda
            weight_T = F.softmax(torch.Tensor(chall_label1),dim=0).unsqueeze(dim=1)
            weight3 = ((weight_T.repeat(1,128))/128).view(-1)
            weight33 = ((weight_T.repeat(1,32))/32).view(-1)

            #####################   cls_loss for Semantically-discriminative   ###################
            criterion = BinaryLoss() 
            cls_loss_RGBT = criterion(pos_score, neg_score)
            cls_loss_RSH = criterion(pos_score1, neg_score1)
            cls_loss_TSH = criterion(pos_score2, neg_score2)

            prec[k] = evaluator(pos_score, neg_score)
            prec_r[k] = evaluator(pos_score1, neg_score1)
            prec_t[k] = evaluator(pos_score2, neg_score2)
            # print("863",prec[k],prec_r[k],prec_t[k])
            cls_loss = (cls_loss_RGBT + cls_loss_RSH + cls_loss_TSH)

            ##################################################################################################
            if opts['seg_task']:
                loss_seg = 0.
                for sidx in range(0, len(sim_rts)):
                    sim_1 = sim_rts[sidx]
                    gt_bitmasks = F.interpolate(mask_labels[sidx+1], size=(sim_rts[sidx].shape[2:]), mode='bilinear', align_corners=True)
                    gt_bitmasks = gt_bitmasks.cuda()
                    
                    loss_prj_term = compute_project_term(sim_1, gt_bitmasks.cuda())
                    pairwise_lossfg,_ = compute_pairwise_term1(
                        sim_1, pairwise_size,
                        pairwise_dilation
                    )
                    weights = ((col_sim[sidx+1] >= pairwise_color_thresh).float() * gt_bitmasks.float()).cuda()
                    
                    loss_pairwise1 = (pairwise_lossfg * weights).sum() / weights.sum().clamp(min=1.0)
                    
                    loss_seg+=(loss_prj_term+loss_pairwise1)
                
                loss_mean_seg = loss_seg/(batch_size-1)
                
                loss_total_seg = (loss_mean_seg+loss_supp_seg*0.2)
            else:
                loss_total_seg = 0.
            #################################################################################################
            
            ####################### inter seq classification ########################
            if opts['inter_task']:
                interclass_label = torch.zeros((pos_score.size(0))).long()#32
                
                if opts['use_gpu']:
                    interclass_label = interclass_label.cuda()
                
                total_interclass_score = pos_score[:,1].contiguous()
                total_interclass_score = total_interclass_score.view((pos_score.size(0),1))#32,1
                total_interclass_score1 = pos_score1[:,1].contiguous()
                total_interclass_score1 = total_interclass_score1.view((pos_score1.size(0),1))#32,1
                total_interclass_score2 = pos_score2[:,1].contiguous()
                total_interclass_score2 = total_interclass_score2.view((pos_score2.size(0),1))#32,1
                
                K_perm = np.random.permutation(K1)
                K_perm = K_perm[0:100]
                id_other = []
                id_other.append(k)
                for cidx in K_perm:
                    if k == cidx:
                        continue
                    else: 
                        id_other.append(cidx)
                        interclass_score,class_score1,class_score2 \
                        = model(pos_feats1,pos_feats2,pos_feats3, k=cidx,in_layer='fc4',out_layer='fc45')

                        total_interclass_score = torch.cat((total_interclass_score,
                                                    interclass_score[:,1].contiguous().view((interclass_score.size(0),1))),dim=1)
                        total_interclass_score1 = torch.cat((total_interclass_score1,
                                                    class_score1[:,1].contiguous().view((class_score1.size(0),1))),dim=1)
                        total_interclass_score2 = torch.cat((total_interclass_score2,
                                                    class_score2[:,1].contiguous().view((class_score2.size(0),1))),dim=1)
                        
                interDomainCriterion = nn.CrossEntropyLoss(reduction='mean')           
                interclass_loss = interDomainCriterion(total_interclass_score, interclass_label)#32,n
                interclass_loss1 = interDomainCriterion(total_interclass_score1, interclass_label)
                interclass_loss2 = interDomainCriterion(total_interclass_score2, interclass_label)
                
                class_loss = interclass_loss+interclass_loss1+interclass_loss2
                
                c_loss = cls_loss+0.1*class_loss
            else:
                c_loss = cls_loss
                class_loss = torch.Tensor([0.])

            ###################################################################
            def get_sh_prec(q,k,neg):
                q = F.normalize(q, dim=1)
                k = F.normalize(k, dim=1)
                neg_embbding = F.normalize(neg, dim=1)
                l_pos = torch.mm(q, k.transpose(1, 0))# 256,256
                l_neg = torch.mm(neg_embbding, q.transpose(1, 0))
                l_neg = l_neg.transpose(0, 1)
                out = torch.cat((l_pos, l_neg), dim=1)#256,1024
                prec_batch = np.zeros(out.shape[0])
                for bb in range(out.shape[0]):
                    topk = torch.topk(out[bb], k.shape[0])[1]  #top_ind
                    prec = (topk <  k.shape[0]).float().sum() / (k.shape[0]+1e-8)
                    prec_batch[bb]=prec.item()
                return prec_batch.mean()

            # #######################  cross-modal contrastive learning #################################
            if opts['con_task']:
                intilize = (i==0) #False#
                memory(r_pos_sh,t_pos_sh,r_neg_sh,t_neg_sh, k,idx,intilize)
                criterion_inter = NCESoftmaxLoss()
                alpha = 0.1  #用于控制类间对比学习的超参数
                T1 = 0.07

                r2t_pos = r_pos_sh  
                r2t_neg = r_neg_sh  
                t2r_pos = t_pos_sh
                t2r_neg = t_neg_sh
                
                if not intilize:
                    neg_other,seq_index = memory.return_random(memory.size-1, [k], mode ='inter',modal='rgb')
                    neg_other = neg_other.cuda()
                    r2t_neg = torch.cat((r_neg_sh,neg_other),dim=0)
                    rgb2t_loss1,out_in = get_contrastive_loss1(r_anchor,r2t_pos,r_neg_sh,
                                                    criterion_inter,T = T1, weight =weight_R)#, weight =weight_R

                    rgb2t_loss2,_ = get_contrastive_loss1(r_anchor,r2t_pos,neg_other,
                                                       criterion_inter,T = T1,)
                    sim_r2t = out_in
                    rgb2t_loss = rgb2t_loss1 + alpha*rgb2t_loss2
                else:
                    r2t_neg = r_neg_sh
                    rgb2t_loss,out_in = get_contrastive_loss1(r_anchor,r2t_pos,r2t_neg,
                                                              criterion_inter,T = T1, weight =weight_R)#, weight =weight_R
                    sim_r2t = out_in

                if not intilize:
                    neg_other,seq_index = memory.return_random(memory.size-1, [k], mode ='inter',modal='t')
                    neg_other = neg_other.cuda()#.view(memory.size-1,-1,512)
                    t2r_neg = torch.cat((t_neg_sh,neg_other),dim=0)
                    t2rgb_loss1,out_in = get_contrastive_loss1(t_anchor,t2r_pos,t_neg_sh,
                                                   criterion_inter,T = T1, weight =weight_T)#, weight =weight_T
                    t2rgb_loss2,_ = get_contrastive_loss1(t_anchor,t2r_pos,neg_other,
                                                   criterion_inter,T = T1)
                    sim_t2r = out_in
                    t2rgb_loss = t2rgb_loss1 + alpha*t2rgb_loss2
                else:
                    t2r_neg = t_neg_sh 
                    t2rgb_loss,out_in = get_contrastive_loss1(t_anchor,t2r_pos,t2r_neg,
                                                   criterion_inter,T = T1, weight =weight_T)#, weight =weight_T
                    sim_t2r = out_in

                prec_t2rsh[k] = get_sh_prec(t_anchor,t_pos_sh,t2r_neg)#r2t_neg   r_neg_sh
                prec_r2tsh[k] = get_sh_prec(r_anchor,r_pos_sh,r2t_neg)#t2r_neg   t_neg_sh
                #############################################################################
                fuse_con_loss,_ = get_contrastive_loss1(fuse_anchor,fuse_pos,fuse_neg,
                                                        criterion_inter,T = 0.07,weight=weight_RT)#,weight=weight_RT
                prec_fuse[k] = get_sh_prec(fuse_anchor,fuse_pos,fuse_neg)
                #############################################################################
                if opts['seg_task']:
                    rgb_loss,jl_r = get_contrastive_loss1(anchor_rs,r_anchor,r_neg_sh,
                                                        criterion_inter,T = T1)
                    t_loss,jl_t = get_contrastive_loss1(anchor_ts,t_anchor,t_neg_sh,
                                                      criterion_inter,T = T1)
                    loss_inter = (rgb2t_loss + t2rgb_loss)+(rgb_loss + t_loss)+fuse_con_loss
                else:
                    loss_inter = (rgb2t_loss + t2rgb_loss)+ fuse_con_loss
            else:
                intilize = (i==0)
                memory(r_pos_sh,t_pos_sh,r_neg_sh,t_neg_sh, k,intilize)
                criterion_inter = NCESoftmaxLoss()
                def shuffle_sample(feats):
                    idx = np.asarray(range(feats.size(0)))
                    np.random.shuffle(idx)
                    feats = feats[idx,:]
                    return feats
                T1 = 0.07
                with torch.no_grad():
                    _,sim_r2t = get_contrastive_loss1(r_anchor,t_pos_sh,t_neg_sh,
                                                      criterion_inter,T = T1)

                    _,sim_t2r = get_contrastive_loss1(t_anchor,r_pos_sh,r_neg_sh,
                                                      criterion_inter,T = T1)
                    
                prec_t2rsh[k] = 0
                prec_r2tsh[k] = 0
                loss_inter = 0
                prec_fuse[k] = get_sh_prec(fuse_anchor,fuse_pos,fuse_neg)
            
            ##############################################################################################
            #################################################################################################
            a1 = 0.1
            loss =  c_loss + loss_itm + a1*(loss_inter) + (loss_total_seg)
            # print("1194",c_loss,loss_itm,loss_inter,loss_total_seg)
            
            loss.backward(retain_graph=True)
  
            epoch_loss[k] = loss.item()
            epoch_clsloss[k]=cls_loss.item()
            
            if opts['inter_task']:
                epoch_interloss[k]= class_loss.item()
            else:
                epoch_interloss[k]= 0
            if opts['con_task']:
                epoch_crossloss[k]= loss_inter.item()
            else:
                epoch_crossloss[k]= 0
            if opts['match_task']:
                epoch_matchloss[k] = (loss_itm).item()
            else:
                epoch_matchloss[k] = 0
            if opts['seg_task']:
                epoch_segloss[k] = (loss_total_seg).item()
            else:
                epoch_segloss[k] = 0

            batch_cur_idx=batch_cur_idx+1
            
            if (batch_cur_idx%opts['seqbatch_size'])==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
                optimizer_cls.step()
                model.zero_grad()
                batch_cur_idx = 0

            if batch_cur_idx == 0:
                update_vis_plot(viz,batch_cur_idx, 
                                epoch_clsloss[k], epoch_interloss[k],
                                epoch_crossloss[k],epoch_matchloss[k],epoch_segloss[k],
                                epoch_loss[k],
                                iter_plot,'replace',opts['seqbatch_size'])
            else:
                update_vis_plot(viz,batch_cur_idx, 
                                epoch_clsloss[k], epoch_interloss[k],
                                epoch_crossloss[k],epoch_matchloss[k],epoch_segloss[k],
                                epoch_loss[k],
                                iter_plot,'append',opts['seqbatch_size'])

            toc = time.time()-tic 

            print("Cycle %2d, K %2d (%2d), Loss %.3f,Precr %.3f,Prect %.3f,Prec_sh %.3f,Prec_sp %.3f,Time %.3f" % \
                 (i, j, k,loss.item(),prec_r[k],prec_t[k],prec_r2tsh[k],prec_t2rsh[k],toc))

            x=1
            assert x==0

        update_vis_plot(viz,i, epoch_clsloss.mean(), epoch_interloss.mean(),
                        epoch_crossloss.mean(),epoch_matchloss.mean(),epoch_segloss.mean(),
                        epoch_loss.mean(),
                        epoch_plot,'append')
        update_vis_plot1(viz,i,
                        prec.mean(),
                        prec_r.mean(),prec_t.mean(),
                        epoch_plot1,'append')
        #'prec_match','prec_r2tsh','prec_t2rsh','prec_fuse'
        update_vis_plot2(viz,i,
                        prec_match.mean(),
                        prec_r2tsh.mean(),prec_t2rsh.mean(),prec_fuse.mean(),
                        epoch_plot2,'append')
        
        #####################################################################################
        prec_save[i] = prec
        prec_save1[i] = prec_r
        prec_save2[i] = prec_t
        cur_prec = prec.mean()
        s = cur_prec
        print("Mean Precision: %.3f,Best Precision: %.3f" % (s,best_prec))

        if s > best_prec:
            best_prec = s
            if opts['use_gpu']:
                model = model.cpu()
            states = {
                    'RGB_layers': model.RGB_layers.state_dict(),
                    'T_layers': model.T_layers.state_dict(),
                    'redection_layers': model.redection_layers.state_dict(),
                    
                    'linear_layers': model.linear_layers.state_dict(),
                    'shared_linear': model.shared_linear.state_dict(),
                    'fuse_layers':model.fuse_layers.state_dict(),
                    
                    'cls_branches': model.cls_branches.state_dict(),
                    'match_branches': model.match_branches.state_dict(),

                    'optimizer_cls': optimizer_cls.state_dict(),
                    'epoch':i,
                    'val_precs':val_precs,
                    
                    'train_prec':prec_save,
                    'train_prec_r':prec_save1,
                    'train_prec_t':prec_save2,

                    }

            print("Save model to %s" % opts['model_path'])
            torch.save(states, opts['model_path'])
            
            if opts['use_gpu']:
                model = model.cuda()
        
        if (i+1)%20 ==0 or (i+1) == opts['n_cycles']:
            if opts['val_task']:
                val_prec = np.zeros(K1)
                val_ious = np.zeros(K1)
                for j in range(K1):
                    print("626",j)
                    rgb_scenes, t_scenes, pos_rois, neg_rois,samples,idx,anchor_rois,pos_ious, neg_ious= dataset2[j].next(opts)
                    
                    batch_size = len(rgb_scenes)
                    prec_batch = np.zeros(batch_size)
                    iou_batch = np.zeros(batch_size)
                    pos_ious = pos_ious.view(batch_size,-1).cuda()
                    neg_ious = neg_ious.cuda()
                    
                    for sidx in range(0, batch_size):
                        sample = samples[sidx]
                        if opts['use_gpu'] & torch.cuda.is_available():
                            cur_rgbscene = rgb_scenes[sidx].cuda()
                            cur_tscene = t_scenes[sidx].cuda()
                            cur_anchor_roi = anchor_rois[sidx].cuda()
                            cur_pos_rois = pos_rois[sidx].cuda()
                            cur_neg_rois = neg_rois[sidx].cuda()  
                        if cur_pos_rois.shape[0] == 0:
                            continue
                        model.eval()
                        
                        r_sp_map,t_sp_map,cls_map = model(cur_rgbscene,cur_tscene,\
                                                          k=j,out_layer='conv4')

                        def get_roi_feats(cur_feat_map,pos_rois,neg_rois):

                            cur_pos_feats = model.roi_align_model(cur_feat_map, pos_rois)
                            cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                            
                            cur_neg_feats = model.roi_align_model(cur_feat_map, neg_rois)
                            cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)
                            return cur_pos_feats,cur_neg_feats

                        r_sp_pos,r_sp_neg = get_roi_feats(r_sp_map,cur_pos_rois,cur_neg_rois)
                        t_sp_pos,t_sp_neg = get_roi_feats(t_sp_map,cur_pos_rois,cur_neg_rois)
                        cls_pos,cls_neg = get_roi_feats(cls_map,cur_pos_rois,cur_neg_rois)

                        pos_score,pos_score1,pos_score2,\
                            = model(r_sp_pos,t_sp_pos,cls_pos,k=j,in_layer='fc4',out_layer='fc45')
                        neg_score,neg_score1,neg_score2,\
                            = model(r_sp_neg,t_sp_neg,cls_neg,k=j,in_layer='fc4',out_layer='fc45')
                        prec_batch[sidx] = evaluator(pos_score, neg_score)
                        score = torch.cat((pos_score,neg_score),dim=0)
                        top_scores, top_idx = score[:,1].topk(5)
                        top_idx = top_idx.data.cpu().numpy()

                        target_score = top_scores.data.mean()
                        target_bbox = sample[top_idx].mean(axis=0)
                        iou_batch[sidx] = overlap_ratio(cur_anchor_roi[0].cpu().numpy(),np.array(target_bbox))[0]

                    val_prec[j] = prec_batch.mean()
                    val_ious[j] = iou_batch.mean()
                
                # print("670",val_prec,val_prec.mean())
                # print("670",val_ious,val_ious.mean())
                val_precs.append([val_prec.mean(),val_ious.mean()])
                model.train()

            if opts['use_gpu']:
                model = model.cpu()
            states = {
                    'RGB_layers': model.RGB_layers.state_dict(),
                    'T_layers': model.T_layers.state_dict(),
                    'redection_layers': model.redection_layers.state_dict(),

                    'linear_layers': model.linear_layers.state_dict(),
                    'shared_linear': model.shared_linear.state_dict(),
                    'fuse_layers':model.fuse_layers.state_dict(),
                    
                    'cls_branches': model.cls_branches.state_dict(),
                    'match_branches': model.match_branches.state_dict(),
                    'optimizer_cls': optimizer_cls.state_dict(),
                    'epoch':i,
                    'val_precs':val_precs,

                    'train_prec':prec_save,
                    'train_prec_r':prec_save1,
                    'train_prec_t':prec_save2,
                    }

            print("Save model to %s" % opts['save_path'])
            model_path = os.path.join(opts['save_path'],'MANet-2IC_{}.pth'.format(i+1))
            torch.save(states, model_path)
            if opts['use_gpu']:
                model = model.cuda()

if __name__ == "__main__":
    train_mdnet(dataset='gtot')#'vot2019''gtot'