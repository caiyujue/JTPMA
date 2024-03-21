import os
import sys
import numpy as np

from PIL import Image
import cv2

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf
import random
import math
from torchvision import transforms

from img_cropper import *

sys.path.insert(0,'../modules')
from sample_generator import *
from utils import *

def get_sample_num(batch_num,length):
    index = np.arange(length)
    num_per_batch = length/batch_num
    if math.ceil(num_per_batch) > num_per_batch:
        index1 = np.random.permutation(length)
        index = np.concatenate((index, index1[:math.ceil(num_per_batch)*batch_num-length]))
    index=index.reshape(batch_num,-1)
    for bn in range(batch_num):
        index[bn] = np.random.permutation(index[bn])
    index = index.swapaxes(1,0).reshape(-1)
    return index

class RegionDataset(data.Dataset):
    def __init__(self, rgbimg_dir, rgbimg_list, timg_dir, timg_list, 
                 gt, opts):

        self.rgbimg_list = np.array([os.path.join(rgbimg_dir, img) for img in rgbimg_list])
        self.timg_list = np.array([os.path.join(timg_dir, img) for img in timg_list])
        
        self.gt = gt
        self.view=opts['view']

        self.batch_frames = opts['batch_frames']#8
        self.batch_pos = opts['batch_pos']#32
        self.batch_neg = opts['batch_neg']#96
        
        self.overlap_pos = opts['overlap_pos']#[0.7, 1]
        self.overlap_neg = opts['overlap_neg']#[0, 0.5]
        
        self.crop_size = opts['img_size']#107
        self.padding = opts['padding1']#1.2
        self.use_gpu = opts['use_gpu']

        self.index = np.random.permutation(len(self.rgbimg_list))#self.clean_index#
        self.pointer = 0
        self.splitindex = get_sample_num(opts['batch_frames'],len(self.rgbimg_list))
       
        image = Image.open(self.rgbimg_list[0]).convert('RGB')
        self.scene_generator = SampleGenerator('gaussian', image.size,trans_f=1.5, scale_f=1.2,valid=True)
        self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.1, False)
        self.neg_generator = SampleGenerator('uniform', image.size, 1, 1.2, 1.1, False)

        self.receptive_field = 75#receptive_field
        self.interval = opts['frame_interval']
        self.img_crop_model = imgCropper(opts['padded_img_size'])
        self.img_crop_model.eval()
        if opts['use_gpu']:
            self.img_crop_model.gpuEnable()

    def samples2maskroi(self,samples,receptive_field, cshape,padded_scene_size,padding_ratio):
        cur_resize_ratio = cshape / padded_scene_size
        rois = np.copy(samples)
        rois[:, 2:4] += rois[:, 0:2]
        rois_paddings = (rois[:,2:4]-rois[:,0:2])*(padding_ratio-1.)/2. 
        rois[:,0:2]-=rois_paddings
        rois[:,2:4]+=rois_paddings
        rois[:, 0] *= cur_resize_ratio[0]
        rois[:, 1] *= cur_resize_ratio[1]
        rois[:, 2] = np.maximum(rois[:,0] + 1, rois[:, 2] * cur_resize_ratio[0] - receptive_field )
        rois[:, 3] = np.maximum(rois[:,1] + 1, rois[:, 3] * cur_resize_ratio[1] - receptive_field )#receptive_fi
        return rois

    def get_target_crop(self,ref_id):
        ## get initial target
        init_RGB_image = Image.open(self.rgbimg_list[ref_id]).convert('RGB')
        init_RGB_image = np.asarray(init_RGB_image)
        init_T_image = Image.open(self.timg_list[ref_id]).convert('RGB')
        init_T_image = np.asarray(init_T_image)
        init_RGB_bbox = self.gt[ref_id]
        init_T_bbox = self.gt[ref_id]

        init_RGB_target = init_RGB_image[int(init_RGB_bbox[1]):int(init_RGB_bbox[1]+init_RGB_bbox[3]),int(init_RGB_bbox[0]):int(init_RGB_bbox[0]+init_RGB_bbox[2]),:]
        init_T_target = init_T_image[int(init_T_bbox[1]):int(init_T_bbox[1]+init_T_bbox[3]),int(init_T_bbox[0]):int(init_T_bbox[0]+init_T_bbox[2]),:]
        
        init_RGB_target = np.asarray(Image.fromarray(init_RGB_target).resize((95,95),Image.BILINEAR))
        init_T_target = np.asarray(Image.fromarray(init_T_target).resize((95,95),Image.BILINEAR))
        init_RGB_target = init_RGB_target[np.newaxis,:,:,:]
        init_T_target = init_T_target[np.newaxis,:,:,:]
        init_RGB_target = init_RGB_target.transpose(0,3,1,2)
        init_T_target = init_T_target.transpose(0,3,1,2)

        init_RGB_target = torch.from_numpy(init_RGB_target).float()
        init_T_target = torch.from_numpy(init_T_target).float()
        return init_RGB_target,init_T_target

    def __iter__(self):
        return self

    def __next__(self,opts):

        # next_pointer = min(self.pointer + self.batch_frames, len(self.rgbimg_list))
        # idx = self.index[self.pointer:next_pointer]
        # #print("115",self.index,idx)
        # if len(idx) < self.batch_frames:
        #     self.index = np.random.permutation(len(self.rgbimg_list))##len(self.rgbimg_list)#self.clean_index
        #     next_pointer = self.batch_frames - len(idx)
        #     idx = np.concatenate((idx, self.index[:next_pointer]))
        # self.pointer = next_pointer
        
        next_pointer = min(self.pointer + self.batch_frames, len(self.rgbimg_list))
        idx = self.splitindex[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.splitindex = get_sample_num(self.batch_frames,len(self.rgbimg_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.splitindex[:next_pointer]))
        self.pointer = next_pointer
        idx = np.concatenate(([0],idx))

        rgb_ori = []
        t_ori = []
        masks=[]

        bbox_all=[]
        bbox_all1=[]
        pos_ious = np.empty((0))
        neg_ious = np.empty((0))

        init_RGB_targets = []
        init_T_targets = []
        init_RGB_target,init_T_target = self.get_target_crop(0)
        init_RGB_targets.append(init_RGB_target)
        init_T_targets.append(init_T_target)
        
        for ref_i in idx:
            init_RGB_target,init_T_target = self.get_target_crop(ref_i)
            init_RGB_targets.append(init_RGB_target)
            init_T_targets.append(init_T_target)

        for i,(rgbimg_path,timg_path,bbox) in enumerate(zip(self.rgbimg_list[idx],self.timg_list[idx],self.gt[idx])):
            
            rgb_image = Image.open(rgbimg_path).convert('RGB')
            ir_image = Image.open(timg_path).convert('RGB')

            rgb_image = np.asarray(rgb_image)
            ir_image = np.asarray(ir_image)

            ################################################
            ishape = rgb_image.shape
            pos_examples,pos_iou = gen_samples1(SampleGenerator('gaussian', (ishape[1],ishape[0]), 0.1, 1.2, 1.1, False), 
                                                bbox, self.batch_pos, overlap_range=opts['overlap_pos'])
            neg_examples,neg_iou = gen_samples1(SampleGenerator('uniform', (ishape[1],ishape[0]), 1, 1.2, 1.1, False), 
                                                bbox, self.batch_neg, overlap_range=opts['overlap_neg'])

            pos_ious = np.concatenate((pos_ious, pos_iou),axis=0)
            neg_ious = np.concatenate((neg_ious, neg_iou),axis=0)

            ########## compute padded sample################
            padded_x1 = (neg_examples[:, 0] - neg_examples[:,2]*(self.padding-1.)/2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:,3]*(self.padding-1.)/2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2]*(self.padding+1.)/2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3]*(self.padding+1.)/2.).max()
            padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))#(4,)
        
            jitter_scale = 1.1 ** np.clip(3.*np.random.randn(1,1),-2,2)#np.array([[1.]])
            
            crop_img_size = (padded_scene_box[2:4] * ((self.crop_size, self.crop_size) / bbox[2:4])).astype('int64') * jitter_scale[0][0]
            
            ########################################################
            jittered_obj_size = jitter_scale[0][0]*float(self.crop_size)
            cropped_rgbimage, _ = self.img_crop_model.crop_image(rgb_image, np.reshape(padded_scene_box, (1, 4)), crop_img_size)
            cropped_irimage, _ = self.img_crop_model.crop_image(ir_image, np.reshape(padded_scene_box, (1, 4)), crop_img_size)

            cropped_rgbimage = cropped_rgbimage - 128.
            cropped_irimage = cropped_irimage - 128.

            crop_sz = torch.Tensor(crop_img_size)
            
            def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor) -> torch.Tensor:
                box_extract_center = box_extract[0:2] + 0.5*box_extract[2:4]
                box_in_center = box_in[0:2] + 0.5*box_in[2:4]
                box_out_center = (crop_sz-1)/2 + (box_in_center - box_extract_center)*resize_factor
                box_out_wh = box_in[2:4]*resize_factor
                box_out = torch.cat((box_out_center - 0.5*box_out_wh, box_out_wh))
                return box_out
            rf = (jittered_obj_size, jittered_obj_size)/bbox[2:4]
            box_crop = transform_image_to_crop(torch.Tensor(bbox).view(4), torch.Tensor(bbox).view(4),rf, crop_sz)
            bbox_all1.append(box_crop)
            def _make_aabb_mask(map_shape, bbox):
                mask = np.zeros(map_shape, dtype=np.float32)
                mask[int(round(bbox[1].item())):int(round(bbox[1].item() + bbox[3].item())), int(round(bbox[0].item())):int(round(bbox[0].item() + bbox[2].item()))] = 1
                return mask

            train_masks = torch.from_numpy(np.expand_dims(_make_aabb_mask(cropped_rgbimage.shape[-2:], box_crop), axis=0))
            ########################################################################################
            box_crop = np.copy(np.array([bbox]))
            box_crop[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), box_crop.shape[0], axis=0)
            box_crop = self.samples2maskroi(box_crop, self.receptive_field, (jittered_obj_size, jittered_obj_size),bbox[2:4], self.padding)
            
            batch_num = np.zeros((box_crop.shape[0], 1))
            box_crop = np.concatenate((batch_num, box_crop), axis=1)
            ########################################################################################
            batch_num = np.zeros((pos_examples.shape[0], 1))
            pos_rois = np.copy(pos_examples)
            pos_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), pos_rois.shape[0], axis=0)            
            pos_rois = self.samples2maskroi(pos_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),bbox[2:4], self.padding)       
            pos_rois = np.concatenate((batch_num, pos_rois), axis=1)
            ###########################################################################################
            batch_num = np.zeros((neg_examples.shape[0], 1))
            neg_rois = np.copy(neg_examples)
            neg_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), neg_rois.shape[0], axis=0)
            neg_rois = self.samples2maskroi(neg_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),bbox[2:4], self.padding)
            neg_rois = np.concatenate((batch_num, neg_rois), axis=1)
            #########################################################################
            
            if i==0:
                total_pos_rois = [torch.from_numpy(np.copy(pos_rois).astype('float32'))]
                total_neg_rois = [torch.from_numpy(np.copy(neg_rois).astype('float32'))]
            else:
                total_pos_rois.append(torch.from_numpy(np.copy(pos_rois).astype('float32')))
                total_neg_rois.append(torch.from_numpy(np.copy(neg_rois).astype('float32')))

            if self.use_gpu:
                cropped_rgbimage = cropped_rgbimage.cpu()
                cropped_irimage = cropped_irimage.cpu()

            rgb_ori.append(cropped_rgbimage)
            t_ori.append(cropped_irimage)
            masks.append(train_masks.unsqueeze(0))   

            bbox_all.append(torch.Tensor(box_crop).view(-1,5))  

        pos_ious = torch.from_numpy(pos_ious).float()
        neg_ious = torch.from_numpy(neg_ious).float()    

        return rgb_ori,t_ori,masks,total_pos_rois,total_neg_rois,\
                idx,bbox_all,pos_ious,neg_ious,\
                bbox_all1,init_RGB_targets,init_T_targets
            
    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)
        
        regions = regions.transpose(0,3,1,2)
        regions = regions.astype('float32') - 128.
        return regions


class RegionDataset1(data.Dataset):
    def __init__(self, rgbimg_dir, rgbimg_list, timg_dir, timg_list, 
                 gt,sample_num, opts):

        self.rgbimg_list = np.array([os.path.join(rgbimg_dir, img) for img in rgbimg_list])
        self.timg_list = np.array([os.path.join(timg_dir, img) for img in timg_list])
        self.gt = gt
        self.batch_frames = opts['batch_frames']#8
        self.batch_pos = opts['batch_pos']#32
        self.batch_neg = opts['batch_neg']#96
        self.sample_num = sample_num
        self.overlap_pos = opts['overlap_pos']#[0.7, 1]
        self.overlap_neg = opts['overlap_neg']#[0, 0.5]
        
        self.crop_size = opts['img_size']#107
        self.padding = opts['padding1']#1.2
        self.use_gpu = opts['use_gpu']
        #np.arange(len(self.rgbimg_list))#
        self.index = np.random.permutation(len(self.rgbimg_list))#self.clean_index#
        self.pointer = 0
        self.splitindex = get_sample_num(self.sample_num,len(self.rgbimg_list))
       
        image = Image.open(self.rgbimg_list[0]).convert('RGB')
        self.scene_generator = SampleGenerator('gaussian', image.size,trans_f=1.5, scale_f=1.2,valid=True)
        self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.1, False)
        self.neg_generator = SampleGenerator('uniform', image.size, 1, 1.2, 1.1, False)

        self.receptive_field = 75#receptive_field
        self.interval = opts['frame_interval']
        self.img_crop_model = imgCropper(opts['padded_img_size'])
        self.img_crop_model.eval()
        if opts['use_gpu']:
            self.img_crop_model.gpuEnable()

    def samples2maskroi(self,samples,receptive_field, cshape,padded_scene_size,padding_ratio):
        cur_resize_ratio = cshape / padded_scene_size
        rois = np.copy(samples)
        rois[:, 2:4] += rois[:, 0:2]
        rois_paddings = (rois[:,2:4]-rois[:,0:2])*(padding_ratio-1.)/2. 
        rois[:,0:2]-=rois_paddings
        rois[:,2:4]+=rois_paddings
        rois[:, 0] *= cur_resize_ratio[0]
        rois[:, 1] *= cur_resize_ratio[1]
        rois[:, 2] = np.maximum(rois[:,0] + 1, rois[:, 2] * cur_resize_ratio[0] - receptive_field )
        rois[:, 3] = np.maximum(rois[:,1] + 1, rois[:, 3] * cur_resize_ratio[1] - receptive_field )#receptive_fi
        return rois

    def __iter__(self):
        return self

    def __next__(self,opts):

        # next_pointer = min(self.pointer + self.batch_frames, len(self.rgbimg_list))
        # idx = self.index[self.pointer:next_pointer]
        # #print("115",self.index,idx)
        # if len(idx) < self.batch_frames:
        #     self.index = np.random.permutation(len(self.rgbimg_list))##len(self.rgbimg_list)#self.clean_index
        #     next_pointer = self.batch_frames - len(idx)
        #     idx = np.concatenate((idx, self.index[:next_pointer]))
        # self.pointer = next_pointer
        
        next_pointer = min(self.pointer + self.sample_num, len(self.rgbimg_list))
        idx = self.splitindex[self.pointer:next_pointer]
        if len(idx) < self.sample_num:
            self.splitindex = get_sample_num(self.sample_num,len(self.rgbimg_list))
            next_pointer = self.sample_num - len(idx)
            idx = np.concatenate((idx, self.splitindex[:next_pointer]))
        self.pointer = next_pointer
        
        rgb_ori = []
        t_ori = []

        bbox_all=[]
        pos_ious = np.empty((0))
        neg_ious = np.empty((0))

        for i,(rgbimg_path,timg_path,bbox) in enumerate(zip(self.rgbimg_list[idx],self.timg_list[idx],self.gt[idx])):
            
            rgb_image = Image.open(rgbimg_path).convert('RGB')
            ir_image = Image.open(timg_path).convert('RGB')
            #rgb_image1,ir_image1 = self.augmenter(1,rgb_image,ir_image)
            rgb_image = np.asarray(rgb_image)
            ir_image = np.asarray(ir_image)

            ################################################
            ishape = rgb_image.shape
            pos_examples,pos_iou = gen_samples1(SampleGenerator('gaussian', (ishape[1],ishape[0]), 0.1, 1.2, 1.1, False), 
                                                bbox, self.batch_pos, overlap_range=opts['overlap_pos'])
            neg_examples,neg_iou = gen_samples1(SampleGenerator('uniform', (ishape[1],ishape[0]), 1, 1.2, 1.1, False), 
                                                bbox, self.batch_neg, overlap_range=opts['overlap_neg'])

            pos_ious = np.concatenate((pos_ious, pos_iou),axis=0)
            neg_ious = np.concatenate((neg_ious, neg_iou),axis=0)

            ########## compute padded sample计算填充样本 ################
            padded_x1 = (neg_examples[:, 0] - neg_examples[:,2]*(self.padding-1.)/2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:,3]*(self.padding-1.)/2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2]*(self.padding+1.)/2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3]*(self.padding+1.)/2.).max()
            padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))#(4,)
        
            jitter_scale = 1.1 ** np.clip(3.*np.random.randn(1,1),-2,2)#np.array([[1.]])
            
            crop_img_size = (padded_scene_box[2:4] * ((self.crop_size, self.crop_size) / bbox[2:4])).astype('int64') * jitter_scale[0][0]
            
            ########################################################
            cropped_rgbimage, _ = self.img_crop_model.crop_image(rgb_image, np.reshape(padded_scene_box, (1, 4)), crop_img_size)
            cropped_irimage, _ = self.img_crop_model.crop_image(ir_image, np.reshape(padded_scene_box, (1, 4)), crop_img_size)

            cropped_rgbimage = cropped_rgbimage - 128.
            cropped_irimage = cropped_irimage - 128.

            ########################################################################################
            jittered_obj_size = jitter_scale[0][0]*float(self.crop_size)

            box_crop = np.copy(np.array([bbox]))
            # box_crop[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), box_crop.shape[0], axis=0)
            # box_crop = self.samples2maskroi(box_crop, self.receptive_field, (jittered_obj_size, jittered_obj_size),bbox[2:4], self.padding)
            # batch_num = np.zeros((box_crop.shape[0], 1))
            # box_crop = np.concatenate((batch_num, box_crop), axis=1)
            ########################################################################################
            batch_num = np.zeros((pos_examples.shape[0], 1))
            pos_rois = np.copy(pos_examples)
            pos_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), pos_rois.shape[0], axis=0)            
            pos_rois = self.samples2maskroi(pos_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),bbox[2:4], self.padding)       
            pos_rois = np.concatenate((batch_num, pos_rois), axis=1)
            pos_sample = torch.Tensor(pos_examples)
            ###########################################################################################
            batch_num = np.zeros((neg_examples.shape[0], 1))
            neg_rois = np.copy(neg_examples)
            neg_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), neg_rois.shape[0], axis=0)
            neg_rois = self.samples2maskroi(neg_rois, self.receptive_field, (jittered_obj_size, jittered_obj_size),bbox[2:4], self.padding)
            neg_rois = np.concatenate((batch_num, neg_rois), axis=1)
            neg_sample = torch.Tensor(neg_examples)
            all_sample = torch.cat((pos_sample,neg_sample),dim=0)
            #########################################################################
            
            if i==0:
                total_pos_rois = [torch.from_numpy(np.copy(pos_rois).astype('float32'))]
                total_neg_rois = [torch.from_numpy(np.copy(neg_rois).astype('float32'))]
                all_samples=[all_sample]
            else:
                total_pos_rois.append(torch.from_numpy(np.copy(pos_rois).astype('float32')))
                total_neg_rois.append(torch.from_numpy(np.copy(neg_rois).astype('float32')))
                all_samples.append(all_sample)

            if self.use_gpu:
                cropped_rgbimage = cropped_rgbimage.cpu()
                cropped_irimage = cropped_irimage.cpu()

            rgb_ori.append(cropped_rgbimage)
            t_ori.append(cropped_irimage)   

            bbox_all.append(torch.Tensor(box_crop).view(-1,4))  

        pos_ious = torch.from_numpy(pos_ious).float()
        neg_ious = torch.from_numpy(neg_ious).float()    
       
        return rgb_ori,t_ori,total_pos_rois,total_neg_rois,all_samples,\
               idx,bbox_all,pos_ious,neg_ious
            
    next = __next__








