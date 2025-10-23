from .dataset_utils import get_head_mask, get_label_map, to_torch
import os
import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    crop,
)
import pickle

class DYgaze(Dataset):

    def __init__(self, data_dir, labels_dir, input_size=224, output_size=64,sampling_rate = 16,num_frames = 8, train_mode='train'):
        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        self.train_mode = train_mode
        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        f = open(os.path.join(self.data_dir, labels_dir, "event_sample.pkl"), "rb")
        event_dict = pickle.load(f)

        if train_mode=='train':
            self.dict = []#{'SingleGaze': list(), 'GazeFollow': list(), 'AvertGaze': list(), 'MutualGaze': list(), 'JointAtt': list()}
        else:
            self.dict = []
        for mode in event_dict.keys():
            #if train_mode=='train':
                #if mode == 'SingleGaze':
                    #l = len(event_dict[mode])
                    #del event_dict[mode][0:40]
                #if mode == 'MutualGaze':
                #    l = len(event_dict[mode])
                #    del event_dict[mode][0:int(l/2)]


            self.dict.extend(event_dict[mode])


        self.head_bbox_overflow_coeff = 0.1  # Will increase/decrease the bbox of the head by this value (%)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        #self.depth_transform = transforms.Compose(
         #   [ToColorMap(plt.get_cmap("magma")), transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        #)
        self.depth_transform_g = transforms.Compose(
            [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        )

        self.atomic_label = {'Single': 0, 'Miss': 1, 'Void': 2, 'Mutual': 3, 'Share': 4}
        self.event_label = {'SingleGaze': 0, 'GazeFollow': 1, 'AvertGaze': 2, 'MutualGaze': 3, 'JointAtt': 4}



    def __getitem__(self, index):
        if self.train_mode=='test' or self.train_mode=='val':
            return self.__get_test_item__(index)
        else:
            return self.__get_train_item__(index)

    def __len__(self):

        if self.train_mode=='train':
            return len(self.dict)

        else:
            return len(self.dict)

    def _random_sample_frame_idx(self, len):
        frame_indices = []
        if self.sampling_rate < 0:  # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices

    def _test_sample(self,len):

        sampled_indices = [0]  # Start with the first index

        # Compute the remaining 18 indices to sample evenly between 1 and array_length - 2
        additional_indices = np.linspace(1, len - 2, num=self.num_frames-2, dtype=int)

        # Add the additional indices and the last index
        sampled_indices.extend(additional_indices)
        sampled_indices.append(len - 1)

        return sampled_indices


    def _generate_temporal_crops(self, len):
        frame_indices = []
        seg_len = (self.num_frames - 1) * 10 + 1

        if seg_len >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * 10 if i * 10 < len else frame_indices[-1])
        else:
            slide_len = len - seg_len
            start = slide_len // 2
            frame_indices = list(range(start, start + 10 * self.num_frames, 10))
        return frame_indices

    def __get_train_item__(self, index):

        face_sq = torch.zeros((self.num_frames, 2, 3, 224, 224))
        head_mask_seq = torch.zeros((self.num_frames, 2, 1, 224, 224))
        img_sq = torch.zeros((self.num_frames, 3, 224, 224))
        label_atomic_seq = torch.zeros((self.num_frames, 2))
        gaze_inside_seq = torch.zeros((self.num_frames, 2))
        gaze_heatmap_seq = torch.zeros((self.num_frames, 2,self.output_size,self.output_size))
        gaze_point_seq = torch.ones((self.num_frames, 2,2))*-1
        gaze_direction = torch.zeros((self.num_frames, 2, 2))



        need_flip = np.random.random_sample()
        flip = False
        if need_flip<0.5:
            flip = False
        need_color_change = np.random.random_sample()
        c_change = False
        if need_color_change < 0.5:
            c_change = False
        brightness_factor = np.random.uniform(0.5, 1.5)
        contrast_factor = np.random.uniform(0.5, 1.5)
        saturation_factor = np.random.uniform(0.5, 1.5)

        rec = self.dict[int(index)][0]
        vid, nid1, nid2, start_fid, end_fid,mode = rec


        event_label = torch.IntTensor([self.event_label[mode]])
        vid = int(vid)
        nid1 = int(nid1)
        nid2 = int(nid2)
        start_fid = int(start_fid)
        end_fid = int(end_fid)
        #print(start_fid,end_fid)

        video_info = np.load(
            os.path.join(self.data_dir, 'annotation', self.train_mode, 'vid_{}_ant_all.npy'.format(vid)),
            allow_pickle=True)

        container = av.open(os.path.join(self.data_dir, 'vids',self.train_mode, '{}.mp4'.format(vid)))
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]
        #valid_frames = frames[s:e+1]
        frame_idx = self._random_sample_frame_idx(int(end_fid-start_fid))
        #if mode=='GazeFollow':
        #    if np.random.random_sample()<0.5:
        #        frame_idx = self._aug_frame_idx(int(end_fid - start_fid),video_info,start_fid,nid1,nid2)

        for sq_i,idx in enumerate(frame_idx):
            fid = start_fid+idx
            f = int(video_info[fid]['ant'][0]['frame_ind'])
            f_info = video_info[fid]['ant']


            img = frames[f].to_image()
            img = img.convert("RGB")
            width,height = img.size
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if c_change:
                img = adjust_brightness(img, brightness_factor=brightness_factor)
                img = adjust_contrast(img, contrast_factor=contrast_factor)
                img = adjust_saturation(img, saturation_factor=saturation_factor)

            for node_i in [0, 1]:
                nid = [nid1, nid2][node_i]
                head_pos = f_info[nid]['pos']
                label_atomic = f_info[nid]['SmallAtt']
                label_atomic = self.atomic_label[label_atomic]


                label_atomic_seq[sq_i,node_i] = label_atomic

                x_min,y_min,x_max,y_max = int(head_pos[0]),int(head_pos[1]),int(head_pos[2]),int(head_pos[3])
                if flip:
                    x_max_2 = width - x_min
                    x_min_2 = width - x_max
                    x_max = x_max_2
                    x_min = x_min_2

                face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                if self.image_transform is not None:
                    face = self.image_transform(face)
                face_sq[sq_i,node_i,:] = face

                head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)
                head_mask_seq[sq_i,node_i,:] = head
                gaze_x,gaze_y = f_info[nid]['gaze'][0],f_info[nid]['gaze'][1]
                if gaze_x!=-1:
                    if flip:
                        x_max_2 = width - gaze_x
                        gaze_x = x_max_2
                    gaze_inside_seq[sq_i, nid] = 1
                    gaze_x /= float(width)  # fractional gaze
                    gaze_y /= float(height)
                    gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
                    gaze_heatmap = get_label_map(
                        gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], 3, pdf="Gaussian"
                    )
                    gaze_heatmap_seq[sq_i, nid, :] = gaze_heatmap
                    gaze_point_seq[sq_i, nid, 0] = gaze_x
                    gaze_point_seq[sq_i, nid, 1] = gaze_y

                    head_position = [(x_min + x_max) / (2 * width), (y_min + y_max) / (2 * height)]

                    head_position = torch.FloatTensor(head_position)

                    gaze_point = torch.FloatTensor([gaze_x, gaze_y])
                    gaze_direction[sq_i, nid, :] = gaze_point - head_position


            if self.image_transform is not None:
                img = self.image_transform(img)
            img_sq[sq_i,:] = img


        data = {}

        data['head'] = head_mask_seq
        data['img'] = img_sq
        data['event_label'] = event_label
        data['atomic_label'] = label_atomic_seq
        data['true_label_heatmap'] = gaze_heatmap_seq
        data['gaze_point'] = gaze_point_seq
        data['face'] = face_sq
        data['gaze_inside'] = gaze_inside_seq
        data['gaze_direction'] = gaze_direction

        return data

    def __get_test_item__(self, index):


        face_sq = torch.zeros((self.num_frames, 2, 3, 224, 224))
        head_mask_seq = torch.zeros((self.num_frames, 2, 1, 224, 224))
        img_sq = torch.zeros((self.num_frames, 3, 224, 224))
        label_atomic_seq = torch.zeros((self.num_frames, 2))
        gaze_inside_seq = torch.zeros((self.num_frames, 2))
        gaze_coord = torch.zeros((self.num_frames, 2, 2))
        gaze_heatmap_seq = torch.zeros((self.num_frames, 2, 56, 56))
        gaze_point_seq = torch.ones((self.num_frames, 2, 2))*-1

        rec = self.dict[index][0]
        vid, nid1, nid2, start_fid, end_fid,mode = rec
        event_label = torch.IntTensor([self.event_label[mode]])
        vid = int(vid)
        nid1 = int(nid1)
        nid2 = int(nid2)
        start_fid = int(start_fid)
        end_fid = int(end_fid)

        container = av.open(os.path.join(self.data_dir, 'vids',self.train_mode, '{}.mp4'.format(vid)))
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]
        #valid_frames = frames[start_fid:end_fid + 1]
        frame_idx = self._generate_temporal_crops(int(end_fid-start_fid))

        video_info = np.load(
            os.path.join(self.data_dir, 'annotation', self.train_mode, 'vid_{}_ant_all.npy'.format(vid)),allow_pickle=True)

        for sq_i, idx in enumerate(frame_idx):
            fid = start_fid + idx
            f = int(video_info[fid]['ant'][0]['frame_ind'])
            f_info = video_info[fid]['ant']

            img = frames[f].to_image()
            img = img.convert("RGB")
            width, height = img.size

            for node_i in [0, 1]:
                nid = [nid1, nid2][node_i]
                head_pos = f_info[nid]['pos']
                label_atomic = f_info[nid]['SmallAtt']
                label_atomic = self.atomic_label[label_atomic]
                label_atomic_seq[sq_i, node_i] = label_atomic

                x_min,y_min,x_max,y_max = int(head_pos[0]),int(head_pos[1]),int(head_pos[2]),int(head_pos[3])
                face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                if self.image_transform is not None:
                    face = self.image_transform(face)
                face_sq[sq_i, node_i, :] = face

                head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)
                head_mask_seq[sq_i, node_i, :] = head
                gaze_x, gaze_y = f_info[nid]['gaze'][0], f_info[nid]['gaze'][1]
                if gaze_x != -1:
                    gaze_inside_seq[sq_i, nid] = 1
                    gaze_x /= float(width)  # fractional gaze
                    gaze_y /= float(height)
                    gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
                    gaze_heatmap = get_label_map(
                        gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], 3, pdf="Gaussian"
                    )
                    gaze_heatmap_seq[sq_i, nid, :] = gaze_heatmap
                    gaze_point_seq[sq_i, nid, 0] = gaze_x
                    gaze_point_seq[sq_i, nid, 1] = gaze_y





            if self.image_transform is not None:
                img = self.image_transform(img)
            img_sq[sq_i, :] = img


        data = {}

        data['head'] = head_mask_seq
        data['img'] = img_sq
        data['event_label'] = event_label
        data['atomic_label'] = label_atomic_seq
        data['true_label_heatmap'] = gaze_heatmap_seq
        data['face'] = face_sq
        data['gaze_inside'] = gaze_inside_seq
        data['gaze_coord'] = gaze_point_seq

        return data