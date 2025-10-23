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

dicts = []
atomic_label_k = {'Single': 0, 'Miss': 1, 'Void': 2, 'Mutual': 3, 'Share': 4}
event_label_k = {'SingleGaze': 0, 'GazeFollow': 1, 'AvertGaze': 2, 'MutualGaze': 3, 'JointAtt': 4}

f = open(os.path.join("D:/Phdworks/data/D_StaticGazes/D_StaticGazes/event/train", "event_sample.pkl"), "rb")
event_dict = pickle.load(f)

for mode in event_dict.keys():
    dicts.extend(event_dict[mode])

sequence_list = []
for index in range(len(dicts)):
    rec = dicts[int(index)][0]
    vid, nid1, nid2, start_fid, end_fid, mode = rec

    event_label = torch.IntTensor([event_label_k[mode]])
    video_info = np.load(
        os.path.join('D:/Phdworks/data/D_StaticGazes/D_StaticGazes', 'annotation', 'train', 'vid_{}_ant_all.npy'.format(vid)),
        allow_pickle=True)
    item_pre1 = []
    sequence_label1 = []
    sequence_len1 = []

    item_pre2 = []
    sequence_label2 = []
    sequence_len2 = []

    for idx in range(int(end_fid-start_fid)):
        fid = start_fid + idx
        f_info = video_info[fid]['ant']

        label_atomic1 = f_info[0]['SmallAtt']
        label_atomic1 = atomic_label_k[label_atomic1]
        label_atomic2 = f_info[1]['SmallAtt']
        label_atomic2 = atomic_label_k[label_atomic2]

        if item_pre1 != label_atomic1:
            sequence_len1.append(0)
            sequence_label1.append(label_atomic1)
        else:
            sequence_len1[-1] += 1
        item_pre1 = label_atomic1



        if item_pre2 != label_atomic2:
            sequence_len2.append(0)
            sequence_label2.append(label_atomic2)
        else:
            sequence_len2[-1] += 1
        item_pre2 = label_atomic2

    sequence_list.append({'label': event_label, 'data': sequence_label1, 'len': sequence_len1})
    sequence_list.append({'label': event_label, 'data': sequence_label2, 'len': sequence_len2})


len_total = len(sequence_list)
ind = np.random.permutation(len_total)

with open('D:/Phdworks/data/D_StaticGazes/D_StaticGazes/event/train/train_seq.pickle', 'wb') as f:
    pickle.dump([sequence_list[index] for index in ind], f)
f.close()

