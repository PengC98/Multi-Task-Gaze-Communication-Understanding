import os
from os import listdir
from os.path import isfile
import shutil
import numpy as np
import utils
import pandas as pd

# be careful with frame id!

def ReadAnnot(paths):

    ant_files = [f for f in sorted(listdir(os.path.join(paths.data_root, 'annotation_clean/train'))) if
                 isfile(os.path.join(paths.data_root, 'annotation_clean/train', f))]



    for file_ind in range(len(ant_files)):

        vid=os.path.splitext(ant_files[file_ind])[0]
        print('video id : {}'.format(vid))
        df = pd.read_csv(os.path.join(paths.data_root, 'annotation_clean/train',ant_files[file_ind]))

        messages = {}
        event = {}
        video_filename = None
        gaze_points = {}
        head_box = {}
        head_box_2 = {}
        for i, row in df.iterrows():
            frame_index = row['frame_num']
            message = row['static_label']
            h_x_min = row['p1_xmin']
            h_x_max = row['p1_xmax']
            h_y_min = row['p1_ymin']
            h_y_max = row['p1_ymax']
            gt_x = int(row['gt_x'])
            gt_y = int(row['gt_y'])

            h2_x_min = row['p2_xmin']
            h2_x_max = row['p2_xmax']
            h2_y_min = row['p2_ymin']
            h2_y_max = row['p2_ymax']
            event_label = row['event_label']
            if frame_index not in messages:
                messages[frame_index] = []
            messages[frame_index].append(message)

            if frame_index not in event:
                event[frame_index] = []
            event[frame_index].append(event_label)

            if frame_index not in gaze_points:
                gaze_points[frame_index] = []
            gaze_points[frame_index].append((gt_x, gt_y))

            if frame_index not in head_box:
                head_box[frame_index] = []
            head_box[frame_index].append((h_x_min, h_y_min, h_x_max, h_y_max))

        frame_indices = sorted(messages.keys())
        frame_num = len(frame_indices)

        ant_all=list()


        for frame_ind in range(frame_num):

            ant_tmp=list()
            person_tmp = list()
            frame_index = frame_indices[frame_ind]
            m = messages.get(frame_index, [])


            for idx, message in enumerate(m):

                print(head_box[frame_index][idx],gaze_points[frame_index][idx],event[frame_index][idx],message)

                person_tmp.append(
                        {'pos': head_box[frame_index][idx], 'frame_ind': frame_index, 'gaze': gaze_points[frame_index][idx],
                         'BigAtt': event[frame_index][idx], 'SmallAtt': message,'vid':vid})


            # get det_box_tmp and det_class_tmp for this frame, starts with person, then object
            for per_ind in range(len(person_tmp)):
                    ant_tmp.append(person_tmp[per_ind])


            if len(ant_tmp)>0:
                ant_all.append({'ant':ant_tmp})

                # with open(os.path.join(paths.data_root,'all','img_list',vid+'.txt'),'a') as towrite:
                #
                #     towrite.write(os.path.join(paths.data_root,'all','img',vid,str((frame_ind+1)).zfill(5)+'.png'))
                #     towrite.write('\n')


        # np.save(os.path.join(paths.data_root, mode, 'ant_processed', 'vid_{}_adjmat'.format(vid)),adj_mat_all)
        # np.save(os.path.join(paths.data_root, mode, 'ant_processed', 'vid_{}_ant_all'.format(vid)),ant_all)

        np.save(os.path.join(paths.data_root, 'annotation/train', 'vid_{}_ant_all'.format(vid)),ant_all)

def main(paths):

    ReadAnnot(paths)

class Paths(object):
    def __init__(self):
        self.project_root='D:/Phdworks/year2/'
        self.log_root=os.path.join(self.project_root,'log')
        self.data_root=os.path.join('/home/lfan/Dropbox/Projects/ICCV19/DATA/')
        self.tmp_root = os.path.join(self.project_root, 'tmp')

if __name__ == '__main__':

    paths=Paths()

    paths.data_root='D:/Phdworks/data/D_StaticGazes/D_StaticGazes'

    main(paths)


    #tmp=np.load(os.path.join(paths.data_root, 'all', 'ant_processed', 'vid_{}_ant_all.npy'.format(66)))

    pass