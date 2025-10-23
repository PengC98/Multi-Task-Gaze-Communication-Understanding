
from scipy.spatial.transform import Rotation

from .dataset_utils import get_head_mask, get_label_map, to_torch

import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import cv2

import glob
import os

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



cudnn.enabled = True
class ToColorMap(object):
    """Applies a color map to the given sample.
    Args:
        colormap: a valid plt.get_cmap
    """

    def __init__(self, colormap=plt.get_cmap("magma")):
        self.colormap = colormap

    def __call__(self, sample):
        sample_colored = self.colormap(np.array(sample))

        return Image.fromarray((sample_colored[:, :, :3] * 255).astype(np.uint8))

class GazeFollow(Dataset):
    def __init__(self, data_dir, labels_path, input_size=224, output_size=63, is_test_set=False):
        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        self.is_test_set = is_test_set
        self.head_bbox_overflow_coeff = 0.1  # Will increase/decrease the bbox of the head by this value (%)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        self.image_transform_o = transforms.Compose(
            [
                transforms.Resize((output_size, output_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )


        self.depth_transform = transforms.Compose(
            [ToColorMap(plt.get_cmap("magma")), transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        )
        self.depth_transform_g = transforms.Compose(
            [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        )
        self.transform_to = transforms.Compose([transforms.Resize((input_size, input_size)),transforms.ToTensor()])
        self.resize = transforms.Resize([input_size, input_size])
        self.t = transforms.ToTensor()


        column_names = [
            "path",
            "idx",
            "body_bbox_x",
            "body_bbox_y",
            "body_bbox_w",
            "body_bbox_h",
            "eye_x",
            "eye_y",
            "gaze_x",
            "gaze_y",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
        ]
        if is_test_set is False:
            column_names.append("inout")

        #df = pd.read_csv(labels_path, sep=",", names=column_names,nrows=12000, usecols=column_names, index_col=False)
        df = pd.read_csv(labels_path, sep=",", names=column_names, usecols=column_names, index_col=False)
        #df.sample(frac=1)

        if is_test_set:
            df = df[
                ["path", "eye_x", "eye_y", "gaze_x", "gaze_y", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"]
            ].groupby(["path", "eye_x"])
            self.keys = list(df.groups.keys())
            self.X = df
            self.length = len(self.keys)
        else:
            df = df[df["inout"] != -1]  # only use "in" or "out "gaze (-1 is invalid, 0 is out gaze)
            #df = df[df["inout"] == 1]
            df = df[df["bbox_x_min"]<df["bbox_x_max"]]
            #df.reset_index(inplace=True)

            self.X = df["path"]
            self.y = df[
                ["idx","bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "eye_x", "eye_y", "gaze_x", "gaze_y", "inout"]
            ]
            self.length = len(df)
        #self.pose_model = WHENet('WHENet\\WHENet.h5')
        #self.saliency_detector = SaliencyNet(True)
        #self.saliency_detector.eval()




    def __getitem__(self, index):
        if self.is_test_set:
            return self.__get_test_item__(index)
        else:
            return self.__get_train_item__(index)

    def __len__(self):
        return self.length

    def __get_train_item__(self, index):

        path = self.X.iloc[index]
        idx,x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, gaze_inside = self.y.iloc[index]
        idx = int(idx)

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert("RGB")
        #img.resize((self.input_size, self.input_size)).show()

        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        depth_path = path.replace("train", "depth").replace("test2", "depth2")


        depth = Image.open(os.path.join(self.data_dir, depth_path))

        #depth = depth.convert("L")

        # Data augmentation
        # Jitter (expansion-only) bounding box size
        if np.random.random_sample() < 0:
            self.head_bbox_overflow_coeff = np.random.random_sample() * 0.2
            x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
            x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)
        

        # Random flip
        if np.random.random_sample() <0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)#
            x_max_2 = width - x_min
            x_min_2 = width - x_max
            x_max = x_max_2
            x_min = x_min_2
            gaze_x = 1 - gaze_x

        #edge = cv2.Canny(np.asarray(o_depth), 10, 30)
        # Random color change
        if np.random.random_sample() < 0.5:
            img = adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
            img = adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
            img = adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))



        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)


        # Crop the face


        if int(x_min)>int(x_max):
            temp = x_min
            x_min = x_max
            x_max = temp

        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))


        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)

        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        true_label_heatmap = torch.zeros(self.output_size, self.output_size)
        if gaze_inside:
            true_label_heatmap = get_label_map(
                true_label_heatmap, [int(gaze_x * self.output_size), int(gaze_y * self.output_size)], 3,
                pdf="Gaussian"
            )
        head_position = [(x_min + x_max) / (2 * width), (y_min + y_max) / (2 * height)]

        head_position = torch.FloatTensor(head_position)

        gaze_point = torch.FloatTensor([gaze_x, gaze_y])
        gaze_direction = gaze_point - head_position
        theta = np.array(
            [[1, 0, 0], [0, 1, 0]])
        theta = torch.from_numpy(theta).float().unsqueeze(0)
        out_size224 = torch.Size(
            (1, 1, 224, 224))
        g224 = F.affine_grid(theta, out_size224).squeeze(0)



        data = {}

        data['head'] = head
        data['img'] = img
        data['depth'] = depth
        data['true_label_heatmap'] = true_label_heatmap
        data['face'] = face
        data['gt_direction'] = gaze_direction
        data['g224'] = g224
        data['inside'] = torch.FloatTensor([gaze_inside])

        return data

    def __get_test_item__(self, index):
        eye_coords = []
        gaze_coords = []
        gaze_inside = []
        for _, row in self.X.get_group(self.keys[index]).iterrows():
            path = row["path"]
            x_min = row["bbox_x_min"]
            y_min = row["bbox_y_min"]
            x_max = row["bbox_x_max"]
            y_max = row["bbox_y_max"]
            gaze_x = row["gaze_x"]
            gaze_y = row["gaze_y"]
            eye_x = row["eye_x"]
            eye_y = row["eye_y"]
            # All ground truth gaze are stacked up
            eye_coords.append([eye_x, eye_y])
            gaze_coords.append([gaze_x, gaze_y])
            gaze_inside.append(True)
        for _ in range(len(gaze_coords), 20):
            # Pad dummy gaze to match size for batch processing
            eye_coords.append([-1, -1])
            gaze_coords.append([-1, -1])
            gaze_inside.append(False)

        eye_coords = torch.FloatTensor(eye_coords)
        gaze_coords = torch.FloatTensor(gaze_coords)
        gaze_inside = torch.FloatTensor(gaze_inside)

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)
        head_box = np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * self.input_size
        head_box = head_box.astype(int)
        head_box = np.clip(head_box, 0, self.input_size - 1)
        head_point_x = (head_box[0] + head_box[2]) / 2
        head_point_y = (head_box[1] + head_box[3]) / 2
        

        img_o = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))


        #segments = slic(img_as_float(np.asarray(img.resize((self.input_size, self.input_size)))), n_segments=75,slic_zero=True)
        # Load depth image
        depth_path = path.replace("train", "depth").replace("test2", "depth2")

        depth = Image.open(os.path.join(self.data_dir, depth_path))
        o_depth = depth.resize((self.input_size, self.input_size))
        o_depth = np.array(o_depth).astype('float32')

        #grad = to_torch(np.concatenate((np.concatenate((p_xv, p_yv), 0), p_d), 0).transpose(2, 1, 0).reshape(224 * 224, 3))

        # Apply transformation to images...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)

        # ... and depth
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)


        # Generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)
        num_valid = 0
        for gaze_x, gaze_y in gaze_coords:
            if gaze_x == -1:
                continue

            num_valid += 1
            gaze_heatmap = get_label_map(
                gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size], 9, pdf="Gaussian"
            )

        gaze_heatmap /= num_valid

        data = {}


        data['face'] = face
        data['head'] = head
        data['img'] = img
        data['gaze_coords'] = gaze_coords
        data['eye_coords'] = eye_coords
        data['img_size'] = torch.IntTensor([width, height])

        data['gaze_heatmap'] = gaze_heatmap
        data['inside'] = gaze_inside[0]


        return data

    def get_head_coords(self, path):
        if not self.is_test_set:
            raise NotImplementedError("This method is not implemented for training set")

        # NOTE: this is not 100% accurate. I should also condition by eye_x
        # However, for the application of this method it should be enough
        key_index = next((key for key in self.keys if key[0] == path), -1)
        if key_index == -1:
            raise RuntimeError("Path not found")

        for _, row in self.X.get_group(key_index).iterrows():
            x_min = row["bbox_x_min"]
            y_min = row["bbox_y_min"]
            x_max = row["bbox_x_max"]
            y_max = row["bbox_y_max"]

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        return x_min, y_min, x_max, y_max






class VideoAttentionTargetImages(Dataset):

    def __init__(self, data_dir, labels_dir, input_size=224, output_size=64, is_test_set=False):
        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        self.is_test_set = is_test_set
        self.head_bbox_overflow_coeff = 0.1  # Will increase/decrease the bbox of the head by this value (%)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.depth_transform = transforms.Compose(
            [ToColorMap(plt.get_cmap("magma")), transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        )
        self.depth_transform_g = transforms.Compose(
            [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        )

        self.X = []
        for show_dir in glob.glob(os.path.join(labels_dir, "*")):
            for sequence_path in glob.glob(os.path.join(show_dir, "*", "*.txt")):
                df = pd.read_csv(
                    sequence_path,
                    header=None,
                    index_col=False,
                    names=["path", "xmin", "ymin", "xmax", "ymax", "gazex", "gazey"],
                )
                #df = df[df["gazex"] != -1]


                sequence_path = sequence_path.replace("\\", "/")

                show_name = sequence_path.split("/")[-3]
                clip = sequence_path.split("/")[-2]
                n = sequence_path.split("/")[-1]
                n = n.split(".")[-2]

                df["path"] = df["path"].apply(lambda path: os.path.join(show_name, clip, path, n))

                self.X.extend(df.values.tolist())

        self.length = len(self.X)

        print(f"Total images: {self.length} (is test set? {is_test_set})")

    def __getitem__(self, index):
        if self.is_test_set:
            return self.__get_test_item__(index)
        else:
            return self.__get_train_item__(index)

    def __len__(self):
        return self.length

    def __get_train_item__(self, index):
        path, x_min, y_min, x_max, y_max, gaze_x, gaze_y = self.X[index]

        path = os.path.join("images", path)
        path = path.replace("\\", "/")
        idx = path.split("/")[-1]
        path = path.split("/")[:-1]
        path = os.path.join(path[0], path[1], path[2], path[3])
        path = path.replace("\\", "/")

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size

        x_min, y_min, x_max, y_max, gaze_x, gaze_y = map(float, [x_min, y_min, x_max, y_max, gaze_x, gaze_y])

        depth_path = path.replace("images", "depths").replace("test2", "depth2")

        depth = Image.open(os.path.join(self.data_dir, depth_path))
        depth = depth.convert("L")


        if np.random.random_sample() < 0.5:
            self.head_bbox_overflow_coeff = np.random.random_sample() * 0.2
            x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
            x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        if np.random.random_sample() < 0.5:
            # if True:
            # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target

            crop_x_min = np.min([gaze_x, x_min, x_max])
            crop_y_min = np.min([gaze_y, y_min, y_max])
            crop_x_max = np.max([gaze_x, x_min, x_max])
            crop_y_max = np.max([gaze_y, y_min, y_max])


            # Randomly select a random top left corner
            if crop_x_min >= 0:
                crop_x_min = np.random.uniform(0, crop_x_min)
            if crop_y_min >= 0:
                crop_y_min = np.random.uniform(0, crop_y_min)

            # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
            crop_width_min = crop_x_max - crop_x_min
            crop_height_min = crop_y_max - crop_y_min
            crop_width_max = width - crop_x_min
            crop_height_max = height - crop_y_min
            # Randomly select a width and a height
            crop_width = np.random.uniform(crop_width_min, crop_width_max)
            crop_height = np.random.uniform(crop_height_min, crop_height_max)

            # Crop it
            img = crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
            depth = crop(depth, crop_y_min, crop_x_min, crop_height, crop_width)
            # Record the crop's (x, y) offset
            offset_x, offset_y = crop_x_min, crop_y_min

            # convert coordinates into the cropped frame
            x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y

            # normalize to [0,1]

            gaze_x, gaze_y = (gaze_x - offset_x) / float(crop_width), \
                                 (gaze_y - offset_y) / float(crop_height)

            gaze_x = np.clip(gaze_x, 0, 1)
            gaze_y = np.clip(gaze_y, 0, 1)
            # else:
            #     gaze_x = -1; gaze_y = -1

            # convert the spatial feat to cropped frame
            width, height = crop_width, crop_height


        else:
            gaze_x = gaze_x / width
            gaze_y = gaze_y / height

        # Random flip
        if np.random.random_sample() <= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)  #


            x_max_2 = width - x_min
            x_min_2 = width - x_max
            x_max = x_max_2
            x_min = x_min_2
            gaze_x = 1 - gaze_x

        img_o = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


        # edge = cv2.Canny(np.asarray(o_depth), 10, 30)
        # Random color change
        if np.random.random_sample() <= 0.5:
            img = adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
            img = adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
            img = adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)
        bound_remove = get_head_mask(5, 5, 219, 219, 224, 224, resolution=self.input_size).unsqueeze(0)

        # transforms.ToPILImage()(head.squeeze(0)).show()
        head_box = np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * self.input_size
        head_box = head_box.astype(int)
        head_box = np.clip(head_box, 0, self.input_size - 1)
        # Crop the face

        if int(x_min) > int(x_max):
            temp = x_min
            x_min = x_max
            x_max = temp

        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)

        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        true_label_heatmap = torch.zeros(64, 64)
        if(gaze_x<0):
            print(path)


        true_label_heatmap = get_label_map(
            true_label_heatmap, [int(gaze_x * 64), int(gaze_y * 64)], 3,
            pdf="Gaussian"
        )
        head_point_x = (head_box[0] + head_box[2]) / 2
        head_point_y = (head_box[1] + head_box[3]) / 2
        p_x = np.linspace(0, 224 - 1, 224)
        p_y = np.linspace(0, 224 - 1, 224)
        p_xv, p_yv = np.meshgrid(p_x, p_y)
        p_yv = p_yv - head_point_y
        p_xv = p_xv - head_point_x
        # p_d = o_depth - o_depth[int(head_point_x), int(head_point_y)]

        p_xv = p_xv[np.newaxis, :, :]
        p_yv = p_yv[np.newaxis, :, :]
        # p_d = p_d[np.newaxis, :, :]

        grad = to_torch(np.concatenate((p_xv, p_yv), 0).transpose(2, 1, 0).reshape(224 * 224, 2))

        theta = np.array(
            [[1, 0, 0], [0, 1, 0]])
        theta = torch.from_numpy(theta).float().unsqueeze(0)
        out_size224 = torch.Size(
            (1, 1, 224, 224))
        g224 = F.affine_grid(theta, out_size224).squeeze(0)


        data = {}

        data['grad'] = grad
        data['head'] = head
        data['img'] = img
        data['depth'] = depth
        data['true_label_heatmap'] = true_label_heatmap
        data['g224'] = g224
        data['face'] = face

        return data

    def __get_test_item__(self, index):
        (path, x_min, y_min, x_max, y_max, gaze_x, gaze_y) = self.X[index]

        path = os.path.join("images", path)
        path = path.replace("\\", "/")
        idx = path.split("/")[-1]
        path = path.split("/")[:-1]
        path = os.path.join(path[0], path[1], path[2], path[3])
        path = path.replace("\\", "/")

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert("RGB")
        width, height = img.size
        x_min, y_min, x_max, y_max, gaze_x, gaze_y = map(float, [x_min, y_min, x_max, y_max, gaze_x, gaze_y])
        depth_path = path.replace("images", "depths").replace("test2", "depth2")



        width, height = img.size
        x_min, y_min, x_max, y_max, gaze_x, gaze_y = map(float, [x_min, y_min, x_max, y_max, gaze_x, gaze_y])

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)
        head_box = np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * self.input_size
        head_box = head_box.astype(int)
        head_box = np.clip(head_box, 0, self.input_size - 1)
        head_point_x = (head_box[0] + head_box[2]) / 2
        head_point_y = (head_box[1] + head_box[3]) / 2

        img_o = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        depth = Image.open(os.path.join(self.data_dir, depth_path))

        # grad = to_torch(np.concatenate((np.concatenate((p_xv, p_yv), 0), p_d), 0).transpose(2, 1, 0).reshape(224 * 224, 3))

        # depth = depth.convert("L")
        bound_remove = get_head_mask(5, 5, 219, 219, 224, 224, resolution=self.input_size).unsqueeze(0)

        # Apply transformation to images...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)

        # ... and depth
        if self.depth_transform is not None:
            depth = self.depth_transform_g(depth)

        gaze_x /= float(width)
        gaze_y /= float(height)
        gaze_heatmap = torch.zeros(64, 64)  # set the size of the output
        gaze_heatmap = get_label_map(
                gaze_heatmap, [gaze_x * 64, gaze_y * 64], 3, pdf="Gaussian"
            )

        eye_x = np.mean([x_max, x_min]) / width
        eye_y = np.mean([y_max, y_min]) / height

        eye_coords = [eye_x, eye_y]
        gaze_coords = [gaze_x, gaze_y]
        eye_coords = torch.FloatTensor(eye_coords)
        gaze_coords = torch.FloatTensor(gaze_coords)

        theta = np.array(
            [[1, 0, 0], [0, 1, 0]])
        theta = torch.from_numpy(theta).float().unsqueeze(0)
        out_size224 = torch.Size(
            (1, 1, 224, 224))

        g224 = F.affine_grid(theta, out_size224).squeeze(0)

        data = {}

        data['face'] = face
        data['head'] = head

        data['img'] = img
        data['img_o'] = img_o
        data['depth'] = depth
        data['gaze_coords'] = gaze_coords
        data['eye_coords'] = eye_coords
        data['img_size'] = torch.IntTensor([width, height])
        data['gaze_heatmap'] = gaze_heatmap
        data['g224'] = g224
        data['bound_remove'] = bound_remove

        return data


class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}


def convert_to_aflw(rotvec, is_rotvec=True):
    if is_rotvec:
        rotvec = Rotation.from_rotvec(rotvec).as_matrix()
    rot_mat_2 = np.transpose(rotvec)
    angle = Rotation.from_matrix(rot_mat_2).as_euler('xyz', degrees=True)

    return np.array([angle[0], -angle[1], -angle[2]])
