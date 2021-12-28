import torch.utils.data as data
import skimage
import skimage.io
import skimage.transform

from PIL import Image
import numpy as np
import random
from struct import unpack
import re
import sys
# import albumentations as A


def readPFM(file):
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
    #        quit()
    return img, height, width


def train_transform(temp_data, crop_height, crop_width, disp_right=False, shift=0):
    _, h, w = np.shape(temp_data)

    if h > crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, h + shift, crop_width + shift], 'float32')
        temp_data[6:7, :, :] = 1000
        temp_data[:, h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)

    start_x = random.randint(0, w - crop_width)
    start_y = random.randint(0, h - crop_height)
    temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]

    if disp_right:
        target_right = temp_data[7, :, :]
        return left, right, target, target_right

    return left, right, target


def test_transform(temp_data, crop_height, crop_width, disp_right=False):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height, crop_width], 'float32')
        temp_data[6: 7, :, :] = 1000
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]

    if disp_right:
        target_right = temp_data[7, :, :]
        return left, right, target, target_right

    return left, right, target


def train_transform_occ(temp_data, crop_height, crop_width, left_right=False, shift=0):
    _, h, w = np.shape(temp_data)

    if h > crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, h + shift, crop_width + shift], 'float32')
        temp_data[6:7, :, :] = 1000
        temp_data[:, h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)

    # start_x = random.randint(0, w - crop_width)
    start_x = random.randint(0, w - crop_width) #192
    start_y = random.randint(0, h - crop_height)
    crop_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left = crop_data[0: 3, :, :]
    right = crop_data[3: 6, :, :]
    target = crop_data[6: 7, :, :]
    occ = crop_data[7: 8, :, :]
    occ = occ.astype(bool)

    occ_2 = crop_data[8: 9, :, :]
    occ_2 = occ_2.astype(bool)

    target_2 = crop_data[9: 10, :, :]

    # x_full = np.zeros(3, crop_height, crop_width + 192*2)
    # y_full = np.zeros(3, crop_height, crop_width + 192*2)

    if start_x < 192:
        x_diff = 192 - start_x
        right_idx = start_x + crop_width + 192
        # x_full[:3, start_y: start_y + crop_height, x_diff:crop_width] = temp_data[:3, start_y: start_y + crop_height, :crop_width - x_diff]
        # x_full[:3, start_y: start_y + crop_height, :x_diff] = temp_data[:3, start_y: start_y + crop_height, 0].repeat(1, 1, x_diff)
        x_full = np.pad(temp_data[:3, start_y: start_y + crop_height, :right_idx], ((0,0), (0,0), (x_diff,0)), 'edge')
        y_full = np.pad(temp_data[3:6, start_y: start_y + crop_height, :right_idx], ((0,0), (0,0), (x_diff,0)), 'edge')

        # print(1, x_full.shape)

    elif start_x + crop_width + 192 > w:
        x_diff = start_x + crop_width + 192 - w
        x_full = np.pad(temp_data[:3, start_y: start_y + crop_height, start_x-192:], ((0,0), (0,0), (0,x_diff)), 'edge')
        y_full = np.pad(temp_data[3:6, start_y: start_y + crop_height, start_x-192:], ((0,0), (0,0), (0,x_diff)), 'edge')

        # print(2, x_full.shape)
    else:
        x_full = temp_data[:3, start_y: start_y + crop_height, start_x-192 : start_x + crop_width + 192*2]
        y_full = temp_data[3:6, start_y: start_y + crop_height, start_x-192 : start_x + crop_width + 192*2]

        # print(3, x_full.shape)
    return left, right, target, target_2, occ, occ_2, x_full, y_full

def raw_transform(temp_data, crop_height, crop_width, left_right=False, shift=0):
    _, h, w = np.shape(temp_data)

    # start_x = random.randint(0, w - crop_width)
    start_x = random.randint(192, w - crop_width) #192
    start_y = random.randint(0, h - crop_height)
    crop_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left = crop_data[0: 3, :, :]
    right = crop_data[3: 6, :, :]

    x_full = temp_data[0: 3, start_y: start_y + crop_height, start_x-192 : start_x + crop_width]
    y_full = temp_data[3: 6, start_y: start_y + crop_height, start_x-192 : start_x + crop_width]
    return left, right, x_full, y_full

def test_transform_occ(temp_data, crop_height, crop_width, left_right=False):
    _, h, w = np.shape(temp_data)

    start_x = int((w - crop_width) / 2)
    start_y = int((h - crop_height) / 2)
    temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]
    target_2 = temp_data[9: 10, :, :]
    occ = temp_data[7: 8, :, :]
    occ = occ.astype(bool)
    occ_2 = temp_data[8: 9, :, :]
    occ_2 = occ_2.astype(bool)
    return left, right, target, target_2, occ, occ_2


def test_transform_full(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    start_x = int((w - crop_width) / 2)
    start_y = int((h - crop_height) / 2)
    temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6: 7, :, :]
    occ = temp_data[7: 8, :, :]
    occ = occ.astype(bool)
    return left, right, target, occ


def load_data(data_path, current_file, method, transform=None):
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    A = current_file
    filename = data_path + 'frames_finalpass/' + A[0: len(A) - 1]
    left = Image.open(filename)
    filename = data_path + 'frames_finalpass/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9:len(A) - 1]
    right = Image.open(filename)
    filename = data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
    disp_left, height, width = readPFM(filename)
    filename = data_path + 'disparity/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9: len(A) - 4] + 'pfm'
    disp_right, height, width = readPFM(filename)

    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)

    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    # left = np.asarray(left).astype(float) #/ 255.
    # right = np.asarray(right).astype(float) #/ 255.

    if method == 0:
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])
    else:
        # mean_left = mean_right = np.array([0.485, 0.456, 0.406])
        # std_left = std_right = np.array([0.229, 0.224, 0.225])
        mean_left = mean_right = np.array([0., 0., 0.])
        std_left = std_right = np.array([1., 1., 1.])
        left /= 255.
        right /= 255.
    #
    temp_data[0:3, :, :] = np.moveaxis((left[:, :, :3] - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right[:, :, :3] - mean_right) / std_right, -1, 0)

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left
    temp_data[7, :, :] = disp_right
    return temp_data


def load_raw_data(data_path, current_file, method, transform=None):
    # imagenet_mean = [0.485, 0.456, 0.406]
    # imagenet_std = [0.229, 0.224, 0.225]
    A = current_file
    folder = current_file[0]
    frame_index = int(current_file[1])

    filename = data_path + folder + "/image_02/data/" + "{:010d}.png".format(frame_index)
    left = Image.open(filename)
    filename = data_path + folder + "/image_03/data/" + "{:010d}.png".format(frame_index)
    right = Image.open(filename)

    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)

    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    # left = np.asarray(left).astype(float) #/ 255.
    # right = np.asarray(right).astype(float) #/ 255.

    if method == 0:
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])
    else:
        # mean_left = mean_right = np.array([0.485, 0.456, 0.406])
        # std_left = std_right = np.array([0.229, 0.224, 0.225])
        mean_left = mean_right = np.array([0., 0., 0.])
        std_left = std_right = np.array([1., 1., 1.])
        left /= 255.
        right /= 255.
    #
    temp_data[0:3, :, :] = np.moveaxis((left[:, :, :3] - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right[:, :, :3] - mean_right) / std_right, -1, 0)

    return temp_data


def load_subset_data(data_path, current_file, method):
    A = current_file

    filename = data_path + 'disparity_occlusions/' + A[0: len(A) - 1]
    occ_left = Image.open(filename)

    filename_l = data_path + 'image_clean/' + A[0: len(A) - 1]
    filename_r = data_path + 'image_clean/' + 'right/' + A[5:len(A) - 1]

    filename_disp = data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
    disp_left, height, width = readPFM(filename_disp)
    disp_left = -disp_left

    filename_disp_r = data_path + 'disparity/' + 'right/' + A[5:len(A) - 4] + 'pfm'
    disp_right, height, width = readPFM(filename_disp_r)

    filename_occ = data_path + 'disparity_occlusions/' + A[0: len(A) - 1]
    occ_left = Image.open(filename_occ)
    occ_left = np.asarray(occ_left)
    # occ_left = occ_left | (disp_left >= opt.max_disp)

    filename_occ_r = data_path + 'disparity_occlusions/' + 'right/' + A[5:len(A) - 1]
    occ_right = Image.open(filename_occ_r)
    occ_right = np.asarray(occ_right)
    # occ_right = occ_right | (occ_right >= opt.max_disp)

    left = Image.open(filename_l)
    right = Image.open(filename_r)

    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([10, height, width], 'float32')
    left = np.asarray(left) / 255.
    right = np.asarray(right) / 255.
    occ_left = np.asarray(occ_left).astype(float)

    mean_left = mean_right = np.array([0., 0., 0.])
    std_left = std_right = np.array([1., 1., 1.])
    temp_data[0:3, :, :] = np.moveaxis((left - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right - mean_right) / std_right, -1, 0)

    temp_data[6, :, :] = width * 2
    temp_data[6, :, :] = disp_left

    temp_data[7, :, :] = occ_left.astype(float)
    temp_data[8, :, :] = occ_right.astype(float)

    temp_data[9, :, :] = width * 2
    temp_data[9, :, :] = disp_right
    return temp_data


def load_kitti_data(file_path, current_file):
    """ load current file from the list"""
    filename = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
    left = Image.open(filename)
    filename = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
    right = Image.open(filename)
    filename = file_path + 'disp_occ/' + current_file[0: len(current_file) - 1]

    disp_left = Image.open(filename)
    temp = np.asarray(disp_left)
    size = np.shape(left)

    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    disp_left = np.asarray(disp_left)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]

    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]

    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.1] = width * 2 * 256
    temp_data[6, :, :] = temp / 256.

    return temp_data


def load_kitti2015_data(file_path, current_file, method):
    """ load current file from the list"""
    max_disp=192

    filename = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
    left = Image.open(filename)
    filename = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
    right = Image.open(filename)

    filename_disp = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]
    filename_disp_2 = file_path + 'disp_occ_1/' + current_file[0: len(current_file) - 1]

    disp_left = np.asarray(Image.open(filename_disp)).astype(float)
    disp_right = np.asarray(Image.open(filename_disp_2)).astype(float)

    # disp_left = Image.open(filename)
    # temp = np.asarray(disp_left)
    size = np.shape(left)

    height = size[0]
    width = size[1]

    temp_data = np.zeros([10, height, width], 'float32')
    left = np.asarray(left).astype(float)  # / 255.0
    right = np.asarray(right).astype(float)  # / 255.0

    # disp_left = np.asarray(disp_left)

    if method == 0:
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])
    else:
        mean_left = mean_right = np.array([0., 0., 0.])
        std_left = std_right = np.array([1., 1., 1.])
        left /= 255.
        right /= 255.

    disp_left[disp_left < 0.01] = width * 2 * 256
    disp_left = disp_left / 256.
    occ_left = (disp_left >= max_disp).astype(float)

    disp_right[disp_right < 0.01] = width * 2 * 256
    disp_right = disp_right / 256.
    occ_right = (disp_right >= max_disp).astype(float)

    temp_data[0:3, :, :] = np.moveaxis((left[:, :, :3] - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right[:, :, :3] - mean_right) / std_right, -1, 0)

    # temp_data[6: 7, :, :] = width * 2
    # temp_data[6, :, :] = disp_left[:, :]
    # temp = temp_data[6, :, :]
    # temp[temp < 0.01] = width * 2 * 256
    # temp_data[6, :, :] = temp / 256.
    temp_data[6, :, :] = width * 2
    temp_data[6, :, :] = disp_left

    temp_data[7, :, :] = occ_left.astype(float)
    temp_data[8, :, :] = occ_right.astype(float)

    temp_data[9, :, :] = width * 2
    temp_data[9, :, :] = disp_right

    return temp_data


def load_data_md(file_path, current_file, method=1):
    """ load current file from the list"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    imgl = file_path + current_file[0: len(current_file) - 1]
    gt_l = imgl.replace('im0.png', 'disp0GT.pfm')
    imgr = imgl.replace('im0.png', 'im1.png')

    left = Image.open(imgl)
    right = Image.open(imgr)

    disp_left, height, width = readPFM(gt_l)

    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)
    disp_left = np.asarray(disp_left)

    if method == 0:
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])
    else:
        mean_left = mean_right = np.array(imagenet_mean)
        std_left = std_right = np.array(imagenet_std)
        left /= 255.
        right /= 255.

    temp_data[0:3, :, :] = np.moveaxis((left[:, :, :3] - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right[:, :, :3] - mean_right) / std_right, -1, 0)

    temp_data[6: 7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.01] = width * 2  # * 256
    temp_data[6, :, :] = temp  # / 256.

    return temp_data


class DatasetFromList(data.Dataset):
    def __init__(self, data_path, file_list, crop_size=[256, 256], training=True, left_right=False, dataset=0, shift=0,
                 method=0):
        super(DatasetFromList, self).__init__()
        # self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        f = open(file_list, 'r')
        self.data_path = data_path
        self.file_list = f.readlines()
        if dataset==5:
            # get unique file names (originally l and r)
            self.file_list = list(set([tuple(i.split(' ')[:2]) for i in self.file_list]))

        self.training = training
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.left_right = left_right
        self.dataset = dataset
        self.shift = shift
        self.method = method
        # self.transform = A.Compose([A.RandomBrightnessContrast(p=0.5), A.RandomGamma(p=0.5), A.CLAHE(p=0.5)], p=0.5)

    def __getitem__(self, index):
        #    print self.file_list[index]
        if self.dataset == 0:
            temp_data = load_subset_data(self.data_path, self.file_list[index], self.method)
            if self.training:
                input1, input2, target, target_2, occ, occ_2, img_full_l, img_full_r = train_transform_occ(temp_data, self.crop_height, self.crop_width,
                                                                  self.left_right, self.shift)
            else:
                input1, input2, target, target_2, occ, occ_2 = test_transform_occ(temp_data, self.crop_height, self.crop_width)
            return input1, input2, target, target_2, occ, occ_2, img_full_l, img_full_r

        elif self.dataset == 1:

            temp_data = load_data(self.data_path, self.file_list[index], self.method, self.transform)

            if self.training:
                input1, input2, target, target_right = train_transform(temp_data, self.crop_height, self.crop_width, disp_right=True, shift=self.shift)
            else:
                input1, input2, target, target_right = test_transform(temp_data, self.crop_height, self.crop_width, disp_right=True)
            return input1, input2, target, target_right

        elif self.dataset == 2:  # load kitti2012 dataset
            temp_data = load_kitti_data(self.data_path, self.file_list[index])
            if self.training:
                input1, input2, target = train_transform(temp_data, self.crop_height, self.crop_width, self.left_right,
                                                         self.shift)
                return input1, input2, target
            else:
                input1, input2, target = test_transform(temp_data, self.crop_height, self.crop_width)
                return input1, input2, target
        elif self.dataset == 3:  # load kitti2015 dataset
            temp_data = load_kitti2015_data(self.data_path, self.file_list[index], self.method)
            if self.training:
                input1, input2, target, target_2, occ, occ_2, img_full_l, img_full_r = train_transform_occ(temp_data, self.crop_height, self.crop_width, self.left_right,
                                                         self.shift)
                return input1, input2, target, target_2, occ, occ_2, img_full_l, img_full_r
                # return input1, input2, img_full_l, img_full_r
            else:
                # input1, input2, target, occ = test_transform_full(temp_data, self.crop_height, self.crop_width)
                # return input1, input2, target, occ
                input1, input2, target = test_transform(temp_data, self.crop_height, self.crop_width)
                return input1, input2, target
        elif self.dataset == 5:
            temp_data = load_raw_data(self.data_path, self.file_list[index], self.method)
            input1, input2, img_full_1, img_full_2 = raw_transform(temp_data, self.crop_height, self.crop_width, self.left_right, self.shift)
            return input1, input2, img_full_1, img_full_2
        else:  # load scene flow dataset
            temp_data = load_data(self.data_path, self.file_list[index])

        if self.training:
            input1, input2, target = train_transform(temp_data, self.crop_height, self.crop_width, self.left_right,
                                                     self.shift)
            return input1, input2, target
        else:
            input1, input2, target = test_transform(temp_data, self.crop_height, self.crop_width)
            return input1, input2, target

    def __len__(self):
        return len(self.file_list)
