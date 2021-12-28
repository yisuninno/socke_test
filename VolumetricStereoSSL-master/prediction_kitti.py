from __future__ import print_function
import argparse
import os
import torch.nn.parallel
from torch.autograd import Variable
from dataloader_new import readPFM
import numpy as np
from PIL import Image
import skimage
import skimage.io
import skimage.transform

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CompMatch')
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--data_path', type=str, default='../../data/', help="data root")
parser.add_argument('--dataset', type=int, default=3, help='1: sceneflow, 2: kitti. 3: kitti 2015')
parser.add_argument('--whichModel', type=int, default=5, help='0 for GANet, 1 for PSMNet, 2 for this method')
parser.add_argument('--backbone', type=bool, default=False, help='if the backbone is used')
parser.add_argument('--pretrained', type=bool, default=False, help='if the pretrained model is used')
parser.add_argument('--out_dir', type=str, default='rendered_imgs/', help="output image directory")
parser.add_argument('--save_path', type=str, default='./results/v10_train/', help="location to save result")
opt = parser.parse_args()

if opt.dataset == 1:
    opt.test_data_path = opt.data_path + 'FlyingThings3D/'
    opt.val_list = './lists/sceneflow_test.list'
    # opt.crop_height = 576
    # opt.crop_width = 960
    opt.crop_height = 240
    opt.crop_width = 576
elif opt.dataset == 2:
    opt.test_data_path = opt.data_path + 'KITTI2012/training/'
    opt.val_list = './lists/kitti2012_train.list'
    opt.crop_height = 384
    opt.crop_width = 1248
else:
    opt.test_data_path = opt.data_path + 'KITTI2015/training/'
    # opt.test_data_path = opt.data_path + 'KITTI2015/testing/'
    opt.val_list = './lists/kitti2015_train.list'
    # opt.val_list = './lists/kitti2015_test.list'
    # opt.crop_height = 384
    # opt.crop_width = 1248
    opt.crop_height = 240
    opt.crop_width = 576

if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.cuda.manual_seed(123)

# select the model
print('===> Building model')
if opt.whichModel == 0:
    from models.GANet_deep import GANet
    model = GANet(opt.max_disp)
    opt.resume = 'checkpoint/GANet/kitti2015_final.pth'
    if opt.dataset == 1 or opt.pretrained:
        opt.resume = 'checkpoint/GANet/sceneflow_epoch_10.pth'
elif opt.whichModel == 1:
    from models.PSMNet import *
    model = stackhourglass(opt.max_disp)
    opt.resume = 'checkpoint/PSMNet/pretrained_model_KITTI2015.tar'
    if opt.dataset == 1 or opt.pretrained:
        opt.resume = 'checkpoint/PSMNet/pretrained_sceneflow.tar'
        opt.psm_constant = 1.17
    if opt.crop_height == 240:
        opt.crop_height = 256
elif opt.whichModel == 2:
    if opt.backbone:
        from models.SingleFeatDS3_v7 import Model
        opt.resume = 'checkpoint/SingleFeatDS3_v10/kitti_epoch_561_best.pth'
        if opt.dataset == 1 or opt.pretrained:
            opt.resume = 'checkpoint/SingleFeatDS3_v8/_epoch_20.pth'
    else:
        from models.SingleFeatDS3_ct import Model
        opt.resume = 'checkpoint/SingleFeatDS3_ct/kitti_epoch_591_best.pth'
        if opt.dataset == 1 or opt.pretrained:
            opt.resume = 'checkpoint/SingleFeatDS3_ct/_epoch_20.pth'

    model = Model(opt.max_disp)
    model.training = False
elif opt.whichModel == 5:
    # from models.CompMatchDS3Feat_bg import Model
    from models.VR_v10 import Model
    model = Model(opt.max_disp, training=False)
    # opt.resume = 'checkpoint/CompMatchDS3Feat_bg/_epoch_20.pth'
    opt.resume = 'checkpoint/VR_v10/kitti__epoch_60.pth'


print(opt)

model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def inverse_normalize(img, mean_rgb, std_rgb):
    '''inverse the normalization for saving figures'''
    height, width, _ = img.shape
    temp_data = np.zeros([height, width, 3], 'float32')
    temp_data[:, :, 0] = img[:, :, 0] * std_rgb[0] + mean_rgb[0]
    temp_data[:, :, 1] = img[:, :, 1] * std_rgb[1] + mean_rgb[1]
    temp_data[:, :, 2] = img[:, :, 2] * std_rgb[2] + mean_rgb[2]
    return temp_data

def fetch_data(A, crop_height, crop_width):
    if opt.dataset == 1:
        filename_l = opt.test_data_path + 'frames_finalpass/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'frames_finalpass/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9:len(A) - 1]
        filename_disp = opt.test_data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
        disp_left, height, width = readPFM(filename_disp)
        filename = opt.test_data_path + 'disparity/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9: len(A) - 4] + 'pfm'
        disp_right, height, width = readPFM(filename)
    elif opt.dataset == 2:
        filename_l = opt.test_data_path + 'colored_0/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'colored_1/' + A[0: len(A) - 1]
        filename_disp = opt.test_data_path + 'disp_occ/' + A[0: len(A) - 1]
        disp_left = np.asarray(Image.open(filename_disp)).astype(float)
    elif opt.dataset == 3:
        filename_l = opt.test_data_path + 'image_2/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'image_3/' + A[0: len(A) - 1]
        # filename_disp = opt.test_data_path + 'disp_occ_0/' + A[0: len(A) - 1]
        # filename_disp = opt.test_data_path + 'disp_noc_0/' + A[0: len(A) - 1]
        # disp_left = np.asarray(Image.open(filename_disp)).astype(float)

    left = Image.open(filename_l)
    right = Image.open(filename_r)

    # cast to float
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)

    # normalization
    if opt.whichModel == 0:
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])
    else:
        mean_left = mean_right = np.array([0., 0., 0.])
        std_left = std_right = np.array([1., 1., 1.])
        left /= 255.
        right /= 255.

    temp_data[0:3, :, :] = np.moveaxis((left[:,:,:3] - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right[:,:,:3] - mean_right) / std_right, -1, 0)

    # crop data
    if height <= crop_height and width <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - height: crop_height, crop_width - width: crop_width] = temp
    else:
        start_x = int((width - crop_width) / 2)
        start_y = int((height - crop_height) / 2)

        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    input1_np = np.expand_dims(temp_data[0:3], axis=0)
    input2_np = np.expand_dims(temp_data[3:6], axis=0)

    return input1_np, input2_np, height, width

if __name__ == '__main__':
    # initialize
    f = open(opt.val_list, 'r')
    file_list = f.readlines()
    file_list.sort()

    data_total = len(file_list)

    # start to evaluate
    model.eval()
    for data_num in range(data_total):
        A = file_list[data_num]

        input1_np, input2_np, height, width = fetch_data(A, opt.crop_height, opt.crop_width)

        # from np to torch
        input1 = Variable(torch.from_numpy(input1_np), requires_grad=False)
        input2 = Variable(torch.from_numpy(input2_np), requires_grad=False)

        # to gpu
        input1 = input1.cuda()
        input2 = input2.cuda()

        # compute disparity
        temp = model(input1, input2)
        temp = temp.detach().cpu().numpy()
        if height <= opt.crop_height and width <= opt.crop_width:
            temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
        else:
            temp = temp[0, :, :]
        savename = opt.save_path + A[0: len(A) - 1]
        skimage.io.imsave(savename, (temp * 256).astype('uint16'))
        print("finished", savename)