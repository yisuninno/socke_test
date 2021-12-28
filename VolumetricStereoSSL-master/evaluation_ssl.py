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
parser.add_argument('--dataset', type=int, default=1, help='0: sceneflow subset 1: sceneflow, 2: kitti. 3: kitti 2015')
parser.add_argument('--whichModel', type=int, default=5, help='0 for GANet, 1 for PSMNet, 2 for this method')
parser.add_argument('--backbone', type=bool, default=False, help='if the backbone is used')
parser.add_argument('--pretrained', type=bool, default=False, help='if the pretrained model is used')
parser.add_argument('--out_dir', type=str, default='rendered_imgs/', help="output image directory")
opt = parser.parse_args()

if opt.dataset == 0:
    opt.train_data_path = opt.data_path + 'FlyingThings3D_subset/train/'
    opt.test_data_path = opt.data_path + 'FlyingThings3D_subset/val/'
    opt.training_list = './lists/sceneflow_subset_train.list'
    opt.val_list = './lists/sceneflow_subset_val.list'
    # opt.crop_height = 240
    # opt.crop_width = 576
    opt.crop_height = 576
    opt.crop_width = 960
elif opt.dataset == 1:
    opt.test_data_path = opt.data_path + 'FlyingThings3D/'
    opt.val_list = './lists/sceneflow_test.list'
    opt.crop_height = 576
    opt.crop_width = 960
    # opt.crop_height = 240
    # opt.crop_width = 576
elif opt.dataset == 2:
    opt.test_data_path = opt.data_path + 'KITTI2012/training/'
    opt.val_list = './lists/kitti2012_train.list'
    opt.crop_height = 384
    opt.crop_width = 1248
else:
    opt.test_data_path = opt.data_path + 'KITTI2015/training/'
    opt.val_list = './lists/kitti2015_train.list'
    opt.crop_height = 384
    opt.crop_width = 1248
    # opt.crop_height = 240
    # opt.crop_width = 576

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
    # opt.resume = 'checkpoint/VR_v10/kitti__epoch_60.pth'
    opt.resume = 'checkpoint/VR_v10_3/sf_epoch_16.pth'
    # opt.resume = 'checkpoint/VR_v10/kitti_disp_epoch_100.pth'
    # opt.resume = 'checkpoint/VR_v9/raw_epoch_1.pth'
    # opt.whichModel = 2

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
    if opt.dataset == 0:
        filename_l = opt.test_data_path + 'image_clean/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'image_clean/' + 'right/' + A[5:len(A) - 1]
        filename_disp = opt.test_data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
        disp_left, height, width = readPFM(filename_disp)
        disp_left = -disp_left
    elif opt.dataset == 1:
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
        filename_disp = opt.test_data_path + 'disp_occ_0/' + A[0: len(A) - 1]
        # filename_disp = opt.test_data_path + 'disp_noc_0/' + A[0: len(A) - 1]
        disp_left = np.asarray(Image.open(filename_disp)).astype(float)

    left = Image.open(filename_l)
    right = Image.open(filename_r)

    # cast to float
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
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

    # ignore disparities that are smaller than a threshold
    disp_left[disp_left < 0.01] = width * 2 * 256
    #disp_right[disp_right < 0.01] = width * 2 * 256
    if opt.dataset != 1 and opt.dataset != 0:
        disp_left = disp_left / 256.
        # disp_right = disp_right / 256.

    # range mask
    #mask_min = (disp_left < opt.max_disp).astype(float)

    temp_data[6, :, :] = width * 2
    temp_data[6, :, :] = disp_left
    #temp_data[7, :, :] = mask_min
    # temp_data[7, :, :] = disp_right

    # crop data
    if height <= crop_height and width <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - height: crop_height, crop_width - width: crop_width] = temp
    else:
        start_x = int((width - crop_width) / 2)
        start_y = int((height - crop_height) / 2)
        input1_full = np.expand_dims(temp_data[0: 3, start_y: start_y + crop_height, start_x-192 : start_x + crop_width], axis=0)
        input2_full = np.expand_dims(temp_data[3: 6, start_y: start_y + crop_height, start_x-192 : start_x + crop_width], axis=0)

        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    input1_np = np.expand_dims(temp_data[0:3], axis=0)
    input2_np = np.expand_dims(temp_data[3:6], axis=0)
    target_1_np = np.expand_dims(temp_data[6:7], axis=0)
    #mask_min_np = np.expand_dims(temp_data[7:8], axis=0).astype(bool)
    # target_2_np = np.expand_dims(temp_data[7:8], axis=0)



    return input1_np, input2_np, target_1_np#, input1_full, input2_full#, target_2_np

if __name__ == '__main__':
    # initialize
    f = open(opt.val_list, 'r')
    file_list = f.readlines()
    file_list.sort()

    # thresholds for the error rates
    thr_list = [1,2,3]

    data_total = len(file_list)
    mask_min_loss_list = np.zeros(data_total)
    mask_max_loss_list = np.zeros(data_total)
    whole_loss_list = np.zeros(data_total)

    err_mask_min_list = np.zeros((len(thr_list), data_total))
    err_mask_max_list = np.zeros((len(thr_list), data_total))
    err_whole_list = np.zeros((len(thr_list), data_total))

    # start to evaluate
    model.eval()
    for data_num in range(data_total):
        A = file_list[data_num]
        #input1_np, input2_np, target_np, mask_min_np = fetch_data(A, opt.crop_height, opt.crop_width)
        # input1_np, input2_np, target_1_np, input1_full_np, input2_full_np = fetch_data(A, opt.crop_height, opt.crop_width)
        input1_np, input2_np, target_1_np = fetch_data(A, opt.crop_height, opt.crop_width)

        # from np to torch
        input1 = Variable(torch.from_numpy(input1_np), requires_grad=False)
        input2 = Variable(torch.from_numpy(input2_np), requires_grad=False)
        target_1 = Variable(torch.from_numpy(target_1_np), requires_grad=False)
        target_1 = torch.squeeze(target_1, 1)

        # input1_full = Variable(torch.from_numpy(input1_full_np), requires_grad=False)
        # input2_full = Variable(torch.from_numpy(input2_full_np), requires_grad=False)

        # target_2 = Variable(torch.from_numpy(target_2_np), requires_grad=False)
        # target_2 = torch.squeeze(target_2, 1)
        #mask_min = Variable(torch.from_numpy(mask_min_np), requires_grad=False)
        #mask_min = torch.squeeze(mask_min, 1)

        # to gpu
        input1 = input1.cuda()
        input2 = input2.cuda()
        target_1 = target_1.cuda()

        # input1_full = input1_full.cuda()
        # input2_full = input2_full.cuda()

        # target_2 = target_2.cuda()
        #mask_min = mask_min.cuda()

        # range mask
        mask_1 = target_1 < opt.max_disp
        mask_1.detach_()

        # compute disparity
        if opt.whichModel!=5:
            disp = model(input1, input2).detach()
        else:
            disp = model(input1, input2).detach()
            # color_1a, color_1b, disp, disp_2 = model(input1, input2, input1_full, input2_full)

            # color_1a = color_1a.detach()
            # color_1b = color_1b.detach()
            # disp = disp.detach()
            # disp_2 = disp_2.detach()

            # # generate images
            # mean_left = mean_right = np.array([0.0, 0.0, 0.0])
            # std_left = std_right = np.array([1.0, 1.0, 1.0])

            # # save the perturbed left image
            # temp = color_1a.detach().cpu().numpy().squeeze()
            # temp = np.moveaxis(temp, 0, -1)
            # temp = inverse_normalize(temp, mean_left, std_left)

            # model_str = 'm' + str(opt.whichModel) + '_'

            # savename = opt.out_dir + model_str + 'left_' + A[0: len(A) - 1] 
            # skimage.io.imsave(savename, (temp * 255).astype('uint8'))

            # # save the perturbed right image
            # temp = color_1b.detach().cpu().numpy().squeeze()
            # temp = np.moveaxis(temp, 0, -1)
            # temp = inverse_normalize(temp, mean_left, std_left)

            # savename = opt.out_dir + model_str + 'right_' + A[0: len(A) - 1] 
            # skimage.io.imsave(savename, (temp * 255).astype('uint8'))

        # compute EPE, bad 1.0, and bad 3.0
        diff_mask_min = torch.abs(disp[mask_1] - target_1[mask_1]).detach()
        mask_min_loss = torch.mean(diff_mask_min).detach()
        mask_min_loss_list[data_num] = mask_min_loss.item()
        mask_min_total = mask_1.cpu().numpy().sum()
        diff_mask_min = diff_mask_min.cpu().numpy()

        for idx, thr in enumerate(thr_list):
            # over threshold error rates
            err_mask_min_thr = (diff_mask_min > thr).sum() / mask_min_total
            err_mask_min_list[idx, data_num] = err_mask_min_thr

            if thr==3:
                print("data", data_num+1, A[:-1], "EPE", mask_min_loss.item(), "| error rate (" + str(thr) + " px):", err_mask_min_thr)

    print("number of nans:", np.isnan(mask_min_loss_list).sum())
    print("loss mean:", mask_min_loss_list[~np.isnan(mask_min_loss_list)].mean())

    for idx, thr in enumerate(thr_list):
        print("mean error rate for threshold (" + str(thr) + " px):", err_mask_min_list[idx, :][~np.isnan(err_mask_min_list[idx, :])].mean())
