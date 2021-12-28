from __future__ import print_function
import argparse
import os
import torch
import torch.nn.parallel
from torch.autograd import Variable
from dataloader import readPFM
import numpy as np
from PIL import Image

# settings
parser = argparse.ArgumentParser(description='Stereo-Constrained PGD Attack')
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--crop_height', type=int, default=240, help="crop height")
parser.add_argument('--crop_width', type=int, default=384, help="crop width")
parser.add_argument('--data_path', type=str, default='../../data/', help="data root")
parser.add_argument('--dataset', type=int, default=3, help='1: sceneflow, 3: kitti 2015')
parser.add_argument('--whichModel', type=int, default=2, help='0 for GANet, 1 for PSMNet, 2 for this method')
parser.add_argument('--total_iter', type=int, default=20, help='iterations of PGD attack')
parser.add_argument('--e', type=float, default=0.03, help='epsilon of PGD attack')
parser.add_argument('--a', type=float, default=0.01, help='step size of PGD attack')
parser.add_argument('--double_occ', type=bool, default=False, help='if occlusion of the right image is excluded')
parser.add_argument('--backbone', type=bool, default=False, help='if the backbone is used')
parser.add_argument('--unconstrained_attack', type=bool, default=False, help='if the backbone is used')
opt = parser.parse_args()

# select file list according to dataset
if opt.dataset == 1:
    opt.test_data_path = opt.data_path + 'FT_subset/val/'
    opt.val_list = './lists/sceneflow_subset_val_1000.list'
elif opt.dataset == 2:
    opt.test_data_path = opt.data_path + 'KITTI2012/training/'
    opt.val_list = './lists/kitti2012_train.list'
else:
    opt.test_data_path = opt.data_path + 'KITTI2015/training/'
    opt.val_list = './lists/kitti2015_train.list'

if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# select the model
# print('===> Building model')
# if opt.whichModel == 5:
#     from models.CompMatchDS3Feat_bg import Model
#     model = Model(opt.max_disp)
#     opt.resume = 'checkpoint/CompMatchDS3Feat_bg/kitti_epoch_783_best.pth'
#     opt.whichModel = 2
#     if opt.dataset == 1:
#         opt.resume = 'checkpoint/CompMatchDS3Feat_bg/_epoch_20.pth'

print(opt)
# print("load parameters:", opt.resume)
# model = torch.nn.DataParallel(model).cuda()

# load trained parameters
# if opt.resume:
#     if os.path.isfile(opt.resume):
#         print("=> loading checkpoint '{}'".format(opt.resume))
#         checkpoint = torch.load(opt.resume, map_location='cuda:0')
#         model.load_state_dict(checkpoint['state_dict'], strict=False)
#     else:
#         print("=> no checkpoint found at '{}'".format(opt.resume))

def fetch_data(A, crop_height=240, crop_width=576):
    if opt.dataset == 1:
        filename_l = opt.test_data_path + 'frames_finalpass/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'frames_finalpass/' + 'right/' + A[5:len(A) - 1]

        filename_disp = opt.test_data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
        disp_left, height, width = readPFM(filename_disp)
        disp_left = -disp_left

        filename_disp_r = opt.test_data_path + 'disparity/' + 'right/' + A[5:len(A) - 4] + 'pfm'
        disp_right, height, width = readPFM(filename_disp_r)

        filename_occ = opt.test_data_path + 'disparity_occlusions/' + A[0: len(A) - 1]
        occ_left = Image.open(filename_occ)
        occ_left = np.asarray(occ_left)
        occ_left = occ_left | (disp_left >= opt.max_disp)

        filename_occ_r = opt.test_data_path + 'disparity_occlusions/' + 'right/' + A[5:len(A) - 1]
        occ_right  = Image.open(filename_occ_r)
        occ_right = np.asarray(occ_right)
        occ_right = occ_right | (occ_right >= opt.max_disp)

    elif opt.dataset == 3:
        filename_l = opt.test_data_path + 'image_2/' + A[0: len(A) - 1]
        filename_r = opt.test_data_path + 'image_3/' + A[0: len(A) - 1]
        filename_disp = opt.test_data_path + 'disp_noc_0/' + A[0: len(A) - 1]
        filename_disp_2 = opt.test_data_path + 'disp_noc_1/' + A[0: len(A) - 1]
        disp_left = np.asarray(Image.open(filename_disp)).astype(float)
        disp_right = np.asarray(Image.open(filename_disp_2)).astype(float)

    left = Image.open(filename_l)
    right = Image.open(filename_r)

    # cast to float
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([10, height, width], 'float32')
    left = np.asarray(left).astype(float)
    right = np.asarray(right).astype(float)

    # generate masks for KITTI2015
    if opt.dataset != 1:
        disp_left[disp_left < 0.01] = width * 2 * 256
        disp_left = disp_left / 256.
        occ_left = (disp_left >= opt.max_disp).astype(float)

        disp_right[disp_right < 0.01] = width * 2 * 256
        disp_right = disp_right / 256.
        occ_right = (disp_right >= opt.max_disp).astype(float)

    # normalization
    scale = 1.0
    if opt.whichModel == 0:
        mean_left = np.array([np.mean(left[:, :, 0]), np.mean(left[:, :, 1]), np.mean(left[:, :, 2])])
        std_left = np.array([np.std(left[:, :, 0]), np.std(left[:, :, 1]), np.std(left[:, :, 2])])

        mean_right = np.array([np.mean(right[:, :, 0]), np.mean(right[:, :, 1]), np.mean(right[:, :, 2])])
        std_right = np.array([np.std(right[:, :, 0]), np.std(right[:, :, 1]), np.std(right[:, :, 2])])

        scale = 255.0
    else:
        mean_left = mean_right = np.array([0.485, 0.456, 0.406])
        std_left = std_right = np.array([0.229, 0.224, 0.225])
        left /= 255.
        right /= 255.


    temp_data[0:3, :, :] = np.moveaxis((left - mean_left) / std_left, -1, 0)
    temp_data[3:6, :, :] = np.moveaxis((right - mean_right) / std_right, -1, 0)

    temp_data[6, :, :] = width * 2
    temp_data[6, :, :] = disp_left

    temp_data[7, :, :] = occ_left.astype(float)
    temp_data[8, :, :] = occ_right.astype(float)

    temp_data[9, :, :] = width * 2
    temp_data[9, :, :] = disp_right

    # crop data
    if height <= crop_height and width <= crop_width:
        temp = temp_data
        temp_data = np.zeros([9, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - height: crop_height, crop_width - width: crop_width] = temp

        # set the filled-in areas as occluded to avoid to count as results
        temp_data[7, 0:crop_height - height, :] = 1.0
        temp_data[7, :, 0:crop_width - width] = 1.0
        temp_data[8, 0:crop_height - height, :] = 1.0
        temp_data[8, :, 0:crop_width - width] = 1.0
    else:
        start_x = int((width - crop_width) / 2)
        start_y = int((height - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]


    input1_np = np.expand_dims(temp_data[0:3], axis=0)
    input2_np = np.expand_dims(temp_data[3:6], axis=0)
    target_np = np.expand_dims(temp_data[6:7], axis=0)
    occ_np = np.expand_dims(temp_data[7:8], axis=0)
    occ_np = occ_np.astype(bool)
    occ_2_np = np.expand_dims(temp_data[8:9], axis=0)
    occ_2_np = occ_2_np.astype(bool)
    target_2_np = np.expand_dims(temp_data[9:10], axis=0)


    # print(rgb_min_l, rgb_min_r, rgb_max_l, rgb_max_r)
    return input1_np, input2_np, target_np, target_2_np, occ_np, occ_2_np


def opt_volume_render(x, y, num_steps, step_size, maxdisp = 192):
    """optimize disp through volume rendering"""

    # construct the volume of colors
    num, channels, height, width = x.size()
    temp = torch.zeros(num, channels, height, maxdisp).cuda()
    y = torch.cat((y, temp), dim=-1)
    c_adv = x.new().resize_(num, channels * 2, maxdisp, height, width + maxdisp).zero_()
    for i in range(maxdisp):
        if i > 0 :
            c_adv[:, :, maxdisp-1-i, :, i:width]  = (x[:,:,:,i:width] + y[:,:,:,:width-i]) * 0.5
            c_adv[:, :, maxdisp-1-i, :, width:]  = y[:,:,:,:-i]
        else:
            c_adv[:, :, maxdisp-1-i, :, :width]  = (x[:,:,:,:width] + y[:,:,:,:width]) * 0.5
            c_adv[:, :, maxdisp-1-i, :, width:]  = y


    # batch_size, channels, height, width = x.detach().cpu().numpy().shape
    s_adv = torch.zeros([batch_size, 1, maxdisp, height, width + maxdisp], requires_grad=True, device='cuda')
    # c_adv = torch.zeros([batch_size, 3, maxdisp, height, width + maxdisp], requires_grad=True, device='cuda')


    for i in range(num_steps):
        # new
        _s_adv = s_adv.clone().detach().requires_grad_(True)
        _c_adv = c_adv.clone().detach().requires_grad_(True)

        block_1 = _s_adv[:, :, :, :, :width]
        through_1 = 1.0 - block_1
        cumprod = torch.cumprod(through_1, dim=2)
        w = block_1.clone()
        w[:, 1:, :, :] = cumprod[:, :, maxdisp - 1, :, :] * block_1[:, :, 1:, :, :]
        disp_filter = Variable(
            torch.Tensor(np.reshape(np.array(range(maxdisp - 1, -1, -1)), [1, maxdisp, 1, 1])).cuda(),
            requires_grad=False)

        disp_filter = disp_filter.repeat(batch_size, 1, 1, height, width)
        disp_1 = torch.sum(w * disp_filter, 1)

        color_1 = _c_adv[:, :, :, :, :width]

        block_2 = torch.zeros_like(block_1)
        color_2 = torch.zeros_like(color_1)
        for i in range(maxdisp):
            if i > 0:
                block_2[:, :, maxdisp - 1 - i, :, :-i] = block_1[:, :, maxdisp - 1 - i, :, i:]
                color_2[:, :, maxdisp - 1 - i, :, :-i] = color_1[:, :, maxdisp - 1 - i, :, i:]
            else:
                block_2[:, :, maxdisp - 1 - i, :, :] = block_1[:, :, maxdisp - 1 - i, :, :]
                color_2[:, :, maxdisp - 1 - i, :, :] = color_1[:, :, maxdisp - 1 - i, :, :]
        through_2 = 1.0 - block_2
        cumprod_2 = torch.cumprod(through_2, dim=1)
        w2 = block_2.clone()
        w2[:, 1:, :, :] = cumprod_2[:, :, maxdisp - 1, :, :] * through_2[:, :, 1:, :, :]
        disp_2 = torch.sum(w2 * disp_filter, 1)

        color_1 = torch.sum(w * color_1, 2)
        color_2 = torch.sum(w2 * color_2, 2)

        loss_1 = torch.mean(torch.abs(x1 - color_1))
        loss_2 = torch.mean(torch.abs(x2 - color_2))
        loss = loss_1 + loss_2 
        loss.backward()

        print(loss_1.detach().cpu().numpy())
        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            gradients = _s_adv.grad.sign() * step_size
            # gradients_2 = _x2_adv.grad.sign() * step_size

            s_adv -= gradients
            # x2_adv += gradients_2


    # input1 = x1 + _x_adv

    return disp_1, disp_2
    # return input1.detach(), input2.detach(), err_list


def my_mean(temp):
    return temp[~np.isnan(temp)].mean()


if __name__ == '__main__':

    # preprocessing
    f = open(opt.val_list, 'r')
    file_list = f.readlines()

    # initialize lists to keep records
    data_total = 1 #len(file_list)
    before_loss_list = np.zeros(data_total)
    after_loss_list = before_loss_list.copy()
    diff_over_thr_3_ori_list = before_loss_list.copy()
    diff_over_thr_1_ori_list = before_loss_list.copy()
    diff_over_thr_3_list = before_loss_list.copy()
    diff_over_thr_1_list = before_loss_list.copy()
    err_iter_list = np.zeros((data_total, opt.total_iter))

    # start to loop through data
    model.eval()
    for data_num in range(data_total):
        A = file_list[data_num]

        # fetch data
        input1_np, input2_np, target_np, target_2_np, occ_np, occ_2_np = fetch_data(A, opt.crop_height, opt.crop_width)

        # from np to torch
        input1 = Variable(torch.from_numpy(input1_np), requires_grad=False)
        input2 = Variable(torch.from_numpy(input2_np), requires_grad=True)
        target = Variable(torch.from_numpy(target_np), requires_grad=False)
        target2 = Variable(torch.from_numpy(target_2_np), requires_grad=False)
        occ = Variable(torch.from_numpy(occ_np), requires_grad=False)
        occ_2 = Variable(torch.from_numpy(occ_2_np), requires_grad=False)

        # mask is the indices for fetching noise
        mask = torch.linspace(0, opt.crop_width - 1, steps=opt.crop_width, requires_grad=True)
        mask = mask.repeat(target.size()[0], target.size()[1], target.size()[2], 1)
        mask = mask - target
        mask = mask.round().long()

        # set those with out-of-crop correspondence as occluded
        occ = occ | (mask < 0)
        occ = torch.squeeze(occ, 1)

        mask = torch.clamp(mask, 0, opt.crop_width - 1)
        mask = mask.repeat(1, 3, 1, 1)

        # for the right image
        mask2 = torch.linspace(0, opt.crop_width-1, steps=opt.crop_width, requires_grad=True)
        mask2 = mask2.repeat(target2.size()[0], target2.size()[1], target2.size()[2], 1)
        mask2 = mask2 + target2

        occ_2 = occ_2 | (mask2 >= opt.crop_width)
        occ_2 = torch.squeeze(occ_2, 1)

        # occ_mask is occ repeated for RGB channels
        occ_mask = occ.repeat(1, 3, 1, 1)
        occ_2_mask = occ_2.repeat(1, 3, 1, 1)

        # to gpu
        input1 = input1.cuda()
        input2 = input2.cuda()
        target = target.cuda()
        occ = occ.cuda()
        mask = mask.cuda()
        occ_mask = occ_mask.cuda()
        occ_2_mask = occ_2_mask.cuda()

        target = torch.squeeze(target, 1)

        # unconstrained attack
        disp_1, disp_2 = opt_volume_render(input1, input2, num_steps=opt.total_iter, step_size=opt.a)

        # record EPE, bad 1.0, bad 3.0
        err = torch.mean(torch.abs(disp_1[~occ] - target[~occ])).detach().cpu().numpy()
        print("data", A, err)

        thr = 3
        diff_ori = torch.abs(disp_1[~occ] - target[~occ]).detach().cpu().numpy()
        diff_over_thr_ori = (diff_ori > thr).sum() / (~occ_np).sum()
        print("data", A, "Original error rate (3 px):", diff_over_thr_ori)
        # diff_over_thr_3_ori_list[data_num] = diff_over_thr_ori


