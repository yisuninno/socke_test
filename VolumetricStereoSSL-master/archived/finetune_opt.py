from __future__ import print_function
import argparse
# from math import log10

import sys
import shutil
import os
import torch
import torch.nn.parallel
import torch.distributed as dist

from apex.parallel import DistributedDataParallel
from apex import amp
from apex.parallel import convert_syncbn_model

import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn.functional as F

from dataloader_new import DatasetFromList
from models.VR_v10 import Model
from loss import *

# Training settings
parser = argparse.ArgumentParser(description='Attack Stereo Matching')
parser.add_argument('--crop_height', type=int, default=256, help="crop height")
parser.add_argument('--crop_width', type=int, default=512, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")

# parser.add_argument('--resume', type=str, default='checkpoint/VR_v6/_epoch_2.pth', help="resume from saved model")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--start_ep', type=int, default=1, help="start epoch")

parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')

parser.add_argument('--data_path', type=str, default='/raid/home/kbcheng/data/', help="path to the data folder")
parser.add_argument('--dataset', type=int, default=5, help='0: sceneflow_subset, 1: sceneflow, 2: kitti. 3: kitti 2015. 5: raw kitti 2015')
parser.add_argument('--whichModel', type=int, default=1, help='0: GANet, 1: ours')


# parser.add_argument('--save_path', type=str, default='./checkpoint/SingleFeatDS3_ct/', help="location to save models")
parser.add_argument('--save_path', type=str, default='./checkpoint/VR_v10/raw', help="location to save models")
# parser.add_argument('--save_path', type=str, default='./checkpoint/GANet/kitti', help="location to save models")

# parser.add_argument('--log_dir', type=str, default='logs/CompMatchDouble_sf_mask', help="directory for saving logs")
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

opt = parser.parse_args()
print(opt)

if opt.dataset == 0:
    opt.train_data_path = opt.data_path + 'FlyingThings3D_subset/train/'
    opt.test_data_path = opt.data_path + 'FlyingThings3D_subset/val/'
    opt.training_list = './lists/sceneflow_subset_train.list'
    opt.val_list = './lists/sceneflow_subset_val.list'
    opt.crop_height = 240
    opt.crop_width = 576
elif opt.dataset == 2:
    opt.train_data_path = opt.data_path + 'KITTI2012/training/'
    opt.test_data_path = opt.train_data_path
    opt.training_list = './lists/kitti2012_train.list'
    opt.val_list = './lists/kitti2012_train.list'
    opt.crop_height = 240
    opt.crop_width = 576
elif opt.dataset == 3:
    opt.train_data_path = opt.data_path + 'KITTI2015/training/'
    opt.test_data_path = opt.train_data_path
    opt.training_list = './lists/kitti2015_val20.list'
    opt.val_list = './lists/kitti2015_val20.list'#'./lists/kitti2015_train.list'
    opt.crop_height = 240
    opt.crop_width = 576
elif opt.dataset == 5:
    opt.train_data_path = opt.data_path + 'kitti_data/'
    opt.test_data_path = opt.train_data_path
    opt.training_list = './splits/eigen_full/train_files.txt'
    opt.val_list = './splits/eigen_full/val_files.txt'
    opt.crop_height = 240
    opt.crop_width = 576

dist.init_process_group(backend='nccl')
torch.cuda.set_device(opt.local_rank)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

train_set = DatasetFromList(opt.train_data_path, opt.training_list, [opt.crop_height, opt.crop_width], True,
                            opt.left_right, opt.dataset, opt.shift, opt.whichModel) #opt.whichModel
test_set = DatasetFromList(opt.test_data_path, opt.val_list, [opt.crop_height, opt.crop_width], False, opt.left_right, opt.dataset, opt.whichModel) #opt.whichModel

train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False,
                                  drop_last=True, sampler=train_sampler, pin_memory=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


print('===> Building model')
# model = Model(opt.max_disp)
if opt.whichModel == 0:
    model = GANet(opt.max_disp)
elif opt.whichModel == 2:
    model = stackhourglass(opt.max_disp)
else:
    model = Model(opt.max_disp)
    # model.training = False

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of parameters:", pytorch_total_params)

if cuda:
    # model = torch.nn.DataParallel(model).cuda()
    model = convert_syncbn_model(model).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
# model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
model = DistributedDataParallel(model)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50,300], gamma=0.5)

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if opt.whichModel==1:
            amp.load_state_dict(checkpoint['amp'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def train(epoch):
    train_sampler.set_epoch(epoch)
    model.train()

    # sift = kornia.SIFTDescriptor(16, 8, 4)

    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target, target_right, occ, occ_2, img_full_l, img_full_r = Variable(batch[0], requires_grad=True), \
                                               Variable(batch[1],requires_grad=True), \
                                               Variable(batch[2], requires_grad=False),\
                                               Variable(batch[3], requires_grad=False), \
                                                Variable(batch[4], requires_grad=False), \
                                                Variable(batch[5], requires_grad=False), \
                                                Variable(batch[6], requires_grad=False), \
                                                Variable(batch[7], requires_grad=False)
        # input1, input2, img_full_l, img_full_r = Variable(batch[0], requires_grad=True), \
        #                                        Variable(batch[1],requires_grad=True), \
        #                                        Variable(batch[2], requires_grad=False),\
        #                                        Variable(batch[3], requires_grad=False)

        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
            target_right = target_right.cuda()
            occ = occ.cuda()
            occ_2 = occ_2.cuda()
            img_full_l = img_full_l.cuda()
            img_full_r = img_full_r.cuda()

        
        target = torch.squeeze(target, 1)
        target_right = torch.squeeze(target_right, 1)

        # mask = torch.squeeze(~occ, 1) & (target < opt.max_disp)
        mask = target < opt.max_disp
        mask.detach_()

        # mask_right = torch.squeeze(~occ_2, 1) & (target_right < opt.max_disp)
        mask_right = target_right < opt.max_disp
        mask_right.detach_()

        c_mask = torch.unsqueeze(mask, 1).repeat(1, 3, 1, 1)
        c_mask_r = torch.unsqueeze(mask_right, 1).repeat(1, 3, 1, 1)

        valid = target[mask].size()[0]
        if valid > 0:

            for t in range(200):
                optimizer.zero_grad()

                # disp1, disp2, disp3 = model(input1, input2, input3, input4)
                disp1, _, __ = model(input1, input2, img_full_l, img_full_r)

                pred_img_l, pred_img_r, pred_disp_l, pred_disp_r = disp1[0], disp1[1], disp1[2], disp1[3]
                pred_disp_l = pred_disp_l.unsqueeze(1)
                pred_disp_r = pred_disp_r.unsqueeze(1)

                c1_loss_l = 10 * loss_disp_unsupervised(pred_img_l, input1) #input1, pred_img_l
                c1_loss_r = 10 * loss_disp_unsupervised(pred_img_r[:,:,:-192], input2[:,:,:-192])     

                s1_loss_l = loss_disp_smoothness(pred_disp_l, input1)
                s1_loss_r = loss_disp_smoothness(pred_disp_r[:,:, :,:-192], input2[:,:, :,:-192])

                loss = c1_loss_l + 0.5*c1_loss_r + 0.1 * s1_loss_l #+ 0.1 * s1_loss_r

                disp1_loss_l = F.smooth_l1_loss(disp1[2][mask], target[mask], reduction='mean')
                # disp1_loss_r = F.smooth_l1_loss(disp1[3][mask_right], target_right[mask_right], reduction='mean')
                disp1_loss_r = F.smooth_l1_loss(disp1[3][:,:,:-192], target_right[:,:,:-192], reduction='mean')

                # loss = disp1_loss_l + 0.5 * disp1_loss_r

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # loss.backward()
                optimizer.step()

                if opt.local_rank == 0:
                    print("===> Epoch[{}]({}/{}): c_loss1: {:.4f}, c_loss2: {:.4f}, s_loss_l: {:.4f}, s_loss_r: {:.4f}, disp_loss1: {:.4f}, , disp_loss2: ({:.4f})".format(t, 200, #epoch, iteration,
                                                                                        len(training_data_loader),
                                                                                        c1_loss_l.item(), c1_loss_r.item(), s1_loss_l.item(), s1_loss_r.item(), disp1_loss_l.item(), disp1_loss_r.item()))
                    sys.stdout.flush()

            break


def save_checkpoint(save_path, epoch, state, is_best):
    filename = save_path + "_epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        # shutil.copyfile(filename, save_path + '_best.pth')
        torch.save(state, save_path + "_epoch_{}_best.pth".format(epoch))
    print("Checkpoint saved to {}".format(filename))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 400:
       lr = opt.lr
    else:
       lr = opt.lr*0.1
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    error = 100
    start_epoch = opt.start_ep
    for epoch in range(start_epoch, start_epoch + opt.nEpochs):
        # adjust_learning_rate(optimizer, epoch)
        train(epoch)
        # scheduler.step()
        # is_best = False

        if opt.local_rank == 0:
            is_best = False

            if opt.dataset == 1 or opt.dataset == 0 or opt.dataset==5:
                if epoch>=0:
                    save_checkpoint(opt.save_path, epoch,{
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'amp': amp.state_dict()
                    }, is_best)
            else:
                # loss = val()
                # if loss < error:
                #     error = loss
                #     is_best = True

                if epoch%10 == 0 and epoch >= 20:
                    save_checkpoint(opt.save_path, epoch,{
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'amp': amp.state_dict()
                        }, is_best)

