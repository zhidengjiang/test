import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import FCN, U_Net, R2U_Net, AttU_Net, R2AttU_Net, V_Net, MDV_Net, BM_U_Net, init_weights, MD_UNet, \
    UNet_double
from ellipse import drawline_AOD
from ellipse2 import drawline_AOD as drawline_AOD2
import csv
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
import cv2 as cv
# from ranger import Ranger
from loss_func import FocalLoss, FocalLoss2d, BinaryDiceLoss, DiceLoss, Mul_FocalLoss, Convey_Loss, Shape_prior_Loss
# from DeformConvnet import Deform_U_Net,MD_Defm_V_Net
from tensorboardX import SummaryWriter  # Tensorboard显示
# import torch.onnx.symbolic_opset9
# from thop import profile
import pandas as pd
from edge_filter_test import *
import matplotlib.pyplot as plt


# 以上是导入相关的包
# @torch.onnx.symbolic_opset9.parse_args('v', 'is')
# def upsample_nearest2d(g, input, output_size):
# 	height_scale = float(output_size[-2]) / input.type().sizes()[-2]
# 	width_scale = float(output_size[-1]) / input.type().sizes()[-1]
# 	return g.op("Upsample", input,
# 		scales_f=(1, 1, height_scale, width_scale),
# 		mode_s="nearest")
# torch.onnx.symbolic_opset9.upsample_nearest2d = upsample_nearest2d

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # Data loader
        self.train_loader = train_loader  # 定义相关数据集的路径
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        # self.unet = None
        # self.optimizer = None
        self.img_ch = config.img_ch  # 定义输入通道
        self.output_ch = config.output_ch  # 定义输出通道
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.5, 2.,
                                                                                     0.2])).float()).cuda()  # self.criterion = torch.nn.BCELoss()--weight=torch.from_numpy(np.array([0.1,2.,0.5])).float()
        # 定义交叉熵
        self.criterion1 = Mul_FocalLoss().cuda()
        self.criterion2 = torch.nn.MSELoss(size_average=True).cuda()
        self.criterion3 = DiceLoss(ignore_index=0).cuda()
        self.criterionBCE = torch.nn.BCELoss().cuda()
        self.criterionDICE = BinaryDiceLoss().cuda()
        self.criterion_CE = torch.nn.CrossEntropyLoss().cuda()
        # self.criterion4 = lovasz_softmax()
        self.criterion_convey = Convey_Loss().cuda()
        self.criterion_shapeprior = Shape_prior_Loss().cuda()
        self.augmentation_prob = config.augmentation_prob
        # image size
        self.image_size = config.image_size

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.EM_MOM = config.EM_momentum
        self.EM_iternum = config.EM_iternum

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)  # img_ch为输入通道数，原为3
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'V_Net':
            self.unet = V_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'MDV_Net':
            self.unet = MDV_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'BM_U_Net':
            self.unet = BM_U_Net(img_ch=self.img_ch, output_ch=self.output_ch, iter_num=self.EM_iternum)
        # elif self.model_type == 'Deform_U_Net':
        # 	self.unet = Deform_U_Net(img_ch=self.img_ch, output_ch=self.output_ch, iter_num=self.EM_iternum)
        # elif self.model_type == 'MD_Defm_V_Net':
        # 	self.unet = MD_Defm_V_Net(img_ch=self.img_ch, output_ch=self.output_ch, iter_num=self.EM_iternum)
        elif self.model_type == 'FCN':
            self.unet = FCN(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'MD_UNet':
            self.unet = MD_UNet(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'UNet_double':
            self.unet = UNet_double(in_ch=self.img_ch, out_ch=self.output_ch)

        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.unet.parameters()),
        # 							self.lr, [self.beta1, self.beta2])
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)

        self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)  # 大于0.5的判定为分割区域

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def img_catdist_channel(self, img):
        img_w = img.size()[2]
        img_h = img.size()[3]
        dist_channel = torch.tensor([i for i in range(0, img_w)]).repeat(img_h, 1).t().unsqueeze(0).unsqueeze(0).float()
        # print(dist_channel.dtype)
        # print(img.dtype)
        img_cat = torch.cat((img, dist_channel), 1)

        return img_cat

    def onehot_to_mulchannel(self, GT):

        for i in range(GT.max() + 1):
            if i == 0:
                GT_sg = GT == i
            else:
                GT_sg = torch.cat([GT_sg, GT == i], 1)

        return GT_sg.float()

    def gray2color(self, gray_array, color_map):

        rows, cols = gray_array.shape
        color_array = np.zeros((rows, cols, 3), np.uint8)

        for i in range(0, rows):
            for j in range(0, cols):
                color_array[i, j] = color_map[gray_array[i, j]]

        # color_image = Image.fromarray(color_array)

        return color_array

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        time_begin = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        unet_path = os.path.join(self.model_path, 'md_unet_three_task_0.1_0.9-%s-%d-%.4f-%d-%.4f-%d.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob, self.image_size))
        unet_final_path = os.path.join(self.model_path,
                                       'md_unet_three_task_0.1_0.9-Final_%s-%d-%.4f-%d-%.4f-%d.pkl' % (
                                           self.model_type, self.num_epochs, self.lr, self.num_epochs_decay,
                                           self.augmentation_prob, self.image_size))
        # unet_path = os.path.join(self.model_path, 'set2-Multi-landmark-MD_UNet-270-0.0001-189-0.5000-512.pkl')
        # unet_final_path = os.path.join(self.model_path, 'Finallandmark-landmark-U_Net-220-0.0003-154-0.0000-128.pkl')
        # old_unet_path = os.path.join(self.model_path, 'Deform_U_Net-195-0.0001-136-0.5000-512.pkl')
        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            ##原代码
            # init_weights(self.unet, init_type='kaiming')
            # print('New network initiated')
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        ##

        # model_dict = self.unet.state_dict()
        # pretrained_dict = torch.load(unet_path)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.unet.load_state_dict(model_dict)
        # print('Successfully Loaded partly weight')
        else:
            init_weights(self.unet, init_type='kaiming')
            print('New network initiated')
        ####加载部分预训练模型
        # model = ...
        # model_dict = model.state_dict()
        # pretrained_dict = torch.load(load_name)
        # # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # # 3. load the new state dict
        # model.load_state_dict(model_dict)

        # Train for Encoder
        lr = self.lr
        best_unet_score = 0.
        small_dist = 300000
        avg_cost = np.zeros([self.num_epochs, 2], dtype=np.float32)
        lambda_weight = np.ones([2, self.num_epochs])
        self.unet.iter_num = self.EM_iternum
        writer = SummaryWriter('runs-AOP/AOP')
        T = 2

        # 读取深度文件
        depth_Path = 'D:/py_seg/Landmark-Net/depth.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        pixel_num = self.image_size * 0.715
        a_5 = 0
        a_10 = 0
        length = 0
        jet_map = np.loadtxt('jet_int.txt', dtype=np.int)

        for epoch in range(self.num_epochs):

            self.unet.train(True)
            epoch_loss = 0

            # for p in self.unet.parameters():
            # 	print(p.requires_grad)

            ## DWA平衡梯度
            # index = epoch
            # if index == 0 or index == 1:
            # 	lambda_weight[:, index] = 1.0
            # else:
            # 	w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
            # 	w_2 = avg_cost[index - 1, 1] / avg_cost[index - 2, 1]
            # 	# w_3 = avg_cost[index - 1, 2] / avg_cost[index - 2, 2]
            # 	lambda_weight[0, index] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            # 	lambda_weight[1, index] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
            # 	# lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            DC_1 = 0.
            DC_2 = 0.
            length = 0
            dist_1 = 0.
            dist_2 = 0.
            angle_div = 0.
            class_acc = 0.
            for i, (images, GT, GT_Lmark, GT_class) in enumerate(self.train_loader):
                # GT : Ground Truth

                # images = self.img_catdist_channel(images)
                images = images.to(self.device)

                # GT_1 = F.interpolate(GT, scale_factor=0.5)
                # GT_2 = F.interpolate(GT, scale_factor=0.25)

                GT = GT.to(self.device, torch.long)
                GT_class = GT_class.to(self.device, torch.long)
                GT_Lmark = GT_Lmark.to(self.device)
                # # GT_class = GT_class.to(self.device, torch.long)
                GT_Lmark_d2 = F.interpolate(GT_Lmark, scale_factor=0.5, mode='bilinear')
                GT_Lmark_d4 = F.interpolate(GT_Lmark, scale_factor=0.25, mode='bilinear')
                GT_Lmark_d8 = F.interpolate(GT_Lmark, scale_factor=0.125, mode='bilinear')
                GT_Lmark_d2 = GT_Lmark_d2.to(self.device)
                GT_Lmark_d4 = GT_Lmark_d4.to(self.device)
                GT_Lmark_d8 = GT_Lmark_d8.to(self.device)
                # GT = F.interpolate(GT, scale_factor=0.5,mode='bilinear').long()
                # print(GT.shape)
                # ##多分支输出网络
                # GT_1 = (GT == 1).to(self.device, torch.float)  # 耻骨联合label
                # GT_head = (GT == 2).to(self.device, torch.float)  # 胎头label
                # ####
                # print('44444', GT.max(0))
                GT_sg = self.onehot_to_mulchannel(GT)  # 转换GT编码方式

                # print(GT_sg.shape)
                GT = GT.squeeze(1)

                # GT_1 = GT_1.to(self.device, torch.long)
                # GT_2 = GT_2.to(self.device, torch.long)

                # SR : Segmentation Result
                SR_lm, SR_seg, SR_lm_d2, SR_lm_d4, SR_lm_d8, logsigma, SR_cls = self.unet(images)
                # SR_probs = SR          # SR_probs = F.sigmoid(SR)
                # SR_flat = SR_probs.view(SR_probs.size(0),2)        # SR_flat = SR_probs.view(SR_probs.size(0),-1)
                # GT_flat = GT.view(-1)
                # SR = torch.sigmoid(SR)
                # SR,mu5,mu4,mu3,mu2,mu1= self.unet(images)

                # ##多分支输出网络
                SR_lm = torch.sigmoid(SR_lm)
                SR_lm_d2 = torch.sigmoid(SR_lm_d2)
                SR_lm_d4 = torch.sigmoid(SR_lm_d4)
                SR_lm_d8 = torch.sigmoid(SR_lm_d8)
                # SR_seg = torch.sigmoid(SR)
                # sup1 = torch.sigmoid(sup1)
                # sup2 = torch.sigmoid(sup2)

                # SR_seg = torch.softmax(SR_seg, 1)
                # SR = torch.cat((SR1, SR1, SR2), dim=1)
                # ####
                # print(SR_flat.dtype)
                # print(GT_flat.dtype)
                # SR = self.unet(images)

                # loss = self.criterion(SR,GT)
                # loss_1 = self.criterion2(F.sigmoid(SR[:,1,:,:]).squeeze(1), GT_1)		#Focal loss
                # loss_2 = self.criterion2(F.sigmoid(SR[:,2,:,:]).squeeze(1), GT_2)		#BCE loss
                # loss_3 = self.criterion3(F.sigmoid(SR[:,2,:,:]).squeeze(1), GT_2)		#Dice loss
                # loss_4 = lovasz_softmax(F.sigmoid(SR), GT)
                # print(0.25*loss_1,'|  |',- loss_2.log())
                # loss_ds3 = self.criterion(ds3, GT)
                # loss = 0.25*loss_1 - loss_2.log()
                # loss = loss_2 - (1.-loss_3).log()

                # loss1 = self.criterion(SR, GT)
                # loss2 = self.criterion3(SR,GT_sg)

                # ##多分支输出网络
                # loss1 = self.criterion2(SR1,GT_1) + self.criterion3(SR1,GT_1)
                # loss2 = self.criterion2(SR2,GT_2) + self.criterion3(SR2,GT_2)
                # loss_seg = self.criterion(SR_seg,GT) + self.criterion3(SR_seg,GT_sg)
                # GT_class = np.array(GT_class.cpu)
                loss_cls = self.criterion_CE(SR_cls, GT_class)
                a = torch.tensor([1, 1])
                a = a.to(self.device, torch.long)
                for i in range(1):
                    if GT_class[i] == a[i]:
                        loss_seg = self.criterion3(SR_seg, GT_sg)
                        # loss_seg = self.criterionDICE(SR_seg, GT)

                        # ####
                        loss_lm = self.criterion2(SR_lm, GT_Lmark) + 0.8 * self.criterion2(SR_lm_d2,
                                                                                           GT_Lmark_d2) + 0.8 * self.criterion2(
                            SR_lm_d4, GT_Lmark_d4)
                        loss = (1 / (2 * torch.exp(logsigma[0])) * loss_lm + logsigma[0] / 2) + (
                                1 / (2 * torch.exp(logsigma[1])) * loss_seg + logsigma[1] / 2)
                        loss += 0.1 * loss_cls + 0.9 * loss
                    else:
                        loss = loss_cls
                # if GT_class[0] == a[0]:
                #     loss_seg = self.criterion3(SR_seg, GT_sg)
                #     # loss_seg = self.criterionDICE(SR_seg, GT)
                #
                #     # ####
                #     loss_lm = self.criterion2(SR_lm, GT_Lmark) + 0.8 * self.criterion2(SR_lm_d2,
                #                                                                        GT_Lmark_d2) + 0.8 * self.criterion2(
                #         SR_lm_d4, GT_Lmark_d4)
                #     loss = (1 / (2 * torch.exp(logsigma[0])) * loss_lm + logsigma[0] / 2) + (
                #             1 / (2 * torch.exp(logsigma[1])) * loss_seg + logsigma[1] / 2)
                #     loss += loss_cls
                # loss_cf = self.criterion_class(class_out,GT_class)
                # loss = lambda_weight[0, index]*loss_lm + lambda_weight[1, index]*loss_seg
                # loss = loss_lm + loss_seg
                # loss = self.criterion2(SR_lm, GT_Lmark)

                # loss = loss_cls
                # loss = loss_lm
                # loss = self.criterionDICE(SR_seg, GT)     #+ 0.5 * self.criterionDICE(sup1,GT) + 0.5 * self.criterionDICE(sup2,GT)
                # loss_shape = self.criterion_shapeprior(SR_seg[:, 1:2, :, :],
                #                                        GT_sg[:, 1:2, :, :]) + self.criterion_shapeprior(
                #     SR_seg[:, 2:3, :, :], GT_sg[:, 2:3, :, :])
                # loss_shape = self.criterion_convey(SR_seg) + self.criterion_shapeprior(SR_seg[:, 1:2, :, :],
                #                                                                           GT_sg[:, 1:2, :, :])
                # loss = loss + loss_shape
                epoch_loss += loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                # avg_cost[index, 0] += loss_lm.item()
                # avg_cost[index, 1] += loss_seg.item()
                # print(SR.shape)
                # print(GT.shape)
                # GT = GT.squeeze(1)
                SR_seg = torch.softmax(SR_seg, dim=1)
                acc += get_accuracy(SR_seg, GT)
                SE += get_sensitivity(SR_seg, GT)
                SP += get_specificity(SR_seg, GT)
                PC += get_precision(SR_seg, GT)
                F1 += get_F1(SR_seg, GT)
                JS += get_JS(SR_seg, GT)

                dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg, GT)
                DC += dc_ca0
                DC_1 += dc_ca1
                DC_2 += dc_ca2

                _, preds = SR_cls.max(1)
                class_acc += preds.eq(GT_class).sum().float() / float(SR_cls.size(0))
                # length += images.size(0)
                length += 1
            # dist1, dist2, angle_d = get_diatance(SR_lm, GT_Lmark)
            # dist_1 += dist1
            # dist_2 += dist2
            # angle_div += angle_d
            # _, preds = class_out.max(1)
            # class_acc += preds.eq(GT_class).sum() / class_out.size(0)
            # class_out = class_out > 0.5
            # class_acc += torch.sum(class_out == GT_class) / class_out.size(0)

            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length
            DC_1 = DC_1 / length
            DC_2 = DC_2 / length
            class_acc = class_acc / length
            dist_1 = dist_1 / length
            dist_2 = dist_2 / length
            angle_div = angle_div / length
            # class_acc_sum = class_acc.float() / length
            # writer.add_scalars('md-unet-landmark/train_dist',
            #                    {'dist_1': dist_1, 'dist_2': dist_2, 'angle_div': angle_div}, epoch)
            # writer.add_scalar('md-unet-landmark/loss', epoch_loss, epoch)
            writer.add_scalars('shape_ECA_bt2_dynamic_zjyy_dice_md-unet/train_DCgroup', {'DC': DC,
                                                                                         'DC_SH': DC_1,
                                                                                         'DC_Head': DC_2}, epoch)

            # Print the log info
            print('-' * 100)
            # print('class_acc_sum: %.4f' % class_acc_sum)
            # print('Epoch [%d/%d]loss_lm: %.4f,  loss_seg: %.4f' % (epoch + 1, self.num_epochs, loss_lm, loss_seg))
            # print('Epoch [%d/%d]loss_lm: %.4f,  loss_seg: %.4f, loss_lm-weight: %.4f, loss_seg-weight: %.4f' % (epoch + 1, self.num_epochs, loss_lm, loss_seg, logsigma[0], logsigma[1]))

            print(
                'Epoch [%d/%d], Loss: %.6f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f, class_acc: %.4f' % (
                    epoch + 1, self.num_epochs,
                    epoch_loss,
                    acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2, class_acc))
            # print('[Train-dist] Dist1: %.4f, Dist2: %.4f, Angle_div: %.4f' % (dist_1, dist_2, angle_div))
            # print('Epoch [%d/%d], Loss: %.6f' %(epoch+1, self.num_epochs, epoch_loss))

            # Decay learning rate
            if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))

            # ===================================== Validation ====================================#
            self.unet.train(False)
            self.unet.eval()
            self.unet.iter_num = 5
            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            DC_1 = 0.
            DC_2 = 0.
            length = 0
            dist_1 = 0.
            dist_2 = 0.
            angle_div = 0.
            class_acc = 0.
            # for i, (images, GT, GT_Lmark) in enumerate(self.valid_loader):
            for i, (image, GT, GT_Lmark, num, GT_cor, pixel_length_cm, aop, GT_class) in enumerate(self.valid_loader):
                # images = self.img_catdist_channel(images)
                # degth_cm = landmark_depth[num - 1]
                # pixel_mm = degth_cm * 10 / pixel_num
                images = image.to(self.device)

                GT = GT.to(self.device, torch.long)
                GT_Lmark = GT_Lmark.to(self.device)
                GT_class = GT_class.to(self.device, torch.long)
                # GT = F.interpolate(GT, scale_factor=0.5, mode='bilinear').long()
                # GT_1 = (GT == 1).squeeze(1).to(self.device)

                GT = GT.squeeze(1)

                # SR_lm = self.unet(images)
                SR_lm, SR_seg, _, _, _, _, SR_cls = self.unet(images)
                # SR_seg = self.unet(images)
                # SR ,_,_,_,_,_= self.unet(images)
                # SR,d1,d2,d3 = self.unet(images)
                # ##多分支网络
                SR_lm = torch.sigmoid(SR_lm)
                # SR_seg = torch.sigmoid(SR_seg)
                SR_seg = F.softmax(SR_seg, dim=1)
                # print(class_out)
                # class_out = torch.softmax(class_out,dim=1)
                # print('after:',class_out)
                # SR = torch.cat((SR1, SR1, SR2), dim=1)
                # ####
                # SR = F.softmax(SR,dim=1)		#用lovaszloss时不用加sigmoid
                # GT = GT.squeeze(1)
                acc += get_accuracy(SR_seg, GT)
                SE += get_sensitivity(SR_seg, GT)
                SP += get_specificity(SR_seg, GT)
                PC += get_precision(SR_seg, GT)
                F1 += get_F1(SR_seg, GT)
                JS += get_JS(SR_seg, GT)
                dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg, GT)
                DC += dc_ca0
                # DC += get_DC(SR_seg, GT)
                DC_1 += dc_ca1
                DC_2 += dc_ca2
                # length += images.size(0)
                length += 1
                dist1, dist2, angle_dist = get_diatance(SR_lm, GT_Lmark)
                dist_1 += dist1
                dist_2 += dist2
                angle_div += angle_dist
                _, preds = SR_cls.max(1)
                class_acc += preds.eq(GT_class).sum().float() / float(SR_cls.size(0))
            # _, preds = class_out.max(1)
            # print(preds)
            # class_acc += preds.eq(GT_class).sum()/class_out.size(0)
            # print('class:',class_acc)
            #
            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length
            DC_1 = DC_1 / length
            DC_2 = DC_2 / length
            # dist_1 = dist_1 * pixel_mm
            # dist_2 = dist_2 * pixel_mm
            # unet_score = JS +1.5*DC_1+DC_2
            dist_1 = dist_1 / length
            dist_2 = dist_2 / length
            angle_div = angle_div / length
            class_acc = class_acc / length
            # class_acc_sum = class_acc.float() / length
            # print('class_acc_sum:',class_acc_sum)
            unet_score = dist_1 + dist_2 + angle_div
            # unet_score = DC
            # print('class_acc_sum: %.4f' % class_acc_sum)

            writer.add_scalars('shape_ECA_bt2_dynamic_zjyy_dice_md-unet/valid_dist',
                               {'dist_1': dist_1, 'dist_2': dist_2, 'angle_div': angle_div}, epoch)
            writer.add_scalars('shape_ECA_bt2_dynamic_zjyy_dice_md-unet/valid_DCgroup', {'DC': DC,
                                                                                         'DC_SH': DC_1,
                                                                                         'DC_Head': DC_2}, epoch)
            # writer.add_scalars('se1_bt2_shape_md-unet-landmark/acc', acc,epoch)
            print(
                '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f, class_acc:%.4f' % (
                    acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2, class_acc))
            print('[valid-dist] Dist1: %.4f, Dist2: %.4f, Angle_div: %.4f' % (dist_1, dist_2, angle_div))

            '''
            torchvision.utils.save_image(images.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(SR.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(GT.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
            '''

            # unet_score = small_dist
            # Save Best U-Net model
            # if unet_score > best_unet_score:
            #     best_unet_score = unet_score
            #     best_epoch = epoch
            #     best_unet = self.unet.state_dict()
            #     print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
            #     torch.save(best_unet, unet_path)
            ####landmark部分
            if unet_score < small_dist:
                small_dist = unet_score
                best_epoch = epoch
                best_unet = self.unet.state_dict()
                print('Best %s model score : %.4f' % (self.model_type, small_dist))
                torch.save(best_unet, unet_path)
        torch.save(self.unet.state_dict(), unet_final_path)
        writer.close()
        # ===================================== Test ====================================#
        # del self.net
        # del best_unet
        # self.build_model()
        # self.unet.load_state_dict(torch.load(unet_final_path))
        #
        # self.unet.train(False)
        # self.unet.eval()
        #
        # acc = 0.
        # SE = 0.
        # SP = 0.
        # PC = 0.
        # F1 = 0.  # F1 Score
        # JS = 0.  # Jaccard Similarity
        # DC = 0.  # Dice Coefficient
        # DC_1 = 0.
        # DC_2 = 0.
        # length = 0
        # dist1 = 0
        # dist2 = 0
        # angle_d = 0
        # angle_div = 0
        # for i, (images, GT, GT_Lmark, GT_class, cor_num, GT_cor) in enumerate(self.test_loader):
        #     # degth_cm = landmark_depth[num - 1]
        #     # pixel_mm = degth_cm * 10 / pixel_num
        #     images = images.to(self.device)
        #     GT = GT.to(self.device)
        #     GT = GT.squeeze(1)
        #     GT_Lmark = GT_Lmark.to(self.device)
        #     SR_lm, SR_seg, _, _, _, _, = self.unet(images)
        #     SR_lm = torch.sigmoid(SR_lm)
        #     SR_seg = F.softmax(SR_seg, dim=1)
        #
        #     acc += get_accuracy(SR_seg, GT)
        #     SE += get_sensitivity(SR_seg, GT)
        #     SP += get_specificity(SR_seg, GT)
        #     PC += get_precision(SR_seg, GT)
        #     F1 += get_F1(SR_seg, GT)
        #     JS += get_JS(SR_seg, GT)
        #     dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg, GT)
        #     DC += dc_ca0
        #     DC_1 += dc_ca1
        #     DC += dc_ca2
        #     length += 1
        #     dist1, dist2, angle_d = get_diatance(SR_lm, GT_Lmark)
        #     dist1 += dist1
        #     dist2 += dist2
        #     # dist_1 = dist_1 * pixel_mm
        #     # dist_2 = dist_2 * pixel_mm
        #     angle_div += angle_d
        # acc = acc / length
        # SE = SE / length
        # SP = SP / length
        # PC = PC / length
        # F1 = F1 / length
        # JS = JS / length
        # DC = DC / length
        # DC_1 = DC_1 / length
        # DC_2 = DC_2 / length
        # dist1 = dist1 / length
        # dist2 = dist2 / length
        # angle_div = angle_div / length
        # print('[test_dist] Dist1:%.4f,Dist2:%.4f,Angle_div:%.4f' % (dist1, dist2, angle_div))
        # writer.add_scalars('train_zjyy_test_huaqiao_md-unet-landmark/test_DCgroup', {'DC': DC,
        #                                                                              'DC_SH': DC_1, 'DC_head': DC_2},epoch)
        # writer.add_scalars('train_zjyy_test_huaqiao_md-unet-landmark/test_dist', {'dist1': dist1,
        #                                                                           'dist2': dist2,
        #                                                                           'angle_div': angle_div}, epoch)
        # print('[testing] Acc:%.4f,SE:%.4f,SP:%.4f,PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' %
        #       (acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2))
        #
        # f = open(os.path.join(self.result_path,'result.csv'),'a',encoding='utf-8',newline='')
        # wr = csv.writer(f)

    # for i, (images, GT) in enumerate(self.test_loader):
    #
    #
    # 	# images = self.img_catdist_channel(images)
    # 	images = images.to(self.device)
    # 	GT = GT.to(self.device)
    # 	# GT_1 = (GT == 1).squeeze(1).to(self.device)
    #
    # 	GT = GT.squeeze(1)
    # 	# SR = F.sigmoid(self.unet(images))
    # 	# SR = self.unet(images)
    # 	SR = self.unet(images)
    # 	# SR = F.sigmoid(SR)
    # 	# ##多分支网络
    # 	# SR1 = F.sigmoid(SR_l)
    # 	# SR2 = F.sigmoid(SR_r)
    # 	# SR = torch.cat((SR1, SR1, SR2), dim=1)
    # 	# ####
    # 	acc += get_accuracy(SR,GT)
    # 	SE += get_sensitivity(SR,GT)
    # 	SP += get_specificity(SR,GT)
    # 	PC += get_precision(SR,GT)
    # 	F1 += get_F1(SR,GT)
    # 	JS += get_JS(SR,GT)
    # 	dc_ca0,dc_ca1,dc_ca2 = get_DC(SR,GT)
    # 	# DC += get_DC_binary(SR, GT_1)
    # 	DC += dc_ca0
    # 	DC_1 += dc_ca1
    # 	DC_2 += dc_ca2
    #
    # 	length += 1
    # 	# length += images.size(0)
    #
    # acc = acc/length
    # SE = SE/length
    # SP = SP/length
    # PC = PC/length
    # F1 = F1/length
    # JS = JS/length
    # DC = DC/length
    # DC_1 = DC_1/length
    # DC_2 = DC_2/length
    # unet_score = JS + DC
    #
    # time_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    # f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
    # wr = csv.writer(f)
    # wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,DC_1,DC_2,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob,time_begin,time_end])
    # f.cSR_seg,GT)

    def test(self):
        # unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
        unet_path = os.path.join(self.model_path, 'set1-Multi-landmark-MD_UNet-300-0.0001-210-0.5000-512.pkl')
        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        # writer = SummaryWriter('runsv2/unet')

        self.unet.train(False)
        self.unet.eval()
        # 读取深度文件
        depth_Path = 'D:/py_seg/Landmark-Net/depth.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        pixel_num = self.image_size * 0.715
        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        DC_1 = 0.
        DC_2 = 0.
        dist_1 = 0
        dist_2 = 0
        angle_div_0 = 0
        m_2 = 0
        m_5 = 0
        m_10 = 0
        ag_5 = 0
        ag_10 = 0
        length = 0
        list_l = []
        list_r = []
        list_angle = []
        for i, (images, GT, GT_Lmark, num) in enumerate(self.valid_loader):  # images, GT, GT_Lmark, GT_class,num

            # images = self.img_catdist_channel(images)
            degth_cm = landmark_depth[num - 1]
            pixel_mm = degth_cm * 10 / pixel_num
            images = images.to(self.device)
            # print(images.shape)
            GT = GT.to(self.device)
            GT_Lmark = GT_Lmark.to(self.device)
            GT = GT.squeeze(1)

            # GT[GT == 1] = 3
            # GT[GT == 2] = 1
            # GT[GT == 3] = 2
            # SR = F.sigmoid(self.unet(images))
            # time1 = time.time()
            # SR = self.unet(images)
            SR_lm, SR_seg, _, _, _, _ = self.unet(images)

            SR_lm = torch.sigmoid(SR_lm)
            SR_seg = torch.softmax(SR_seg, 1)

            # SR1 = F.sigmoid(SR1)
            # SR2 = F.sigmoid(SR2)
            # SR = torch.cat((SR1, SR1, SR2), dim=1)
            # time2 = time.time()
            # SR = F.sigmoid(SR)
            acc += get_accuracy(SR_seg, GT)
            SE += get_sensitivity(SR_seg, GT)
            SP += get_specificity(SR_seg, GT)
            PC += get_precision(SR_seg, GT)
            F1 += get_F1(SR_seg, GT)
            JS += get_JS(SR_seg, GT)
            dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg, GT)
            DC += dc_ca0
            DC_1 += dc_ca1
            DC_2 += dc_ca2
            dist1, dist2, angle_div = get_diatance(SR_lm, GT_Lmark)
            dist1 = pixel_mm * dist1
            dist2 = pixel_mm * dist2
            list_l.append(dist1)
            list_r.append(dist2)
            list_angle.append(angle_div)
            dist_1 += dist1
            dist_2 += dist2
            angle_div_0 += angle_div
            # print(time2-time1)
            length += 1
            print('num: %d -- Dist1: %.4f, Dist2: %.4f,Angle_div: %.4f' % (num, dist1, dist2, angle_div))
            if dist1 <= 2:
                m_2 += 1
            if dist2 <= 2:
                m_2 += 1
            if dist1 <= 5:
                m_5 += 1
            if dist2 <= 5:
                m_5 += 1
            if dist1 <= 10:
                m_10 += 1
            if dist2 <= 10:
                m_10 += 1
            if angle_div <= 5:
                ag_5 += 1
            if angle_div <= 10:
                ag_10 += 1
        # cv.imwrite(r'D:\py_seg\U-Net\U-Net_vari\result\pic_output',SR.numpy())
        # 绘制模型
        # with SummaryWriter(comment='Deform_U_Net') as w:				#其中使用了python的上下文管理，with 语句，可以避免因w.close未写造成的问题
        # 	w.add_graph(self.unet,(images,))
        # 显示原图和特征图可视化
        # img_grid = torchvision.utils.make_grid(images, normalize=True, scale_each=True, nrow=2)
        # # # 绘制原始图像
        # writer.add_image('raw img', img_grid, global_step=666)  # j 表示feature map数
        # #
        # for name, layer in self.unet._modules.items():
        #
        # 	images = layer(images)
        # 	images2 = torch.sum(images, dim=1, keepdim=True)
        # 	print(f'{name}')
        # 	# 第一个卷积没有进行relu，要记得加上
        # 	# x = F.relu(x) if 'Conv' in name else x
        # 	if 'Conv' in name or 'Up' in name:
        # 		# x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
        # 		img_grid = torchvision.utils.make_grid(images2.transpose(0,1), normalize=True, scale_each=True, nrow=8)  # normalize进行归一化处理
        # 		writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)
        # #
        # writer.close()
        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        DC_1 = DC_1 / length
        DC_2 = DC_2 / length
        dist_1 = dist_1 / length
        dist_2 = dist_2 / length
        angle_div_0 = angle_div_0 / length
        print(
            '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2))
        print('[Validation] Dist1: %.4f, Dist2: %.4f,Angle_div: %.4f' % (dist_1, dist_2, angle_div_0))
        print('[Median] left: %.4f, right: %.4f,Angle: %.4f' % (
            np.median(list_l), np.median(list_r), np.median(list_angle)))
        print('[mean] left: %.4f, right: %.4f,Angle: %.4f' % (np.mean(list_l), np.mean(list_r), np.mean(list_angle)))
        print('[std] left: %.4f, right: %.4f,Angle: %.4f' % (np.std(list_l), np.std(list_r), np.std(list_angle)))
        print('[point-num] m_2: %d, m_5: %d,m_10: %d' % (m_2, m_5, m_10))
        print('[ag-num] total image: %d  ag_5: %d, ag_10: %d' % (length, ag_5, ag_10))

    # print(
    # 	'[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
    # 	acc, SE, SP, PC, F1, JS, DC, DC_1, DC_2))
    # f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
    # wr = csv.writer(f)
    # wr.writerow(
    # 	[self.model_type, acc, SE, SP, PC, F1, JS, DC,DC_1,DC_2, self.lr, 0, self.num_epochs, self.num_epochs_decay,
    # 	 self.augmentation_prob,self.image_size])
    # f.close()
    def test_output(self):
        unet_path = os.path.join(self.model_path, 'multitask_use2vgg1d_cls1use.pkl')

        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        self.unet.train(False)
        self.unet.eval()
        # writer = SummaryWriter('runs3/aop')

        length = 0
        fps = 20
        size = (512, 384)
        videos_src_path = r'F:\ZDJ\our_data\zhujiang\video\video1'
        videos = os.listdir(videos_src_path)
        videos = filter(lambda x: x.endswith('avi'), videos)  # 用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象

        all_video_list = []
        video_name = []
        ####
        for each_video in videos:
            list_aop = []
            print(each_video)
            videowriter = cv.VideoWriter(each_video, cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)  # 视频的写操作
            # videowriter = cv.VideoWriter(each_video, cv.VideoWriter_fourcc(*'avi'), fps, size)  # 视频的写操作
            # cv.VideoWriter_fourcc 指定写入视频帧编码格式，fps:帧速率
            # get the full path of each video, which will open the video tp extract frames
            each_video_full_path = os.path.join(videos_src_path, each_video)

            cap = cv.VideoCapture(each_video_full_path)  # 视频的读操作
            success = True
            frame_num = 0
            while (success):
                success, frame = cap.read()  # 第二个参数frame表示截取到一帧的图片
                frame_num += 1
                if success == False:
                    break
                frame = frame[54:, 528:1823]
                frame[0:434, 1175:] = 0
                frame[959:1003, 199:1048] = 0
                frame[:, 0:72] = 0
                frame = cv.resize(frame, (512, 384))
                # print('Read a new frame: ', success)
                image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 把GRAY图转换为BGR三通道图--BGR转灰度图
                transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(),
                     # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
                     torchvision.transforms.Normalize((0.5,), (0.5,))])

                image = transform(image)
                image = image.unsqueeze(0)
                # image = Norm_(torch.tensor(image)).unsqueeze(0).unsqueeze(0)

                # image = Norm_(image)
                # print(image.dtype)
                # print(type(image))
                # print(image.shape)
                image = image.to(self.device, torch.float)

                # SR, _, _, _, _, _ = self.unet(image)
                SR_lm, SR_seg, _, _, _, SR_cls = self.unet(image)
                SR = torch.softmax(SR_seg, 1)
                SR_cls = torch.softmax(SR_cls, 1)
                SR = SR > 0.5
                SR = SR.mul(255)
                SR = SR.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
                contours, _ = cv.findContours(cv.medianBlur(SR[:, :, 1], 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                contours2, _ = cv.findContours(cv.medianBlur(SR[:, :, 2], 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                img_result = frame
                maxindex1 = 0
                maxindex2 = 0
                max1 = 0
                max2 = 0
                flag1 = 0
                flag2 = 0

                ##########lm
                # lm_h = (SR_lm[:, 0, :, :] + SR_lm[:, 1, :, :]).mul(255)
                # lm_h = lm_h.detach().cpu().numpy().squeeze(0).astype(np.uint8)
                # lm_j = np.zeros((lm_h.shape[0], lm_h.shape[1], 3)).astype(np.uint8)
                # lm_h = self.gray2color(lm_h, jet_map)
                # lm_h = lm_h[:, :, [2, 1, 0]]
                # num = num.cpu().numpy().squeeze(0)
                # image = (image.mul(127)) + 128
                # image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
                # image1 = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                # image_h = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                ########landmark
                SR1 = SR_lm[0, 0, :, :].cpu().detach()
                SR2 = SR_lm[0, 1, :, :].cpu().detach()
                out_cor1 = np.unravel_index(np.argmax(SR1), SR1.shape)
                out_cor2 = np.unravel_index(np.argmax(SR2), SR2.shape)

                for j in range(len(contours)):
                    if contours[j].shape[0] > max1:
                        maxindex1 = j
                        max1 = contours[j].shape[0]
                    if j == len(contours) - 1:
                        approxCurve = cv.approxPolyDP(contours[maxindex1], 2, closed=True)
                        if approxCurve.shape[0] > 5:
                            # approxCurve = self.edgefliter(approxCurve, turn_angle=60)
                            if approxCurve.shape[0] > 5:
                                img_result = cv.drawContours(img_result, [contours[maxindex1]], 0, (0, 255, 255), 1)
                                img_result = cv.drawContours(img_result, [approxCurve], 0, (0, 0, 255),
                                                             1)  # 得到的耻骨联合区域曲线
                                ellipse = cv.fitEllipse(approxCurve)
                                cv.ellipse(img_result, ellipse, (0, 255, 0), 2)
                                flag1 = 1

                for k in range(len(contours2)):
                    if contours2[k].shape[0] > max2:
                        maxindex2 = k
                        max2 = contours2[k].shape[0]
                    if k == len(contours2) - 1:
                        approxCurve2 = cv.approxPolyDP(contours2[maxindex2], 2, closed=True)
                        if approxCurve2.shape[0] > 5:
                            # approxCurve2 = self.edgefliter(approxCurve2, turn_angle=60)
                            if approxCurve2.shape[0] > 5:
                                img_result = cv.drawContours(img_result, [contours2[maxindex2]], 0, (0, 255, 255),
                                                             1)
                                img_result = cv.drawContours(img_result, [approxCurve2], 0, (255, 0, 0), 1)
                                ellipse2 = cv.fitEllipse(approxCurve2)
                                cv.ellipse(img_result, ellipse2, (0, 255, 0), 2)
                                flag2 = 1

                if flag2 == 1:
                    # img_result, Aod = drawline_AOD(img_result,ellipse2, ellipse,out_cor2,out_cor1)
                    img_result, Aod = drawline_AOD2(img_result, ellipse2, ellipse)
                    list_aop.append(Aod)

                    cv.putText(img_result, "AOP: " + str(round(Aod, 2)) + ""
                                                                          "", (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 255, 255), 1, cv.LINE_AA)  # 在图像上绘制文字
                    cv.line(img_result, (out_cor1[1] - 4, out_cor1[0]), (out_cor1[1] + 4, out_cor1[0]), (0, 255, 0), 2)
                    cv.line(img_result, (out_cor1[1], out_cor1[0] - 4), (out_cor1[1], out_cor1[0] + 4), (0, 255, 0), 2)

                    cv.line(img_result, (out_cor2[1] - 4, out_cor2[0]), (out_cor2[1] + 4, out_cor2[0]), (0, 255, 0), 2)
                    cv.line(img_result, (out_cor2[1], out_cor2[0] - 4), (out_cor2[1], out_cor2[0] + 4), (0, 255, 0), 2)

                    cv.line(img_result, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)
                    cv.line(img_result, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)

                # writer.add_scalar('AOP/'+each_video,  Aod, frame_num)
                else:
                    # writer.add_scalar('AOP/' + each_video, 0, frame_num)
                    pass
                if SR_cls[0, 1] > 0.5:
                    list_aop.append(Aod)
                else:
                    list_aop.append(0)

                videowriter.write(img_result)

            videowriter.release()
            print('[AOP]  median: %.4f  mean: %.4f  std: %.4f' % (
                np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
            all_video_list.append(list_aop)
            video_name.append(each_video)

        aop_csv = pd.DataFrame(index=video_name, data=all_video_list)
        aop_csv.to_csv(r'F:\ZDJ\our_data\zhujiang\video/aop_video1.csv', encoding='gbk')
        cap.release()

    def test_output_pic(self):
        unet_path = os.path.join(self.model_path,
                                 'shape_ECA_bt2_dynamic_zjyy_dice_md-unet-MD_UNet-199-0.0001-139-0.5000-512.pkl')
        ####
        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        self.unet.train(False)
        self.unet.eval()
        # 读取深度文件
        depth_Path = r'F:\ZDJ\MTAFN_510/depth.csv'
        landmark_depth = np.loadtxt(depth_Path, delimiter=',')
        pixel_num = self.image_size * 0.715
        a_5 = 0
        a_10 = 0
        length = 0
        jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        list_aop = []
        list_l = []
        list_r = []
        list_angle_div = []
        list_aop_root_mse = []
        out_aop = []
        true_aop = []
        r2_score_aop = []
        list_num = []
        DC = 0
        DC_1 = 0
        DC_2 = 0
        ASDD_1 = [0, 0]
        ASDD_2 = [0, 0]
        HD_1 = 0
        HD_2 = 0
        length = 0
        aop_1 = 0
        for i, (images, GT, GT_Lmark, num, GT_cor, pixel_length_cm, aop) in enumerate(self.test_loader):

            # for i, (images, GT, GT_Lmark, num, GT_cor) in enumerate(self.valid_loader):
            # num = num - 761
            # degth_cm = landmark_depth[num - 1]
            # pixel_mm = degth_cm * 10 / pixel_num
            image = images
            # images = self.img_catdist_channel(images)
            images = images.to(self.device)
            # GT处理
            GT_Lmark = GT_Lmark.to(self.device)
            GT = GT.to(self.device, torch.long)

            GT_onehot = self.onehot_to_mulchannel(GT)
            GT_onehot = GT_onehot.squeeze(0)
            GT = GT.squeeze(0)  ###!!!!!!!!
            GT_sp = GT_onehot[1, :, :].mul(255)
            GT_sp = GT_sp.cpu().numpy().astype(np.uint8)
            # GT_sp = np.asarray(GT_sp, dtype=np.bool)

            GT_head = GT_onehot[2, :, :].mul(255)
            GT_head = GT_head.detach().cpu().numpy().astype(np.uint8)
            size = (635, 522)
            GT_head_resize = cv.resize(GT_head, size, interpolation=cv.INTER_AREA)

            # GT_head = np.asarray(GT_head, dtype=np.bool)

            # SR1 = SR_seg[0, 1, :, :]
            # SR1 = SR1.mul(255)
            # SR1 = SR1.detach().cpu().numpy().astype(np.uint8)

            # SR = F.sigmoid(self.unet(images))
            t1 = time.time()
            # SR_lm = self.unet(images)
            # SR = F.sigmoid(SR)
            # SR2 = F.sigmoid(SR_r)
            # SR = torch.cat((SR_l, SR_l, SR_r), dim=1)
            # GT_sg = self.onehot_to_mulchannel(GT)

            # SR_lm, SR_seg, _, _, _, _ = self.unet(images)
            SR_lm, SR_seg, _, _, _, _ = self.unet(images)
            # SR_seg=SR_lm
            SR_lm = torch.sigmoid(SR_lm)
            SR_seg = torch.softmax(SR_seg, 1)

            dist1, dist2, angle_div = get_diatance(SR_lm, GT_Lmark)
            dist1 = pixel_length_cm * dist1
            dist2 = pixel_length_cm * dist2
            dist1 += dist1
            dist2 += dist2
            angle_div += angle_div
            list_l.append(dist1)
            list_r.append(dist2)
            list_angle_div.append(angle_div)

            dc_ca0, dc_ca1, dc_ca2 = get_DC(SR_seg, GT)
            # DC += get_DC(SR_seg,GT_head)
            DC += dc_ca0
            DC_1 += dc_ca1
            DC_2 += dc_ca2

            # SR = SR/torch.max(SR)
            # print('~',torch.max(SR))
            SR_seg = SR_seg > 0.5
            SR_sp = SR_seg[:, 1, :, :].mul(255)
            SR_sp = SR_sp.detach().cpu().numpy().squeeze(0).astype(np.uint8)

            # lm_h = GT_Lmark[:, 1, :, :].mul(255)
            # lm_h = lm_h.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # lm_h = self.gray2color(lm_h, jet_map)
            # lm_h = lm_h[:, :, [2, 1, 0]]

            lm_h = (GT_Lmark[:, 0, :, :] + GT_Lmark[:, 1, :, :]).mul(255)
            lm_h = lm_h.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            lm_h = self.gray2color(lm_h, jet_map)
            # lm_j = np.zeros((lm_h.shape[0], lm_h.shape[1], 3)).astype(np.uint8)
            # lm_h = self.gray2color(lm_h, jet_map)
            # lm_h = lm_h[:, :, [2, 1, 0]]

            # SR_lm_1 = SR_lm[:, 1, :, :].mul(255)
            # SR_lm_1 = SR_lm_1.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # SR_lm_1 = self.gray2color(SR_lm_1, jet_map)
            # SR_lm_1 = SR_lm_1[:, :, [2, 1, 0]]
            #
            # SR_lm_0 = SR_lm[:, 0, :, :].mul(255)
            # SR_lm_0 = SR_lm_0.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # SR_lm_0 = self.gray2color(SR_lm_0, jet_map)
            # SR_lm_0 = SR_lm_0[:, :, [2, 1, 0]]
            # plt.figure()
            # plt.imshow(lm_h)
            # plt.show()
            # pass
            lm_p = (SR_lm[:, 0, :, :] + SR_lm[:, 1, :, :]).mul(255)
            lm_p = lm_p.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            lm_p = self.gray2color(lm_p, jet_map)
            lm_p = lm_p[:, :, [2, 1, 0]]
            # plt.figure()
            # plt.imshow(lm_p)
            # plt.show()
            # pass
            # lm_p = self.gray2color(lm_h, jet_map)
            # lm_p = lm_h[:, :, [2, 1, 0]]

            t2 = time.time()
            num = num.cpu().numpy().squeeze(0)
            list_num.append(num)
            image = (image.mul(127)) + 128
            image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
            image1 = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            image_h = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            ######## landmark
            # SR1 = SR_lm[0, 0, :, :].cpu().detach()
            # SR2 = SR_lm[0, 1, :, :].cpu().detach()
            GT1 = GT_Lmark[0, 0, :, :].cpu().detach()
            GT2 = GT_Lmark[0, 1, :, :].cpu().detach()
            # GT3 = GT_Lmark[0, 2, :, :].cpu().detach()
            out_cor1 = np.unravel_index(np.argmax(GT1), GT1.shape)
            out_cor2 = np.unravel_index(np.argmax(GT2), GT2.shape)
            # out_cor3 = np.unravel_index(np.argmax(GT3), GT3.shape)
            SR1 = SR_lm[0, 0, :, :].cpu().detach()
            SR2 = SR_lm[0, 1, :, :].cpu().detach()
            out_cor_1 = np.unravel_index(np.argmax(SR1), SR1.shape)
            out_cor_2 = np.unravel_index(np.argmax(SR2), SR2.shape)

            GT_cor1 = GT_cor[0][0]
            GT_cor1 = np.array(GT_cor1.numpy(), dtype=int)

            # GT_cor1 = GT_cor1[0]
            GT_cor2 = GT_cor[1][0]
            GT_cor2 = np.array(GT_cor2.numpy(), dtype=int)
            # GT_cor2 = GT_cor2[0][0]
            # GT_cor1[0] = GT_cor_l[0][0]
            # GT_cor1[1] = GT_cor_l[0][1]
            # GT_cor2[0] = GT_cor_r[0][0]
            # GT_cor2[1] = GT_cor_r[0][1]
            # out_cor3 = GT_cor[2]
            # print(out_cor1.shape)

            ###
            # cv.line(image1, (out_cor1[1] - 4, out_cor1[0]), (out_cor1[1] + 4, out_cor1[0]), (0, 255, 0), 2)
            # cv.line(image1, (out_cor1[1], out_cor1[0] - 4), (out_cor1[1], out_cor1[0] + 4), (0, 255, 0), 2)
            #
            # cv.line(image1, (out_cor2[1] - 4, out_cor2[0]), (out_cor2[1] + 4, out_cor2[0]), (0, 255, 0), 2)
            # cv.line(image1, (out_cor2[1], out_cor2[0] - 4), (out_cor2[1], out_cor2[0] + 4), (0, 255, 0), 2)
            #
            # cv.line(image1, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)

            #######显示标签AOP

            # cv.line(image1, (GT_cor1[1], GT_cor1[0]), (GT_cor2[1], GT_cor2[0]), (0, 255, 0), 1)

            # image_3 = cv.cvtColor(image,cv.COLOR_GRAY2BGR)		#把GRAY图转换为BGR三通道图
            # # print(image)
            # # print(image.shape)
            # # print(type(image))
            # # print(SR.shape)
            # # print(type(SR))
            # # cv.imshow(SR)
            # # cv.waitkey(0)
            # img_result = image_3
            # contours, _ = cv.findContours(SR[:, :, 1], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # contours2, _ = cv.findContours(SR[:, :, 2], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # maxindex1 = 0
            # maxindex2 = 0
            # max1=0
            # max2=0
            # # for j in range(len(contours)):
            # # 	if contours[j].shape[0] > max1:
            # # 		maxindex1 = j
            # # 		max1 = contours[j].shape[0]
            # #
            # # approxCurve = cv.approxPolyDP(contours[maxindex1], 2, closed=True)
            # #
            # # img_result = cv.drawContours(image_3, [approxCurve], 0, (0, 0, 255), 1)		#得到的耻骨联合区域曲线
            # # ellipse = cv.fitEllipse(approxCurve)
            # # cv.ellipse(img_result,ellipse,(0,255,0),2)
            # #
            # # for k in range(len(contours2)):
            # # 	if contours2[k].shape[0] > max2:
            # # 		maxindex2 = k
            # # 		max2 = contours2[k].shape[0]
            # #
            # # approxCurve2 = cv.approxPolyDP(contours2[maxindex2], 2, closed=True)
            # # ellipse2 = cv.fitEllipse(approxCurve2)
            # # cv.ellipse(img_result, ellipse2, (0, 255, 0), 2)
            # for j in range(len(contours)):
            # 	if contours[j].shape[0] > max1:
            # 		maxindex1 = j
            # 		max1 = contours[j].shape[0]
            #
            # approxCurve = cv.approxPolyDP(contours[maxindex1], 2, closed=True)
            # if approxCurve.shape[0] > 10:
            # 	img_result = cv.drawContours(img_result, [approxCurve], 0, (0, 0, 255), 1)  # 得到的耻骨联合区域曲线
            # 	ellipse = cv.fitEllipse(approxCurve)
            # 	cv.ellipse(img_result, ellipse, (0, 255, 0), 2)
            #
            # for k in range(len(contours2)):
            # 	if contours2[k].shape[0] > max2:
            # 		maxindex2 = k
            # 		max2 = contours2[k].shape[0]
            #
            # approxCurve2 = cv.approxPolyDP(contours2[maxindex2], 2, closed=True)
            #
            # if approxCurve2.shape[0] > 10:
            # 	img_result = cv.drawContours(img_result, [approxCurve2], 0, (255, 0, 0), 1)
            # 	ellipse2 = cv.fitEllipse(approxCurve2)
            # 	cv.ellipse(img_result, ellipse2, (0, 255, 0), 2)
            # img_result = cv.drawContours(img_result, [approxCurve2], 0, (255, 0, 0), 1)
            # print(image.shape)
            image_lm = image1
            # print(t2 - t1)
            # cv.imwrite(r'D:\py_seg\Landmark-Net\result\exp_output\GTaop/' + str(num).zfill(4) + 'aop.png',image1)
            # print(sp_j.shape)
            # img_add_sp = cv.addWeighted(image, 0.6, lm_p, 0.4, 0)

            # img_add_h = cv.addWeighted(img_add_sp, 0.8, SR_h, 0.2, 0)
            # img_lm = cv.addWeighted(image_h, 0.6, lm_h, 0.4, 0)

            # img_lm = cv.addWeighted(lm_h, 0.4, image_h, 0.6, 0)
            # SR_lm = cv.addWeighted(image_h, 0.6, lm_p, 0.4,  0)
            # SR_lm = cv.addWeighted(SR_lm, 0.8, SR_lm_1, 0.2, 0)

            # img_label_coor = cv.addWeighted(image, 0.6, GT_sp, 0.4, 0)
            # img_label_coor = cv.addWeighted(img_label_coor, 0.6, GT_head, 0.4, 0)
            # img_label_coor = cv.addWeighted(img_label_coor, 0.6, lm_h, 0.4, 0)

            # image_single = np.copy(image).squeeze(2)
            # image_color = self.gray2color(image_single, jet_map)
            # lm_p = lm_p[:, :, [2, 1, 0]]
            # img_label_coor = cv.addWeighted(image_color, 0.6, lm_h, 0.4, 0)

            # SR_lm = cv.addWeighted(image, 0.7, GT_sp, 0.3, 0)
            image_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            SR_lm = cv.addWeighted(image_color, 0.6, lm_p, 0.4, 0)
            img_label_coor = cv.addWeighted(image_color, 0.6, lm_h, 0.4, 0)
            # plt.figure()
            # plt.imshow(img_label_coor)
            # # plt.imshow(SR_lm)
            # plt.show()
            # pass
            t3 = time.time()
            # print(ngt2.shape)
            SR_seg[:, 0, :, :] = 0
            SR_seg_all = SR_seg.mul(255)
            SR_seg_all = SR_seg_all.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
            # SR_lm = SR_lm > 0.5
            # SR_lm = SR_lm.mul(255)
            # SR_lm = SR_lm.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
            # SR = cv.cvtColor(SR, cv.COLOR_BGR2GRAY)
            # ngt2 = ngt2.mul(255)
            # ngt2 = ngt2.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)

            SR1 = SR_seg[0, 1, :, :]
            SR2 = SR_seg[0, 2, :, :]
            # print('sr1', SR1.shape)
            SR1 = SR1.mul(255)
            SR1 = SR1.detach().cpu().numpy().astype(np.uint8)
            # SR1 = SR1.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8) #old

            SR2 = SR2.mul(255)
            SR2 = SR2.detach().cpu().numpy().astype(np.uint8)
            # SR2 = SR2.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)  # old

            # contours, _ = cv.findContours(SR1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # contours2, _ = cv.findContours(SR2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            # contours, _ = cv.findContours(cv.medianBlur(SR1, 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)# 一开始默认的参数
            # contours2, _ = cv.findContours(cv.medianBlur(SR2, 11), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contours, _ = cv.findContours(cv.medianBlur(SR1, 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contours2, _ = cv.findContours(cv.medianBlur(SR2, 51), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # contours_GT_head, _ = cv.findContours(cv.medianBlur(GT_head, 1), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contours_GT_head_resize, _ = cv.findContours(cv.medianBlur(GT_head_resize, 1), cv.RETR_EXTERNAL,
                                                         cv.CHAIN_APPROX_NONE)
            contours_GT_head_resize = contours_GT_head_resize[::-1]
            # contours_GT_sp, _ = cv.findContours(cv.medianBlur(GT_sp, 1), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            img_result = image1
            maxindex1 = 0
            maxindex2 = 0
            maxindex3 = 0
            aop_mse = 0
            max1 = 0
            max2 = 0
            max3 = 0
            pre_sp = np.zeros_like(SR1)
            pre_hd = np.zeros_like(SR2)
            flag1 = 0
            flag2 = 0

            for j in range(len(contours)):
                if contours[j].shape[0] > max1:
                    maxindex1 = j
                    max1 = contours[j].shape[0]
                if j == len(contours) - 1:
                    approxCurve = cv.approxPolyDP(contours[maxindex1], 2, closed=True)
                    if approxCurve.shape[0] > 5:
                        # approxCurve = self.edgefliter(approxCurve, turn_angle=60)
                        # img_result = cv.drawContours(img_result, [approxCurve], 0, (0, 0, 255), 1)  # 得到的耻骨联合区域曲线
                        ellipse = cv.fitEllipse(approxCurve)
                        # cv.ellipse(img_result, ellipse, (0, 255, 0), 2)
                        flag1 = 1
            if len(contours) > 0:
                cv.fillPoly(pre_sp, [contours[maxindex1]], 1)  # 填充内部
            pre_sp = np.asarray(pre_sp, dtype=np.bool)

            for k in range(len(contours2)):
                if contours2[k].shape[0] > max2:
                    maxindex2 = k
                    max2 = contours2[k].shape[0]
                if k == len(contours2) - 1:
                    approxCurve2 = cv.approxPolyDP(contours2[maxindex2], 2, closed=True)
                    if approxCurve2.shape[0] > 5:
                        approxCurve2 = self.edgefliter(approxCurve2, turn_angle=90)
                        # img_result = cv.drawContours(img_result, [approxCurve2], 0, (255, 0, 0), 1) # 得到胎头区域曲线
                        ellipse2 = cv.fitEllipse(approxCurve2)
                        cv.ellipse(img_result, ellipse2, (0, 255, 0), 2)
                        flag2 = 1

            # for m in range(len(contours_GT_head)):
            #     if contours_GT_head[m].shape[0] > max2:
            #         maxindex3 = m
            #         max3 = contours_GT_head[m].shape[0]
            #     if m == len(contours_GT_head) - 1:
            #         approxCurve3 = cv.approxPolyDP(contours_GT_head[maxindex3], 2, closed=True)
            #         if approxCurve3.shape[0] > 5:
            #             # approxCurve2 = self.edgefliter(approxCurve2, turn_angle=60)
            #             # img_result = cv.drawContours(img_result, [approxCurve2], 0, (255, 0, 0), 1)
            #             ellipse3 = cv.fitEllipse(approxCurve3)
            #             # cv.ellipse(img_result, ellipse3, (0, 255, 0), 2)
            #             flag3 = 1

            # # 耻骨联合轮廓 豪斯多夫距离
            # surface_distances_sp = surface_distance.compute_surface_distances(
            #     GT_sp, pre_sp, spacing_mm=(1, 1))
            # sp_asdd = surface_distance.compute_average_surface_distance(surface_distances_sp)
            # SP_hausdorff_100 = surface_distance.compute_robust_hausdorff(surface_distances_sp, 100)
            # if len(contours) == 0:
            #     print('1111111111111111111111111111111111111111111111111111111111111111111111111111111')
            #     SP_hausdorff_100 = 1
            #     ASDD_1[0] += 1
            #     ASDD_1[1] += 1
            # else:
            #     ASDD_1[0] += sp_asdd[0]
            #     ASDD_1[1] += sp_asdd[1]
            # HD_1 += SP_hausdorff_100
            # # 胎头轮廓 豪斯多夫距离
            # surface_distances_head = surface_distance.compute_surface_distances(
            #     GT_head, pre_hd, spacing_mm=(1, 1))
            # head_asdd = surface_distance.compute_average_surface_distance(surface_distances_head)
            # Head_hausdorff_100 = surface_distance.compute_robust_hausdorff(surface_distances_head, 100)
            # ASDD_2[0] += head_asdd[0]
            # ASDD_2[1] += head_asdd[1]
            # HD_2 += Head_hausdorff_100
            if flag2 == 1:
                img_result, Aod = drawline_AOD(img_result, ellipse2, ellipse2, out_cor_2, out_cor_1)
                img_result[(out_cor1[0] - 2):(out_cor1[0] + 2), (out_cor1[1] - 2):(out_cor1[1] + 2), :] = [255, 0, 0]
                img_result[(out_cor2[0] - 2):(out_cor2[0] + 2), (out_cor2[1] - 2):(out_cor2[1] + 2), :] = [0, 0, 255]

                # plt.figure()
                # plt.imshow(img_result)
                # plt.show()
                # pass
                out_aop.append(Aod)
                # aop = self.calculate_true_aop_3point(num)

                # aop, GT_cor3 = self.calculate_true_aop_2point(contours_GT_head_resize, GT_cor2, GT_cor1, image1)
                aop = aop
                # img_result, aop = drawline_AOD(img_result, ellipse3, ellipse3, GT_cor2, GT_cor1)
                # aop, GT_cor3 = self.calculate_true_aop_2point(contours_GT_head_resize, GT_cor1, GT_cor2)
                # print('---------', aop)
                list_aop.append(abs(Aod - aop))
                # aop_mse += (Aod - aop) ** 2
                if abs(Aod - aop) <= 5:
                    a_5 += 1
                if abs(Aod - aop) <= 10:
                    a_10 += 1
                length += 1
                # aop = aop.detach().cpu().numpy()
                # aop = aop.cpu().numpy()

                aop_1 = str(np.round(aop[0], 2).__array__())
                true_aop.append(aop_1)
                print('---------', aop_1)
                cv.putText(img_result, "AOD: " + str(round(Aod, 2)), (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1, cv.LINE_AA)
                cv.putText(img_result, "aop: " + aop_1, (50, 80), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1, cv.LINE_AA)

            image11 = np.copy(image1)
            # pred_head = np.copy(SR2)
            size = (635, 522)
            image11 = cv.resize(image11, size, interpolation=cv.INTER_AREA)
            # pred_head = cv.resize(pred_head, size, interpolation=cv.INTER_AREA)
            # image11[(GT_cor1[0] - 2):(GT_cor1[0] + 2), (GT_cor1[1] - 2):(GT_cor1[1] + 2), :] = [255, 0, 0]
            # image11[(GT_cor2[0] - 2):(GT_cor2[0] + 2), (GT_cor2[1] - 2):(GT_cor2[1] + 2), :] = [0, 0, 255]
            # image11[(GT_cor3[0] - 2):(GT_cor3[0] + 2), (GT_cor3[1] - 2):(GT_cor3[1] + 2), :] = [0, 255, 0]

            # cv.line(image11, (GT_cor1[0], GT_cor1[1]), (GT_cor2[0], GT_cor2[1]), (0, 255, 0), 1)
            #
            # cv.line(image11, (GT_cor2[0], GT_cor2[1]), (GT_cor3[0], GT_cor3[1]), (0, 255, 0), 1)
            # image11 = cv.drawContours(image11, contours_GT_head_resize, 0, (0, 0, 255), 1)
            # plt.figure()
            # plt.imshow(image11)
            # plt.show()
            # pass

            # print(t2-t1)
            #
            # print('[aop] aopd: %.4f, median: %.4f, mean: %.4f,std: %.4f' % (abs(Aod-aop),np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
            # print('[std] left: %.4f, right: %.4f,Angle: %.4f' % (np.std(list_l), np.std(list_r), np.std(list_angle)))
            # cv.imwrite(r'D:\py_seg\Landmark-Net\result\exp_output\multiwoafm\seg/' + str(num).zfill(4) + 'seg.png', sp_j)
            # cv.imwrite(r'F:\ZDJ\MTAFN_510\result\multiafm\locjet_2/' + str(num).zfill(4) + 'lmjet_2.png',
            #            img_lm)
            # cv.imwrite(r'F:\ZDJ\MTAFN_510\result\multiafm\SR_lm/' + str(num).zfill(4) + 'SR_lm.png',
            #            SR_lm)
            cv.imwrite(r'F:\ZDJ\MTAFN_510\result\multiafm - MD_Unet\loccor/' + str(num).zfill(4) + 'lmcor.png',
                       image1)
            cv.imwrite(r'F:\ZDJ\MTAFN_510\result\multiafm - MD_Unet\true_aop/' + str(num).zfill(4) + 'lmcor.png',
                       image11)
            cv.imwrite(r'F:\ZDJ\MTAFN_510\result\multiafm - MD_Unet\SR_seg/' + str(num).zfill(4) + 'seg.png',
                       SR_seg_all)
            # cv.imwrite(r'F:\ZDJ\MTAFN_510\result\multiafm\img_lm/' + str(num).zfill(4) + '_r.png', SR_lm)
            cv.imwrite(r'F:\ZDJ\MTAFN_510\result\multiafm - MD_Unet\SR_lm/' + str(num) + '_sp.png', SR_lm)
            cv.imwrite(r'F:\ZDJ\MTAFN_510\result\multiafm - MD_Unet\img_result/' + str(num) + '_result_noaop.png',
                       img_result)
            cv.imwrite(r'F:\ZDJ\MTAFN_510\result\multiafm - MD_Unet\img_label_coor/' + str(num) + '.png',
                       img_label_coor)
        # cv.imwrite(r'D:/py_seg/U-Net/U-Net_vari/result/pic_output/landmark2/' + str(i) + '_r.png', SR_h)
        # cv.imwrite(r'D:/py_seg/U-Net/U-Net_vari/result/pic_output/' + str(i) + '_sp.png', img_add_sp)
        # cv.imwrite(r'D:\py_seg\Landmark-Net\result\exp_output\single\aop/' + str(num).zfill(4) + 'mulaop.png', img_result)
        # cv.imwrite(r'D:/py_seg/U-Net/U-Net_vari/result/pic_output/' + str(i) + '_result_noaop.png', img_result)
        DC = DC / length
        DC_1 = DC_1 / length
        DC_2 = DC_2 / length

        HD_1 = HD_1 / length
        HD_2 = HD_2 / length
        dist_1 = dist1 / length
        dist_2 = dist2 / length
        angle_div = angle_div / length
        # HD = HD_1 + HD_2
        # ASDD_sp = (ASDD_1[0] / length + ASDD_1[1] / length) / 2
        # ASDD_head = (ASDD_2[0] / length + ASDD_2[1] / length) / 2
        # r2_score_aop = r2_score(np.array(true_aop), np.array(out_aop))
        # aop_root_mse = mean_squared_error(np.array(true_aop), np.array(out_aop), squared=False)
        print(
            '[Validation] DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
                DC, DC_1, DC_2))
        print(
            '[aopall] median: %.4f, mean: %.4f,std: %.4f' % (np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
        print(
            '[list_l] median: %.4f, mean: %.4f,std: %.4f' % (np.median(list_l), np.mean(list_l), np.std(list_l)))
        print(
            '[list_r] median: %.4f, mean: %.4f,std: %.4f' % (np.median(list_r), np.mean(list_r), np.std(list_r)))
        print(
            '[list_angle_div] median: %.4f, mean: %.4f,std: %.4f' % (
                np.median(list_angle_div), np.mean(list_angle_div), np.std(list_angle_div)))
        # print('[valid-dist] Dist1: %.4f, Dist2: %.4f, Angle_div: %.4f' % (dist_1, dist_2, angle_div))
        # print(
        #     '[list_aop_root_mse] aop_root_mse: ', aop_root_mse)
        # print(
        #     '[r2_score_aop] r2_score_aop: ', r2_score_aop)

        # aop_csv = pd.DataFrame(data=out_aop)
        # aop_csv.to_csv(r'F:\ZDJ\MTAFN_510\result/out_aop.csv', encoding='gbk')
        # data = {"ACC": np.array(A_acc), "SE": np.array(SE), "SP": np.array(SP), "PC": np.array(PC), "F1": np.array(F1),
        #         "JS": np.array(JS), "DC": np.array(DC), "DC_sp": np.array(DC_1), "DC_hd": np.array(DC_2),
        #         "HD_1": np.array(HD_1), "HD_2": np.array(HD_2), "ASDD_1_1": np.array(ASDD_1)[:, 0],
        #         "ASDD_1_2": np.array(ASDD_1)[:, 1], "ASDD_2_1": np.array(ASDD_2)[:, 0],
        #         "ASDD_2_2": np.array(ASDD_2)[:, 1], "img_num": img_indexlist}
        data = {"true_aop": np.array(true_aop), "img_num": list_num}
        true_aop_csv = pd.DataFrame(data=data)
        true_aop_csv.to_csv(r'F:\ZDJ\MTAFN_510\result/true_aop_test.csv', encoding='gbk')

        data_1 = {"out_aop": np.array(out_aop), "img_num": list_num}
        out_aop_csv = pd.DataFrame(data=data_1)
        out_aop_csv.to_csv(r'F:\ZDJ\MTAFN_510\result/out_aop_test.csv', encoding='gbk')
        # true_aop_csv = pd.DataFrame(data=list_angle_div)
        # true_aop_csv.to_csv(r'F:\ZDJ\MTAFN_510\result/angle_div.csv', encoding='gbk')
        # true_aop_csv = pd.DataFrame(data=list_l)
        # true_aop_csv.to_csv(r'F:\ZDJ\MTAFN_510\result/list_l.csv', encoding='gbk')

    def test_output_pic_hc(self):
        unet_path = os.path.join(self.model_path, '128-fetalhead-bisenet-BiSeNet-199-0.0001-139-0.5000-128.pkl')
        save_path = r'D:\py_seg\Landmark-Net\result\pic_output\fetalhead_result\128bisenet/'
        ####
        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        self.unet.train(False)
        self.unet.eval()

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        DC_1 = 0.
        DC_2 = 0.
        length = 0
        # jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
        # list_aop = []
        for i, (images, GT, img_name) in enumerate(self.valid_loader):
            img_name = img_name[0]
            print(img_name)

            image = images
            print(images.shape)
            # images = self.img_catdist_channel(images)
            images = images.to(self.device)
            GT = GT.to(self.device, torch.long)
            # SR = F.sigmoid(self.unet(images))
            # t1 = time.time()
            SR_seg = self.unet(images)
            SR_seg = torch.sigmoid(SR_seg)
            # SR2 = F.sigmoid(SR_r)
            # SR = torch.cat((SR_l, SR_l, SR_r), dim=1)
            # GT_sg = self.onehot_to_mulchannel(GT)

            # SR_seg=SR_lm
            # SR_lm = torch.sigmoid(SR_lm)
            # SR_seg = torch.softmax(SR_seg, 1)

            acc += get_accuracy(SR_seg, GT)
            SE += get_sensitivity(SR_seg, GT)
            SP += get_specificity(SR_seg, GT)
            PC += get_precision(SR_seg, GT)
            F1 += get_F1(SR_seg, GT)
            JS += get_JS(SR_seg, GT)
            DC += get_DC(SR_seg, GT)
            length += 1

            SR_seg = SR_seg > 0.5
            # SR_sp = SR[:, 1, :,:].mul(255)
            # SR_sp = SR_sp.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # sp_j = np.zeros((SR_sp.shape[0], SR_sp.shape[1],3)).astype(np.uint8)
            # sp_j[:,:,2] = SR_sp
            # SR_h = SR[:, 2, :, :].mul(255)
            # SR_h = SR_h.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # hd_j = np.zeros((SR_h.shape[0], SR_h.shape[1],3)).astype(np.uint8)
            # hd_j[:,:,2] = SR_h
            # sp_j[:, :, 1] = SR_h
            # GT_sg1 = GT_sg[:, 1, :,:].mul(255)
            # GT_sg1 = GT_sg1.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # GT_sg2 = GT_sg[:, 2, :, :].mul(255)
            # GT_sg2 = GT_sg2.detach().cpu().numpy().squeeze(0).astype(np.uint8)
            # GT_j = np.zeros((GT_sg1.shape[0], GT_sg1.shape[1], 3)).astype(np.uint8)
            # GT_j[:, :, 2] = GT_sg1
            # GT_j[:, :, 1] = GT_sg2
            SR_seg = SR_seg.mul(255)
            SR_seg = SR_seg.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
            # lm_j = np.zeros((lm_h.shape[0], lm_h.shape[1], 3)).astype(np.uint8)
            # SR_seg = SR_seg[:,:,[2,1,0]]
            GT = GT.mul(255)
            GT = GT.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
            image = (image.mul(127)) + 128
            image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
        # image1 = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        # image_h = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        ########landmark
        # SR1 = SR_lm[0, 0, :, :].cpu().detach()
        # SR2 = SR_lm[0, 1, :, :].cpu().detach()
        # GT1 = GT_Lmark[0, 0, :, :].cpu().detach()
        # GT2 = GT_Lmark[0, 1, :, :].cpu().detach()
        # GT3 = GT_Lmark[0, 2, :, :].cpu().detach()

        #
        # print('[aop] aopd: %.4f, median: %.4f, mean: %.4f,std: %.4f' % (abs(Aod-aop),np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
        # print('[std] left: %.4f, right: %.4f,Angle: %.4f' % (np.std(list_l), np.std(list_r), np.std(list_angle)))
        # cv.imwrite(r'D:\py_seg\Landmark-Net\result\exp_output\multiwoafm\seg/' + str(num).zfill(4) + 'seg.png', sp_j)
        # cv.imwrite(r'D:\py_seg\Landmark-Net\result\exp_output\single\locjet/' + str(num).zfill(4) + 'lmjet.png', img_lm)

        # cv.imwrite(r'D:/py_seg/U-Net/U-Net_vari/result/pic_output/landmark2/' + str(i) + '_r.png', SR_h)
        # cv.imwrite(save_path + img_name + '_img.png', image)
        # cv.imwrite(save_path + img_name + '_gt.png', GT)
        # cv.imwrite(save_path + img_name + '_sr.png', SR_seg)
        # cv.imwrite(r'D:\py_seg\Landmark-Net\result\exp_output\single\aop/' + str(num).zfill(4) + 'mulaop.png', img_result)
        # cv.imwrite(r'D:/py_seg/U-Net/U-Net_vari/result/pic_output/' + str(i) + '_result_noaop.png', img_result)
        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length

        print(
            '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC_1: %.4f, DC_2: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC, DC, DC))
        # I=cv.imread(r'D:\py_seg\U-Net\U-Net_vari\dataset\test\ATD_0004.png')
        # print('[aopall] median: %.4f, mean: %.4f,std: %.4f' % (np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
        # print('[ag-num] total image: %d  ag_5: %d, ag_10: %d' % (length, a_5, a_10))
        net_input = torch.rand(1, 1, images.shape[2], images.shape[3]).to(self.device)
        print(net_input.size())
        flops, params = profile(self.unet, inputs=(net_input,))
        print("flops: %.f  parmas: %.f", (flops / 1000000000, params / (1024 * 1024)))

    def saveONNX(self, filepath, model_name):
        '''
        保存ONNX模型
        :param model: 神经网络模型
        :param filepath: 文件保存路径
        '''
        unet_path = os.path.join(self.model_path, model_name)
        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

        self.unet.train(False)
        self.unet.eval()
        # 神经网络输入数据类型
        dummy_input = torch.randn(1, 1, 384, 512, device='cuda')
        torch.onnx.export(self.unet, dummy_input, filepath, verbose=True)

    def edgefliter(self, approxCurve, turn_angle=90):
        maxindex1 = 0
        max1 = 0
        angle_list = []
        angle_sign = []
        seg_index = [0, ]
        ellip_group = []
        seged_edgeset = []
        # img_med = cv.medianBlur(img, 7)	# 滤波核7，9

        # contours, _ = cv.findContours(img_med, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        for i in range(1, len(approxCurve) - 1):
            lineseg1 = (np.array(approxCurve[i]) - np.array(approxCurve[i - 1])).squeeze(0)
            lineseg2 = (np.array(approxCurve[i + 1]) - np.array(approxCurve[i])).squeeze(0)
            L_1 = np.sqrt(lineseg1.dot(lineseg1))
            L_2 = np.sqrt(lineseg2.dot(lineseg2))

            pi_dist = lineseg1.dot(lineseg2) / (L_1 * L_2)
            if abs(pi_dist) >= 1.:
                # pi_dist = (pi_dist / abs(pi_dist)) * 0.999  # 防止arccos输入超出范围[-1,1]  判断是否超出1
                print('pi_dist value overflow')
            angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi
            angle_list.append(angle_dist)
            a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])  # 不用计算出角度再判断正负，直接用叉乘值判断正负？
            angle_sign.append(a_sign)

        for i in range(len(angle_list)):
            flag1 = 0
            if angle_list[i] >= turn_angle:
                flag1 = 1
            if i > 0 and angle_sign[i] != angle_sign[i - 1]:
                flag1 = 1
            if flag1 == 1:
                seg_index.append(i + 1)
        ##  如果曲线封闭，直接拟合
        if (len(seg_index) - 1) == 0:
            return approxCurve

        for i in range(len(seg_index) - 1):
            seged_edgeset.append(approxCurve[seg_index[i]:seg_index[i + 1] + 1])
        # lineseg1 = np.array(approxCurve[-1]) - np.array(approxCurve[-1])
        # lineseg2 = np.array(approxCurve[0]) - np.array(approxCurve[i])

        # 最后一个点
        lineseg1 = (np.array(approxCurve[-1]) - np.array(approxCurve[-2])).squeeze(0)
        lineseg2 = (np.array(approxCurve[0]) - np.array(approxCurve[-1])).squeeze(0)
        L_1 = np.sqrt(lineseg1.dot(lineseg1))
        L_2 = np.sqrt(lineseg2.dot(lineseg2))

        pi_dist = lineseg1.dot(lineseg2) / (L_1 * L_2)
        if abs(pi_dist) >= 1.:
            # pi_dist = (pi_dist / abs(pi_dist)) * 0.999  # 防止arccos输入超出范围[-1,1]  判断是否超出1
            print('pi_dist value overflow')
        angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi
        a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])  # 不用计算出角度再判断正负，直接用叉乘值判断正负？
        if angle_dist >= turn_angle or a_sign != angle_sign[-1]:
            seged_edgeset.append(approxCurve[seg_index[-1]:])
            seged_edgeset.append(np.concatenate((approxCurve[-1][np.newaxis,], approxCurve[0][np.newaxis,]), 0))
        else:
            seged_edgeset.append(np.concatenate((approxCurve[seg_index[-1]:], approxCurve[0][np.newaxis,]), 0))
        # 判断首末端点连接
        lineseg1 = (np.array(approxCurve[0]) - np.array(approxCurve[-1])).squeeze(0)
        lineseg2 = (np.array(approxCurve[1]) - np.array(approxCurve[0])).squeeze(0)
        L_1 = np.sqrt(lineseg1.dot(lineseg1))
        L_2 = np.sqrt(lineseg2.dot(lineseg2))

        pi_dist = lineseg1.dot(lineseg2) / (L_1 * L_2)
        if abs(pi_dist) >= 1.:
            # pi_dist = (pi_dist / abs(pi_dist)) * 0.999  # 防止arccos输入超出范围[-1,1]  判断是否超出1
            print('pi_dist value overflow')
        angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi
        a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])  # 不用计算出角度再判断正负，直接用叉乘值判断正负？
        if angle_dist <= turn_angle and a_sign == angle_sign[-1]:
            seged_edgeset[0] = np.concatenate((seged_edgeset[-1], seged_edgeset[0]), 0)
            seged_edgeset.pop()
        # if len(seged_edgeset) == 0:
        # 	print('len = 0', len(seged_edgeset))
        seged_edgeset_b = []
        for i in range(len(seged_edgeset)):
            if seged_edgeset[i].shape[0] > 2:  # 线段阈值数
                seged_edgeset_b.append(seged_edgeset[i])
        seged_edgeset = seged_edgeset_b
        # 如果只剩0个曲线 直接返回approcurve前两个值，跳过拟合
        if len(seged_edgeset) == 0:
            print('--------------')
            return approxCurve[0:2]
        # arc 对应contours
        # filted_contour = self.arc2contours(ori_contour,seged_edgeset)
        # print('filted_contour',filted_contour[0].shape)
        # print('filted_contour1',filted_contour[1].shape)
        # print('seged_edgeset', len(seged_edgeset))
        # 如果只剩一个曲线 直接返回arc
        if len(seged_edgeset) == 1:
            return seged_edgeset[0]
        maxarc = 0
        # print('len',len(seged_edgeset))

        #  将所有弧段旋转顺序调整到同一方向
        for i in range(len(seged_edgeset)):
            if seged_edgeset[i].shape[0] > maxarc:
                max_index = i
                maxarc = seged_edgeset[i].shape[0]
            lineseg1 = (np.array(seged_edgeset[i][2]) - np.array(seged_edgeset[i][1])).squeeze(0)
            lineseg2 = (np.array(seged_edgeset[i][1]) - np.array(seged_edgeset[i][0])).squeeze(0)
            a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])

            if a_sign < 0:
                seged_edgeset[i] = seged_edgeset[i][::-1]

        lineseg1 = (np.array(seged_edgeset[max_index][-1]) - np.array(seged_edgeset[max_index][-2])).squeeze(0)
        lineseg2 = (np.array(seged_edgeset[max_index][1]) - np.array(seged_edgeset[max_index][0])).squeeze(0)
        L_1 = np.sqrt(lineseg1.dot(lineseg1))
        L_2 = np.sqrt(lineseg2.dot(lineseg2))

        pi_dist = lineseg1.dot(lineseg2) / (L_1 * L_2)
        a_sign = np.sign(lineseg1[0] * lineseg2[1] - lineseg2[0] * lineseg1[1])
        if a_sign < 0:
            angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi + 180
        else:
            angle_dist = np.arccos(pi_dist) * 360 / 2 / np.pi
        if angle_dist > 270:
            ellip_group.append(seged_edgeset[max_index])
        # ellip_group.append(filted_contour[max_index])
        else:
            ellip_group = self.arc_group(seged_edgeset, max_index)

        for i in range(len(ellip_group)):
            if i == 0:
                arc_Curve = ellip_group[i]
            else:
                arc_Curve = np.concatenate((arc_Curve, ellip_group[i]), 0)
        # ellipse = cv.fitEllipseDirect(arc_Curve)
        return arc_Curve

    def Issearchregion(self, seedarc, subarc):
        # seedarc subarc 形状为三维.l_1等需要转换为一维数据
        l_1 = seedarc[0, 0] - seedarc[1, 0]
        l_2 = seedarc[-1, 0] - seedarc[-2, 0]
        l_m = seedarc[-1, 0] - seedarc[0, 0]
        # sub_midpoint = (subarc[0] + subarc[-1])/2
        sub_midpoint = subarc[subarc.shape[0] // 2 - 1, 0]
        p_t = sub_midpoint - seedarc[0, 0]
        if (l_1[0] * p_t[1] - l_1[1] * p_t[0]) > 0:
            # flat = 0
            return 0
        if (p_t[0] * l_m[1] - p_t[1] * l_m[0]) > 0:
            return 0
        p_t = sub_midpoint - seedarc[-1, 0]
        if (p_t[0] * l_2[1] - p_t[1] * l_2[0]) > 0:
            return 0
        return 1

    def arc_group(self, seged_edgeset, max_index):
        ellip_g = []
        ellip_g.append(seged_edgeset[max_index])
        # ellip_g.append(filted_contour[max_index])
        for i in range(len(seged_edgeset)):
            if i != max_index:
                if self.Issearchregion(seged_edgeset[max_index], seged_edgeset[i]) == 1:
                    if self.Issearchregion(seged_edgeset[i], seged_edgeset[max_index]) == 1:
                        ellip_g.append(seged_edgeset[i])
                    # ellip_g.append(filted_contour[i])
        return ellip_g

    def arc2contours(self, contours, arc):
        arc_start = -1
        arc_end = -1
        contour_group = []
        #  先找第一个arc
        for j in range(contours.shape[0]):
            if (contours[j] == arc[0][0]).all():
                arc_start = j
                break
        for j in range(contours.shape[0]):
            if (contours[j] == arc[0][-1]).all():
                arc_end = j
                break
        if arc_start > arc_end:
            contour_one = np.concatenate((contours[arc_start:], contours[0:arc_end + 1]), 0)
            contour_group.append(contour_one)
        else:
            contour_one = contours[arc_start:arc_end]
            contour_group.append(contour_one)
        k = 0
        for i in range(1, len(arc)):
            for j in range(k, contours.shape[0]):
                if (contours[j] == arc[i][0]).all():
                    arc_start = j
                if (contours[j] == arc[i][-1]).all():
                    arc_end = j
                    contour_group.append(contours[arc_start:arc_end])
                    k = j
                    break
        return contour_group

    def cal_draw_ellipse(self, I1, I1_g):
        contours, _ = cv.findContours(cv.medianBlur(I1_g, 15), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        maxindex1 = 0
        maxindex2 = 0
        max1 = 0
        max2 = 0
        # flag1 = 0
        # flag2 = 0
        for j in range(len(contours)):
            if contours[j].shape[0] > max1:
                maxindex1 = j
                max1 = contours[j].shape[0]
            if j == len(contours) - 1:
                approxCurve = cv.approxPolyDP(contours[maxindex1], 2, closed=True)
                if approxCurve.shape[0] > 5:
                    approxCurve = self.edgefliter(approxCurve, turn_angle=90)
                    I1 = cv.drawContours(I1, [contours[maxindex1]], 0, (0, 255, 255), 2)
                    # cv.polylines(I1, [approxCurve], isClosed=False, color=(0, 0, 255), thickness=1, lineType=8, shift=0)
                    # I1 = cv.drawContours(I1, [approxCurve], 0, (0, 0, 255), 1)  # 得到的耻骨联合区域曲线
                    # ellipse = cv.fitEllipse(approxCurve)
                    # cv2.fillPoly(img, contours[1], (255, 0, 0))  # 只染色边界
                    ellipse = cv.fitEllipseDirect(approxCurve)
                    ellipse2 = cv.fitEllipseDirect(contours[maxindex1])
                    cv.ellipse(I1, ellipse, (255, 0, 255), 2)
                    cv.ellipse(I1, ellipse2, (255, 0, 0), 2)
                    cv.polylines(I1, [approxCurve], isClosed=False, color=(0, 0, 255), thickness=3, lineType=8, shift=0)

        return I1

    def video_output(self, video_path, model_name, csv_path):
        unet_path = os.path.join(self.model_path, model_name)

        self.build_model()
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            print('No pretrained_model')

            return

        self.unet.train(False)
        self.unet.eval()
        # writer = SummaryWriter('runs3/aop')

        length = 0
        fps = 20
        size = (512, 384)
        videos_src_path = video_path  # 'D:/py_seg/video7'
        videos = os.listdir(videos_src_path)
        videos = filter(lambda x: x.endswith('mp4'), videos)

        all_video_list = []
        video_name = []
        ####
        for each_video in videos:
            list_aop = []
            print(each_video)
            videowriter = cv.VideoWriter(each_video, cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
            # get the full path of each video, which will open the video tp extract frames
            each_video_full_path = os.path.join(videos_src_path, each_video)

            cap = cv.VideoCapture(each_video_full_path)
            success = True
            frame_num = 0
            while (success):
                success, frame = cap.read()
                frame_num += 1
                if success == False:
                    break
                frame = frame[54:, 528:1823]
                frame[0:434, 1175:] = 0
                frame[959:1003, 199:1048] = 0
                frame[:, 0:72] = 0
                frame = cv.resize(frame, (512, 384))
                # print('Read a new frame: ', success)
                image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 把GRAY图转换为BGR三通道图--BGR转灰度图
                transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor(),
                     # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
                     torchvision.transforms.Normalize((0.5,), (0.5,))])

                image = transform(image)
                image = image.unsqueeze(0)
                # image = Norm_(torch.tensor(image)).unsqueeze(0).unsqueeze(0)

                # image = Norm_(image)

                image = image.to(self.device, torch.float)

                # SR, _, _, _, _, _ = self.unet(image)
                SR_lm, SR_seg, _, _, _, SR_cls = self.unet(image)
                SR = torch.softmax(SR_seg, 1)
                SR_cls = torch.softmax(SR_cls, 1)
                SR = SR > 0.5
                SR = SR.mul(255)
                SR = SR.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
                contours, _ = cv.findContours(cv.medianBlur(SR[:, :, 1], 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                contours2, _ = cv.findContours(cv.medianBlur(SR[:, :, 2], 5), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                img_result = frame
                maxindex1 = 0
                maxindex2 = 0
                max1 = 0
                max2 = 0
                flag1 = 0
                flag2 = 0

                ##########lm
                # lm_h = (SR_lm[:, 0, :, :] + SR_lm[:, 1, :, :]).mul(255)
                # lm_h = lm_h.detach().cpu().numpy().squeeze(0).astype(np.uint8)
                # lm_j = np.zeros((lm_h.shape[0], lm_h.shape[1], 3)).astype(np.uint8)
                # lm_h = self.gray2color(lm_h, jet_map)
                # lm_h = lm_h[:, :, [2, 1, 0]]
                # num = num.cpu().numpy().squeeze(0)
                # image = (image.mul(127)) + 128
                # image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.uint8)
                # image1 = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                # image_h = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
                ########landmark
                SR_lm1 = SR_lm[0, 0, :, :].cpu().detach()
                SR_lm2 = SR_lm[0, 1, :, :].cpu().detach()
                out_cor1 = np.unravel_index(np.argmax(SR_lm1), SR_lm1.shape)
                out_cor2 = np.unravel_index(np.argmax(SR_lm2), SR_lm2.shape)

                for j in range(len(contours)):
                    if contours[j].shape[0] > max1:
                        maxindex1 = j
                        max1 = contours[j].shape[0]
                    if j == len(contours) - 1:
                        approxCurve = cv.approxPolyDP(contours[maxindex1], 2, closed=True)
                        if approxCurve.shape[0] > 5:
                            # img_result = cv.drawContours(img_result, [approxCurve], 0, (0, 0, 255), 1)  # 得到的耻骨联合区域曲线
                            ellipse = cv.fitEllipse(approxCurve)
                            # cv.ellipse(img_result, ellipse, (0, 255, 0), 2)
                            flag1 = 1

                for k in range(len(contours2)):
                    if contours2[k].shape[0] > max2:
                        maxindex2 = k
                        max2 = contours2[k].shape[0]
                    if k == len(contours2) - 1:
                        approxCurve2 = cv.approxPolyDP(contours2[maxindex2], 2, closed=True)
                        if approxCurve2.shape[0] > 5:
                            # img_result = cv.drawContours(img_result, [approxCurve2], 0, (255, 0, 0), 1)
                            ellipse2 = cv.fitEllipse(approxCurve2)
                            cv.ellipse(img_result, ellipse2, (0, 255, 0), 2)
                            flag2 = 1

                if flag1 == 1 and flag2 == 1:
                    # img_result, Aod = drawline_AOD(img_result,ellipse2, ellipse,out_cor2,out_cor1)
                    img_result, Aod = drawline_AOD2(img_result, ellipse2, ellipse)
                    list_aop.append(Aod)

                    cv.putText(img_result, "AOP: " + str(round(Aod, 2)) + ""
                                                                          "", (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 255, 255), 1, cv.LINE_AA)
                    cv.line(img_result, (out_cor1[1] - 4, out_cor1[0]), (out_cor1[1] + 4, out_cor1[0]), (0, 255, 0), 2)
                    cv.line(img_result, (out_cor1[1], out_cor1[0] - 4), (out_cor1[1], out_cor1[0] + 4), (0, 255, 0), 2)

                    cv.line(img_result, (out_cor2[1] - 4, out_cor2[0]), (out_cor2[1] + 4, out_cor2[0]), (0, 255, 0), 2)
                    cv.line(img_result, (out_cor2[1], out_cor2[0] - 4), (out_cor2[1], out_cor2[0] + 4), (0, 255, 0), 2)

                    cv.line(img_result, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)
                    cv.line(img_result, (out_cor1[1], out_cor1[0]), (out_cor2[1], out_cor2[0]), (0, 255, 0), 1)

                # writer.add_scalar('AOP/'+each_video,  Aod, frame_num)
                else:
                    # writer.add_scalar('AOP/' + each_video, 0, frame_num)
                    pass
                if SR_cls[0, 1] > 0.5:
                    list_aop.append(Aod)
                else:
                    list_aop.append(0)

                videowriter.write(img_result)

            videowriter.release()
            print('[AOP]  median: %.4f  mean: %.4f  std: %.4f' % (
                np.median(list_aop), np.mean(list_aop), np.std(list_aop)))
            all_video_list.append(list_aop)
            video_name.append(each_video)

        aop_csv = pd.DataFrame(index=video_name, data=all_video_list)
        aop_csv.to_csv(csv_path, encoding='gbk')
        cap.release()

    def calculate_true_aop_3point(self, num):
        landmark_Path = r'F:\ZDJ\MTAFN_510\aopnewGT.csv'
        landmark_coor = np.loadtxt(landmark_Path, delimiter=',')

        landmark_coor_l = landmark_coor[num - 1][0:2]
        landmark_coor_r = landmark_coor[num - 1][2:4]
        landmark_coor_3 = landmark_coor[num - 1][4:]

        out_midline = np.array(landmark_coor_l) - np.array(landmark_coor_r)
        mask_midline = np.array(landmark_coor_3) - np.array(landmark_coor_r)

        L_out = max(np.sqrt(out_midline.dot(out_midline)), 0.001)
        L_mask = max(np.sqrt(mask_midline.dot(mask_midline)), 0.001)
        if abs(out_midline[0]) < 1:
            out_midline[0] = 1
        if abs(mask_midline[0]) < 1:
            mask_midline[0] = 1
        #
        pi_dist = mask_midline.dot(out_midline) / (L_out * L_mask)
        if abs(pi_dist) >= 1.:
            pi_dist = (pi_dist / abs(pi_dist)) * 0.999  # 防止arccos输入超出范围[-1,1]
        aop = np.arccos(pi_dist) * 360 / 2 / np.pi
        if out_midline[1] / out_midline[0] == mask_midline[1] / mask_midline[0]:
            aop = 1000

        return aop

    def calculate_true_aop_2point(self, contours_GT_head, out_cor2, out_cor1, image1):
        out_cor3 = []
        point_num_1 = 0
        point_num_2 = 0
        flag1 = 0
        flag2 = 0
        flag3 = 0
        coor = []
        # plt.figure()
        image2 = np.copy(image1)
        size = (635, 522)
        image2 = cv.resize(image2, size, interpolation=cv.INTER_AREA)
        for angle in range(0, 90, 1):
            # cv.line(image2, (out_cor1[0], out_cor1[1]), (out_cor2[0], out_cor2[1]), (255, 0, 0), 1)
            # cv.line(image2, (out_cor2[0], out_cor2[1]),
            #         (out_cor2[0] + np.int(np.cos(angle * np.pi/180) * 1000), (out_cor2[1] + np.int(np.sin(angle*np.pi/180) * 1000))),
            #         (0, 255, 0), 1)
            # image2 = cv.drawContours(image2, contours_GT_head, 0, (0, 0, 255), 1)
            # print(np.cos(60))
            # plt.imshow(image2)
            # plt.show()
            # print('...')
            for i in range(len(contours_GT_head[0])):
                x = contours_GT_head[0][i][0][0]
                y = contours_GT_head[0][i][0][1]
                pass
                if y < math.tan(angle / 2 * np.pi / 180) * (x - out_cor2[0]) + out_cor2[1]:
                    out_cor3.append(x)
                    out_cor3.append(y)
                    flag1 = 1
                    break
                if y < math.tan(angle * np.pi / 180) * (x - out_cor2[0]) + out_cor2[1]:
                    out_cor3.append(x)
                    out_cor3.append(y)
                    flag1 = 1
                    break
            if 1 == flag1:
                break

        # cv.line(image2, (out_cor1[0], out_cor1[1]), (out_cor2[0], out_cor2[1]), (0, 255, 0), 1)
        # cv.line(image2, (out_cor2[0], out_cor2[1]),
        #         (out_cor2[0] + np.int(np.cos(angle * np.pi/180) * 100), (out_cor2[1] + np.int(np.sin(angle*np.pi/180) * 100))),
        #         (0, 255, 0), 1)
        # image2 = cv.drawContours(image2, contours_GT_head, 0, (0, 0, 255), 1)
        # print(np.cos(60))
        # plt.figure()
        # plt.imshow(image2)
        # plt.show()
        # pass

        if flag1 != 1:
            for j in range(len(contours_GT_head[0])):
                if out_cor2[0] == contours_GT_head[0][i][0][0]:
                    out_cor3.append(out_cor2[0])
                    out_cor3.append(out_cor2[1] + 10)
                    flag2 = 1
                    break

        if flag1 != 1 and flag2 != 1:
            for angle_1 in range(91, 180):
                for i in range(len(contours_GT_head[0])):
                    y = contours_GT_head[0][i][0][0]
                    x = contours_GT_head[0][i][0][1]
                    pass
                    if y > math.tan(angle_1 / 2 * np.pi / 180) * (x - out_cor2[0]) + out_cor2[1]:
                        out_cor3.append(x)
                        out_cor3.append(y)
                        flag3 = 1
                        break
                    if y > math.tan(angle_1 * np.pi / 180) * (x - out_cor2[0]) + out_cor2[1]:
                        out_cor3.append(x)
                        out_cor3.append(y)
                        flag3 = 1
                        break
                if 1 == flag3:
                    break

        if flag1 != 1 and flag2 != 1 and flag3 != 1:
            print('there is a bug')

        out_midline = np.array(out_cor1) - np.array(out_cor2)
        mask_midline = np.array(out_cor3) - np.array(out_cor2)

        L_out = max(np.sqrt(out_midline.dot(out_midline)), 0.001)
        L_mask = max(np.sqrt(mask_midline.dot(mask_midline)), 0.001)
        if abs(out_midline[0]) < 1:
            out_midline[0] = 1
        if abs(mask_midline[0]) < 1:
            mask_midline[0] = 1
        #
        pi_dist = mask_midline.dot(out_midline) / (L_out * L_mask)
        if abs(pi_dist) >= 1.:
            pi_dist = (pi_dist / abs(pi_dist)) * 0.999  # 防止arccos输入超出范围[-1,1]
        aop = np.arccos(pi_dist) * 360 / 2 / np.pi
        if out_midline[1] / out_midline[0] == mask_midline[1] / mask_midline[0]:
            aop = 1000
        # GT_cor3 = out_cor3[:: -1]
        return aop, out_cor3
