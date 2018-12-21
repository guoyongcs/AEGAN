from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from aegan_model import _netG1, _netD1, _netG2, _netD2, _netRS, AEGAN_ResnetGenerator, AEGAN_ResnetDecoder, NLayerDiscriminator
from refiner_model import _RefinerG, _RefinerD, UnetGenerator

# from calculate_fid_pytorch.fid import fid_score
import vutils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter_stage1', type=int, default=50, help='number of epochs to train for RS and G2')
parser.add_argument('--niter_stage2', type=int, default=200, help='number of epochs to train for G1 and D1')
# parser.add_argument('--niter3', type=int, default=1000, help='number of epochs to train for G1+G2 and D2')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrRS', type=float, default=0.00001, help='learning rate, default=0.00001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG1', default='', help="path to netG1 (to continue training)")
parser.add_argument('--netD1', default='', help="path to netD1 (to continue training)")
parser.add_argument('--netG2', default='', help="path to netG2 (to continue training)")
parser.add_argument('--netRS', default='', help="path to netRS (to continue training)")
parser.add_argument('--RefinerG', default='', help="path to Refiner (to continue training)")
parser.add_argument('--RefinerD', default='', help="path to RefinerD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nc', type=int, default=3, help='the number of image channel')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--sub_dir1', default='imgStep1', help='the sub directory 1 of saving images')
parser.add_argument('--sub_dir2', default='imgStep2', help='the sub directory 2 of saving images')
parser.add_argument('--sub_dir3', default='imgStep3', help='the sub directory 3 of saving images')
parser.add_argument('--lsun_class', default='bedroom_train', help='the class of the lsun dataset')

parser.add_argument('--fid_dir', default='fid_dir', help='the sub directory for saving images \
    to computer fid score, and log file')
parser.add_argument('--fid_real_path', type=str, default="/home/chenqi/dataset/text2video/MSVD_DAMSM/train_image",\
    help='the real path to save the real images to fid')
parser.add_argument('--pickle_dir', type=str, default='pickle', help='input path')
parser.add_argument('--train_stage', type=int, default=1, help='training stage index')

parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument('--gan_type', type=str, default='GAN', help='options: GAN | Patch | PatchLSGAN')

opt = parser.parse_args()
print(opt)

# set gpu ids
str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])

try:
    os.makedirs(opt.outf)
    os.makedirs(os.path.join(opt.outf, opt.sub_dir1))
    os.makedirs(os.path.join(opt.outf, opt.sub_dir2))
    os.makedirs(os.path.join(opt.outf, opt.sub_dir3))
except OSError:
    pass

# the folder for saving pickle
pickle_save_path = os.path.join(opt.outf, opt.pickle_dir)
if not os.path.exists(pickle_save_path):
    os.makedirs(pickle_save_path)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    # torch.cuda.set_device(opt.gpu)
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['flowers', 'birds', 'volcano', 'ant', 'monastery', 'fire_engine', 'harvester', 'broccoli', 'studio_couch', 'lfw', 'imagenet']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)


# design model
netG1 = _netG1(nz, ngf, opt.batchSize)
# netG1 = AEGAN_ResnetDecoder(nz, 64, n_blocks=1, n_downsampling=2)
# netG1.apply(weights_init)
if opt.netG1 != '':
    netG1.load_state_dict(torch.load(opt.netG1))
netG1 = vutils.init_net(netG1, opt.gpu_ids) 
print(netG1)

# 
if opt.gan_type=='GAN':
    netD1 = _netD1(ndf)
else:
    netD1 = NLayerDiscriminator(64)
# netD1.apply(weights_init)
if opt.netD1 != '':
    netD1.load_state_dict(torch.load(opt.netD1))
netD1 = vutils.init_net(netD1, opt.gpu_ids) 
print(netD1)

netG2 = _netG2(nz, ngf, nc, opt.batchSize)
# netG2 = AEGAN_ResnetGenerator(64, nc, n_blocks=0, from_to_num=4, final_tanh=True)
# netG2.apply(weights_init)
if opt.netG2 != '':
    netG2.load_state_dict(torch.load(opt.netG2))
netG2 = vutils.init_net(netG2, opt.gpu_ids) 
print(netG2)

netRS = _netRS(nc, ndf)
# netRS = AEGAN_ResnetGenerator(nc, 64, n_blocks=0, from_to_num=4, final_tanh=False)
# netRS.apply(weights_init)
if opt.netRS != '':
    netRS.load_state_dict(torch.load(opt.netRS))
netRS = vutils.init_net(netRS, opt.gpu_ids) 
print(netRS)

RefinerG = _RefinerG(nc, ngf)
# RefinerG = UnetGenerator(nc, nc, 5)
# RefinerG.apply(weights_init)
if opt.RefinerG != '':
    RefinerG.load_state_dict(torch.load(opt.RefinerG))
RefinerG = vutils.init_net(RefinerG, opt.gpu_ids) 
print(RefinerG)

if opt.gan_type=='GAN':
    RefinerD = _RefinerD(nc, ndf)
else:
    RefinerD = NLayerDiscriminator(3)
# RefinerD.apply(weights_init)
if opt.RefinerD != '':
    RefinerD.load_state_dict(torch.load(opt.RefinerD))
RefinerD = vutils.init_net(RefinerD, opt.gpu_ids) 
print(RefinerD)

# design the criterion
criterion = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion_l2 = nn.MSELoss()
criterion_NLL = nn.NLLLoss2d()

# variables
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).uniform_(-1, 1)
label = torch.FloatTensor(opt.batchSize)
# label = torch.LongTensor(opt.batchSize)
real_label = 1
fake_label = 0

device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))

# use cuda
if opt.cuda:
    netD1.cuda()
    netG1.cuda()
    netG2.cuda()
    netRS.cuda()
    RefinerG.cuda()
    RefinerD.cuda()
    criterion.cuda()
    criterion_l1.cuda()
    criterion_l2.cuda()
    criterion_NLL.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

def GANLoss(output, label, bce, mse, gan_type):
    if gan_type=='GAN' or gan_type=='Patch':
        err = bce(output, label.expand_as(output))
    elif gan_type=='PatchLSGAN':
        err = mse(output, label.expand_as(output))
    else:
        assert False, 'unsupported gan type: %s' % gan_type
    return err

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD1 = optim.Adam(netD1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG1 = optim.Adam(netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG2 = optim.Adam(netG2.parameters(), lr=opt.lrRS, betas=(opt.beta1, 0.999))
optimizerRS = optim.Adam(netRS.parameters(), lr=opt.lrRS, betas=(opt.beta1, 0.999))
optimizerRefinerG = optim.Adam(RefinerG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerRefinerD = optim.Adam(RefinerD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# # extract nimages for computing fid
# print('extract nimages for computing fid')
# vutils.extract_nimages(opt.dataroot, os.path.join(opt.fid_real_path, 'images'), 500)

print('start training')
for stage in range(opt.train_stage, 4):
    if stage == 1:
        ###############################################################
        # Stage 1: Update RS net and G2 net (Reverse Skip Connection) #
        ###############################################################
        for epoch in range(opt.niter_stage1):
            for i, data in enumerate(dataloader, 0):
                
                #####################
                # (0) Preprocessing #
                #####################
                # gain the real data
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if batch_size < opt.batchSize:
                    break
                input.data.resize_(real_cpu.size()).copy_(real_cpu)

                # design noise
                noise.data.resize_(batch_size, nz, 1, 1)
                noise.data.uniform_(-1,1)

                #########################################################
                # (1) Update RS + G2 network (Reverse Skip Connection)  #
                #########################################################
                netRS.zero_grad()
                netG2.zero_grad()

                input_hat = netG2(netRS(input))
                errRS = criterion_l1(input_hat, input)
                errRS.backward()
                optimizerRS.step()
                optimizerG2.step()

                print('[%d/%d][%d/%d] Loss_RS: %.4f' % (epoch, opt.niter_stage1, i, len(dataloader), errRS.data[0]))
                # if i % 100 == 0 and opt.dataset!='lsun':
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/%s/real_samples.png' %(opt.outf, opt.sub_dir1), normalize=True)
                    fake = netG2(netRS(input))

                    vutils.save_image(fake.data,
                            '%s/%s/fake_samples_epoch_%s.png' % (opt.outf, opt.sub_dir1, epoch), normalize=True)

            # do checkpointing
            # if (epoch+1) % 50 == 0 and opt.dataset!='lsun':
            if epoch % 50 == 0:
                vutils.save_networks(netG2, '%s/step1_netG2_epoch_%d.pth' % (pickle_save_path, epoch), opt.gpu_ids)
                vutils.save_networks(netRS, '%s/step1_netRS_epoch_%d.pth' % (pickle_save_path, epoch), opt.gpu_ids)
            elif epoch == (opt.niter_stage1 - 1):
                vutils.save_networks(netG2, '%s/step1_netG2_last.pth' % (pickle_save_path), opt.gpu_ids)
                vutils.save_networks(netRS, '%s/step1_netRS_last.pth' % (pickle_save_path), opt.gpu_ids)
    elif stage==2:
        #####################################################
        # Stage 2: Update G1, D1 net and RefinerG, RefinerD #
        #####################################################
        best_fid = 10000.0
        best_epoch = 0
        counter_img = 0
        counter_refine = 0
        for epoch in range(opt.niter_stage2):
            for i, data in enumerate(dataloader, 0):
                #####################
                # (0) Preprocessing #
                #####################
                # gain the real data
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if batch_size < opt.batchSize:
                    break
                input.data.resize_(real_cpu.size()).copy_(real_cpu)

                # design noise
                noise.data.resize_(batch_size, nz, 1, 1)
                noise.data.uniform_(-1,1)
                ################################################################
                # (1) Update D1 network: maximize log(D(x)) + log(1 - D(G(z))) #
                ################################################################
                # train with real
                netD1.zero_grad()
                output_RS = netRS(input.detach())
                # label.data.resize_(batch_size).fill_(real_label)
                label.data.resize_(1).fill_(real_label)
                output = netD1(output_RS)
                # errD1_real = criterion(output, label)
                # errD1_real = criterion(output, label.expand_as(output))
                errD1_real = GANLoss(output, label, criterion, criterion_l2, opt.gan_type)
                errD1_real.backward()
                D1_x = output.data.mean()
                # train with fake
                fake = netG1(noise)
                label.data.fill_(fake_label)
                output = netD1(fake.detach())
                # errD1_fake = criterion(output, label)
                # errD1_fake = criterion(output, label.expand_as(output))
                errD1_fake = GANLoss(output, label, criterion, criterion_l2, opt.gan_type)
                errD1_fake.backward()
                D1_G1_z1 = output.data.mean()
                errD1 = errD1_real + errD1_fake
                optimizerD1.step()
                ################################################
                # (2) Update G1 network: maximize log(D(G(z))) #
                ################################################
                netG1.zero_grad()
                label.data.fill_(real_label)  # fake labels are real for generator cost
                output = netD1(fake)
                # errG1 = criterion(output, label) + 10 * criterion_l1(input_hat, input)
                # errG1 = criterion(output, label.expand_as(output))
                errG1 = GANLoss(output, label, criterion, criterion_l2, opt.gan_type)
                # errG1 = criterion(output, label.expand_as(output)) + 10 * criterion_l1(input_hat, input)
                errG1.backward()
                D1_G1_z1 = output.data.mean()
                optimizerG1.step()
                print('Train G1: [%d/%d][%d/%d] Loss_D1: %.4f Loss_G1: %.4f D1(x): %.4f D1(G1(z)): %.4f'
                    % (epoch, opt.niter_stage2, i, len(dataloader),
                        errD1.data[0], errG1.data[0], D1_x, D1_G1_z1))
                # if i % 100 == 0 and opt.dataset!='lsun':
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/%s/real_samples.png' % (opt.outf, opt.sub_dir2), normalize=True)
                    fake = netG2(netG1(noise))
                    vutils.save_image(fake,
                            '%s/%s/fake_samples_epoch_%s.png' % (opt.outf, opt.sub_dir2, epoch), normalize=True)
            ######################################################################
            # (3) Update RefinerD network: maximize log(D(x)) + log(1 - D(G(z))) #
            ######################################################################
            # if epoch%3==0:
            #     counter_refine += 1
            #     for i, data in enumerate(dataloader, 0):

                #####################
                # (0) Preprocessing #
                #####################
                # # gain the real data
                # real_cpu, _ = data
                # batch_size = real_cpu.size(0)
                # if batch_size < opt.batchSize:
                #     break
                # input.data.resize_(real_cpu.size()).copy_(real_cpu)
                # # design noise
                # noise.data.resize_(batch_size, nz, 1, 1)
                # noise.data.uniform_(-1,1)

                # train with real
                RefinerD.zero_grad()
                fake_input = netG2(netG1(noise))
                # label.data.resize_(batch_size).fill_(real_label)
                label.data.resize_(1).fill_(real_label)
                output = RefinerD(input)
                # errRefinerD_real = criterion(output, label)
                # errRefinerD_real = criterion(output, label.expand_as(output))
                errRefinerD_real = GANLoss(output, label, criterion, criterion_l2, opt.gan_type)
                # errRefinerD_real.backward(retain_variables=True)
                errRefinerD_real.backward(retain_graph=True)
                RefinerD_x = output.data.mean()
                # train with fake
                fake_refined = RefinerG(fake_input)
                label.data.fill_(fake_label)
                output = RefinerD(fake_refined.detach())
                # errRefinerD_fake = criterion(output, label)
                # errRefinerD_fake = criterion(output, label.expand_as(output))
                errRefinerD_fake = GANLoss(output, label, criterion, criterion_l2, opt.gan_type)
                # errRefinerD_fake.backward(retain_variables=True)
                errRefinerD_fake.backward(retain_graph=True)
                RefinerD_G_z = output.data.mean()
                errRefinerD = errRefinerD_real + errRefinerD_fake
                optimizerRefinerD.step()
                ######################################################
                # (4) Update RefinerG network: maximize log(D(G(z))) #
                ######################################################
                RefinerG.zero_grad()
                # netG1.zero_grad()
                # netG2.zero_grad()
                # netRS.zero_grad()
                # input_hat = netG2(netRS(input))
                # errRS = criterion_l1(input_hat, input)
                # errRS.backward()

                label.data.fill_(real_label)  # fake labels are real for generator cost
                output = RefinerD(fake_refined)
                # errRefinerG = criterion(output, label) + opt.lamb * criterion_l1(fake_refined, fake_input.detach())
                # errRefinerG = criterion(output, label.expand_as(output)) + opt.lamb * criterion_l1(fake_refined, fake_input.detach())
                errRefinerG = GANLoss(output, label, criterion, criterion_l2, opt.gan_type) + opt.lamb * criterion_l1(fake_refined, fake_input.detach())
                # errRefinerG = criterion(output, label)
                errRefinerG.backward()
                # optimizerG1.step()
                # optimizerG2.step()
                # optimizerRS.step()
                # RefinerG_z = output.data.mean()
                optimizerRefinerG.step()
                print('Train Refiner: [%d/%d][%d/%d] Loss_RefinerD: %.4f Loss_RefinerG: %.4f D(x): %.4f D(G(z)): %.4f'
                    % (epoch, opt.niter_stage2, i, len(dataloader),
                        errRefinerD.data[0], errRefinerG.data[0], RefinerD_x, RefinerD_G_z))
                fake = RefinerG(netG2(netG1(noise)))
                # if i % 100 == 0 and opt.dataset!='lsun':
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/imgStep3/real_samples.png' % opt.outf, normalize=True)
                    vutils.save_image(fake,
                            '%s/imgStep3/fake_samples_epoch_%03d.png' % (opt.outf, epoch), normalize=True)
                # =================== #
                # Calcualte FID score #
                # =================== #
                # calculate the fid score
                # fid_image_path = os.path.join(args.output_dir, args.fid_fake_foldername, "images")
                # fid_image_path = os.path.join(opt.outf, opt.fid_dir, "images")
                # if not os.path.exists(fid_image_path):
                #     os.makedirs(fid_image_path)
                # counter_img = vutils.save_image_forFID(fake, 
                #     normalize=True, pad_value=1, batchSize=opt.batchSize,
                #     counter=counter_img, output=fid_image_path)

            # # path_fid_images = [args.fid_real_path, fid_image_path]
            # path_fid_images = [opt.fid_real_path, os.path.join(opt.outf, opt.fid_dir)]
            # print('calculate the fid score ...')
            # # fid_value = fid_score.calculate_fid_score(path=path_fid_images,
            # #     batch_size=args.batch_size, gpu=str(args.gpu+args.gpu_num-1))
            # # try:
            # fid_value = fid_score.calculate_fid_score(path=path_fid_images,
            #     batch_size=opt.batchSize, gpu=str(opt.gpu_ids[0]))
            # # except:
            # #     fid_value = best_fid
            # if fid_value < best_fid:
            #     best_fid = fid_value
            #     best_epoch = epoch
            #     vutils.save_networks(netG1, '{}/step2_netG1_best.pth'.format(pickle_save_path), opt.gpu_ids)
            #     vutils.save_networks(netD1, '{}/step2_netD1_best.pth'.format(pickle_save_path), opt.gpu_ids)
            #     vutils.save_networks(RefinerG, '{}/step3_RefinerG_best.pth'.format(pickle_save_path), opt.gpu_ids)
            #     vutils.save_networks(RefinerD, '{}/step3_RefinerD_best.pth'.format(pickle_save_path), opt.gpu_ids)
            # print("\033[1;31m current_epoch[{}] current_fid[{}] \033[0m \033[1;34m best_epoch[{}] best_fid[{}] \033[0m".format(
            #     epoch, fid_value, best_epoch, best_fid))
            # # save fid
            # with open(os.path.join(opt.outf, 'log_fid.txt'), 'a') as f:
            #     f.write("current_epoch[{}] current_fid[{}] best_epoch[{}] best_fid[{}] \n".format(
            #         epoch, fid_value, best_epoch, best_fid))
            # do checkpointing
            # if counter_refine % 2 == 0:
            if epoch % 10 == 0:
                vutils.save_networks(netG1, '%s/step2_netG1_epoch_%d.pth' % (pickle_save_path, epoch), opt.gpu_ids)
                vutils.save_networks(netD1, '%s/step2_netD1_epoch_%d.pth' % (pickle_save_path, epoch), opt.gpu_ids)
                vutils.save_networks(RefinerG, '%s/step3_RefinerG_epoch_%d.pth' % (pickle_save_path, epoch), opt.gpu_ids)
                vutils.save_networks(RefinerD, '%s/step3_RefinerD_epoch_%d.pth' % (pickle_save_path, epoch), opt.gpu_ids)

            vutils.save_networks(netG1, '%s/step2_netG1_last.pth' % (pickle_save_path), opt.gpu_ids)
            vutils.save_networks(netD1, '%s/step2_netD1_last.pth' % (pickle_save_path), opt.gpu_ids)
            vutils.save_networks(RefinerG, '%s/step3_RefinerG_last.pth' % (pickle_save_path), opt.gpu_ids)
            vutils.save_networks(RefinerD, '%s/step3_RefinerD_last.pth' % (pickle_save_path), opt.gpu_ids)
