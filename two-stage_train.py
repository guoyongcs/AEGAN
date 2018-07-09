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
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from RS_model_separated import _netG1, _netD1, _netG2, _netD2, _netRS
from model_refiner import _RefinerG, _RefinerD

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter_stage1', type=int, default=100, help='number of epochs to train for RS and G2')
parser.add_argument('--niter_stage2', type=int, default=200, help='number of epochs to train for G1 and D1')
# parser.add_argument('--niter3', type=int, default=1000, help='number of epochs to train for G1+G2 and D2')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrRS', type=float, default=0.00001, help='learning rate, default=0.00001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
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
parser.add_argument('--gpu', type=int, default=3, help='number of GPUs to use')
parser.add_argument('--sub_dir1', default='imgStep1', help='the sub directory 1 of saving images')
parser.add_argument('--sub_dir2', default='imgStep2', help='the sub directory 2 of saving images')
parser.add_argument('--sub_dir3', default='imgStep3', help='the sub directory 3 of saving images')
parser.add_argument('--lsun_class', default='bedroom_train', help='the class of the lsun dataset')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
    os.makedirs(os.path.join(opt.outf, opt.sub_dir1))
    os.makedirs(os.path.join(opt.outf, opt.sub_dir2))
    os.makedirs(os.path.join(opt.outf, opt.sub_dir3))
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.set_device(opt.gpu)
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
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['{}'.format(opt.lsun_class)],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# design model
netG1 = _netG1(ngpu, nz, ngf, opt.batchSize)
netG1.apply(weights_init)
if opt.netG1 != '':
    netG1.load_state_dict(torch.load(opt.netG1))
print(netG1)

netD1 = _netD1(ngpu, ndf)
netD1.apply(weights_init)
if opt.netD1 != '':
    netD1.load_state_dict(torch.load(opt.netD1))
print(netD1)

netG2 = _netG2(ngpu, nz, ngf, nc, opt.batchSize)
netG2.apply(weights_init)
if opt.netG2 != '':
    netG2.load_state_dict(torch.load(opt.netG2))
print(netG2)

netRS = _netRS(ngpu, nc, ndf)
netRS.apply(weights_init)
if opt.netRS != '':
    netRS.load_state_dict(torch.load(opt.netRS))
print(netRS)

RefinerG = _RefinerG(ngpu, nc, ngf)
RefinerG.apply(weights_init)
if opt.RefinerG != '':
    RefinerG.load_state_dict(torch.load(opt.RefinerG))
print(RefinerG)

RefinerD = _RefinerD(ngpu, nc, ndf)
RefinerD.apply(weights_init)
if opt.RefinerD != '':
    RefinerD.load_state_dict(torch.load(opt.RefinerD))
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
        if i % 100 == 0 and opt.dataset!='lsun':
            vutils.save_image(real_cpu,
                    '%s/%s/real_samples.png' %(opt.outf, opt.sub_dir1), normalize=True)
            fake = netG2(netRS(input))

            vutils.save_image(fake.data,
                    '%s/%s/fake_samples_epoch_%s.png' % (opt.outf, opt.sub_dir1, epoch), normalize=True)
        elif i % 100 == 0 and opt.dataset=='lsun':
            vutils.save_image(real_cpu,
                    '%s/%s/real_samples.png' %(opt.outf, opt.sub_dir1), normalize=True)
            fake = netG2(netRS(input))

            vutils.save_image(fake.data,
                    '%s/%s/fake_samples_epoch_%s_%s.png' % (opt.outf, opt.sub_dir1, epoch, i), normalize=True)

        # do checkpointing
        if i % 1000 == 0 and opt.dataset=='lsun':
            torch.save(netG2.state_dict(), '%s/step1_netG2_epoch_%d_%s.pth' % (opt.outf, epoch+1, i))
            torch.save(netRS.state_dict(), '%s/step1_netRS_epoch_%d_%s.pth' % (opt.outf, epoch+1, i))
        # elif i==(len(dataloader)-2) and opt.dataset=='lsun':
        #     torch.save(netG2.state_dict(), '%s/step1_netG2_epoch_%d_%s.pth' % (opt.outf, epoch+1, i))
        #     torch.save(netRS.state_dict(), '%s/step1_netRS_epoch_%d_%s.pth' % (opt.outf, epoch+1, i))

    # do checkpointing
    if (epoch+1) % 50 == 0 and opt.dataset!='lsun':
        torch.save(netG2.state_dict(), '%s/step1_netG2_epoch_%d.pth' % (opt.outf, epoch+1))
        torch.save(netRS.state_dict(), '%s/step1_netRS_epoch_%d.pth' % (opt.outf, epoch+1))
    elif opt.dataset=='lsun':
        torch.save(netG2.state_dict(), '%s/step1_netG2_epoch_%d.pth' % (opt.outf, epoch+1))
        torch.save(netRS.state_dict(), '%s/step1_netRS_epoch_%d.pth' % (opt.outf, epoch+1))


#####################################################
# Stage 2: Update G1, D1 net and RefinerG, RefinerD #
#####################################################
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

        # ==================================== noise z to embedding ==================================== # 

        ################################################################
        # (1) Update D1 network: maximize log(D(x)) + log(1 - D(G(z))) #
        ################################################################

        # train with real
        netD1.zero_grad()
        output_RS = netRS(input.detach())
        label.data.resize_(batch_size).fill_(real_label)

        output = netD1(output_RS)
        errD1_real = criterion(output, label)
        errD1_real.backward()
        D1_x = output.data.mean()

        # train with fake
        fake = netG1(noise)
        label.data.fill_(fake_label)
        output = netD1(fake.detach())

        errD1_fake = criterion(output, label)
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
        errG1 = criterion(output, label)
        errG1.backward()
        D1_G1_z1 = output.data.mean()
        optimizerG1.step()


        print('[%d/%d][%d/%d] Loss_D1: %.4f Loss_G1: %.4f D1(x): %.4f D1(G1(z)): %.4f'
              % (epoch, opt.niter_stage2, i, len(dataloader),
                 errD1.data[0], errG1.data[0], D1_x, D1_G1_z1))
        if i % 100 == 0 and opt.dataset!='lsun':
            vutils.save_image(real_cpu,
                    '%s/%s/real_samples.png' % (opt.outf, opt.sub_dir2), normalize=True)
            fake = netG2(netG1(fixed_noise))
            vutils.save_image(fake.data,
                    '%s/%s/fake_samples_epoch_%s.png' % (opt.outf, opt.sub_dir2, epoch), normalize=True)
        
        elif i % 100 == 0 and opt.dataset=='lsun':
            vutils.save_image(real_cpu,
                    '%s/%s/real_samples.png' % (opt.outf, opt.sub_dir2), normalize=True)
            fake = netG2(netG1(fixed_noise))
            vutils.save_image(fake.data,
                    '%s/%s/fake_samples_epoch_%s_%s.png' % (opt.outf, opt.sub_dir2, epoch, i), normalize=True)

        # do checkpointing
        if i%1000 == 0 and opt.dataset=='lsun':
            torch.save(netG1.state_dict(), '%s/step2_netG1_epoch_%d_%s.pth' % (opt.outf, epoch+1, i))
            torch.save(netD1.state_dict(), '%s/step2_netD1_epoch_%d_%s.pth' % (opt.outf, epoch+1, i))


        # ==================================== denoiser network ==================================== # 
        
        ######################################################################
        # (3) Update RefinerD network: maximize log(D(x)) + log(1 - D(G(z))) #
        ######################################################################

        # train with real
        RefinerD.zero_grad()
        fake_input = netG2(netG1(noise))
        label.data.resize_(batch_size).fill_(real_label)

        output = RefinerD(input)
        errRefinerD_real = criterion(output, label)
        errRefinerD_real.backward(retain_variables=True)
        RefinerD_x = output.data.mean()

        # train with fake
        fake_refined = RefinerG(fake_input.detach())
        label.data.fill_(fake_label)
        output = RefinerD(fake_refined)

        errRefinerD_fake = criterion(output, label)
        errRefinerD_fake.backward(retain_variables=True)
        RefinerD_G_z = output.data.mean()

        errRefinerD = errRefinerD_real + errRefinerD_fake
        optimizerRefinerD.step()

        ######################################################
        # (4) Update RefinerG network: maximize log(D(G(z))) #
        ######################################################
        RefinerG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = RefinerD(fake_refined)
        errRefinerG = criterion(output, label) + opt.lamb * criterion_l1(fake_refined, fake_input.detach())
        # errRefinerG = criterion(output, label)
        errRefinerG.backward()
        # RefinerG_z = output.data.mean()
        optimizerRefinerG.step()


        print('[%d/%d][%d/%d] Loss_RefinerD: %.4f Loss_RefinerG: %.4f D(x): %.4f D(G(z)): %.4f'
              % (epoch, opt.niter_stage2, i, len(dataloader),
                 errRefinerD.data[0], errRefinerG.data[0], RefinerD_x, RefinerD_G_z))
        if i % 100 == 0 and opt.dataset!='lsun':
            vutils.save_image(real_cpu,
                    '%s/imgStep3/real_samples.png' % opt.outf, normalize=True)
            # fake = RefinerG(netG2(netG1(fixed_noise)))
            fake = RefinerG(netG2(netG1(noise)))
            vutils.save_image(fake.data,
                    '%s/imgStep3/fake_samples_epoch_%03d.png' % (opt.outf, epoch), normalize=True)

        elif i % 100 == 0 and opt.dataset=='lsun':
            vutils.save_image(real_cpu,
                    '%s/imgStep3/real_samples.png' % opt.outf, normalize=True)
            # fake = RefinerG(netG2(netG1(fixed_noise)))
            fake = RefinerG(netG2(netG1(noise)))
            vutils.save_image(fake.data,
                    '%s/imgStep3/fake_samples_epoch_%03d_%s.png' % (opt.outf, epoch, i), normalize=True)

        # do checkpointing
        if i % 100 == 0 and opt.dataset=='lsun':
            torch.save(RefinerG.state_dict(), '%s/step3_RefinerG_epoch_%d_%s.pth' % (opt.outf, epoch, i))
            torch.save(RefinerD.state_dict(), '%s/step3_RefinerD_epoch_%d_%s.pth' % (opt.outf, epoch, i))


    # do checkpointing
    if (epoch+1) % 50 == 0 and opt.dataset!='lsun':
        torch.save(netG1.state_dict(), '%s/step2_netG1_epoch_%d.pth' % (opt.outf, epoch+1))
        torch.save(netD1.state_dict(), '%s/step2_netD1_epoch_%d.pth' % (opt.outf, epoch+1))
        torch.save(RefinerG.state_dict(), '%s/step3_RefinerG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(RefinerD.state_dict(), '%s/step3_RefinerD_epoch_%d.pth' % (opt.outf, epoch))






