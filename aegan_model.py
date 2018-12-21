
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG1(nn.Module):
    def __init__(self, nz, ngf, batchsize, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(_netG1, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            ResnetBlock(self.ngf * 8, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            ResnetBlock(self.ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            ResnetBlock(self.ngf * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (1) x 32 x 32
        )
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output


class _netD1(nn.Module):
    def __init__(self, ndf):
        super(_netD1, self).__init__()
        self.ndf = ndf
        self.main = nn.Sequential(
            # state size. (1) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)


class _netG2(nn.Module):
    def __init__(self, nz, ngf, nc, batchsize, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(_netG2, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.main = nn.Sequential(
            # state size. (1) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            ResnetBlock(self.ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(self.ngf, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            ResnetBlock(self.ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(self.ngf, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            ResnetBlock(self.ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 256 x 256
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 512 x 512
        )
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output


class _netD2(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD2, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 256 x 256
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)


class _netRS(nn.Module):
    def __init__(self, nc, ndf, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(_netRS, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.ndf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 256 x 256
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.ndf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(self.ndf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (1) x 32 x 32
        )
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output


# Resnet block
class _resnet_block(nn.Module):
    def __init__(self, ngf):
        super(_resnet_block, self).__init__()
        self.ngf = ngf
        self.block = nn.Sequential(
            nn.Conv2d(self.ngf, self.ngf, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, 3, 1, 1, bias=False),
            nn.ReLU(True)
            )
        self.relu = nn.ReLU(True)
        self.apply(weights_init)
        
    def forward(self, input_features):
        output_features = self.block(input_features)
        output_features = output_features + input_features

        return self.relu(output_features)
        

# Refiner
class _Refiner(nn.Module):
    def __init__(self, nc, ngf):
        super(_Refiner, self).__init__()
        self.nc = nc
        self.ngf = ngf

        self.conv_block1 = _resnet_block(self.ngf)
        self.conv_block2 = _resnet_block(self.ngf)
        self.conv_block3 = _resnet_block(self.ngf)
        self.conv_block4 = _resnet_block(self.ngf)

        self.conv1 = nn.Conv2d(self.nc, self.ngf, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(self.ngf, self.nc, 1, 1, 0, bias=False)
        self.relu  = nn.ReLU(True)
        self.tanh  = nn.Tanh()
        self.apply(weights_init)

    def forward(self, input):
        # the input layer
        x = self.conv1(input)
        x = self.relu(x)
        # resnet layers
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        # the output layer
        x = self.conv2(x)
        output = self.tanh(x)

        return output


# class _RefinerD(nn.Module):
#     def __init__(self, ngpu, nc, ndf):
#         super(_RefinerD, self).__init__()
#         self.ngpu = ngpu
#         self.nc = nc
#         self.ndf = ndf
#         self.main = nn.Sequential(
#             # input is (nc) x 512 x 512
#             nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 256 x 256
#             nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 128 x 128
#             nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 64 x 64
#             nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(self.ndf * 8, 1, 1, 1, 0, bias=False),
#             # state size. (1) x 4 x 4
#             # nn.Sigmoid()
#         )
#         self.apply(weights_init)

#     def forward(self, input):
#         output = self.main(input)
#         output = F.log_softmax(output)

#         return output


class _RefinerD(nn.Module):
    def __init__(self, nc, ndf):
        super(_RefinerD, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 512 x 512
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 256 x 256
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1)


class AEGAN_ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2, padding_type='reflect', from_to_num=0, final_tanh=True):
        assert(n_blocks >= 0)
        super(AEGAN_ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if from_to_num>0:
            mult = ngf
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, mult, kernel_size=7, padding=0,
                            bias=use_bias),
                    norm_layer(mult),
                    nn.ReLU(True)]
            for i in range(from_to_num):
                model += [nn.ConvTranspose2d(mult, mult,
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(mult),
                        nn.ReLU(True)]
                for j in range(n_blocks):
                    model += [ResnetBlock(mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]         
        elif from_to_num<0:
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, mult, kernel_size=7, padding=0,
                            bias=use_bias),
                    norm_layer(mult),
                    nn.ReLU(True)]
            for i in range(-from_to_num):
                model += [nn.Conv2d(mult, mult, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                        norm_layer(mult),
                        nn.ReLU(True)]
                for j in range(n_blocks):
                    model += [ResnetBlock(mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]  
        else: # from_to_num=0
            pass
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(mult, output_nc, kernel_size=7, padding=0)]
        if final_tanh:
            model += [nn.Tanh()]  
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class AEGAN_ResnetDecoder(nn.Module):
    def __init__(self, z_dim, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=1, padding_type='reflect', n_downsampling=2):
        assert(n_blocks >= 0)
        super(AEGAN_ResnetDecoder, self).__init__()
        self.z_dim = z_dim
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        mult = 2**n_downsampling

        model = [nn.ConvTranspose2d(z_dim, ngf * mult, kernel_size=4, padding=0), nn.ReLU(True)]

        for i in range(n_downsampling):
            for j in range(n_blocks):
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            mult = mult // 2
        model += [nn.ConvTranspose2d(ngf * mult, output_nc,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias)]
        model += [nn.Tanh()]
        # model += [nn.ReflectionPad2d(3)]
        # model += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        out = self.model(input)
        return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)