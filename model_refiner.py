
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RefinerD(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_RefinerD, self).__init__()
        self.ngpu = ngpu
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

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)


# For input size input_nc x 512 x 512
class _RefinerG(nn.Module):
    def __init__(self, ngpu, nc, ngf):
        super(_RefinerG, self).__init__()

        self.ngpu = ngpu
        self.nc = nc
        self.ngf = ngf

        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv9 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 8 , ngf * 4, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 4 , ngf * 2, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2 , ngf, 4, 2, 1)
        self.dconv9 = nn.ConvTranspose2d(ngf , nc, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(ngf)
        self.batch_norm2 = nn.BatchNorm2d(ngf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ngf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ngf * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 512 x 512
        e1 = self.conv1(input)
        # state size is (ngf) x 256 x 256
        e2 = self.batch_norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 128 x 128
        e3 = self.batch_norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 64 x 64
        e4 = self.batch_norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 32 x 32
        e5 = self.batch_norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 16 x 16
        e6 = self.batch_norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 8 x 8
        e7 = self.batch_norm8(self.conv7(self.leaky_relu(e6)))
		# state size is (ngf x 8) x 4 x 4
        e8 = self.batch_norm8(self.conv8(self.leaky_relu(e7)))
        # state size is (ngf x 8) x 2 x 2
        # No batch norm on output of Encoder
        e9 = self.conv8(self.leaky_relu(e8))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1 = self.dropout(self.batch_norm8(self.dconv1(self.relu(e9))))
        # state size is (ngf x 8) x 2 x 2
        d2 = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1))))
        # state size is (ngf x 8) x 4 x 4
        d3 = self.dropout(self.batch_norm8(self.dconv3(self.relu(d2))))
        # state size is (ngf x 8) x 8 x 8
        d4 = self.batch_norm8(self.dconv4(self.relu(d3)))
        # state size is (ngf x 8) x 16 x 16
        d5 = self.batch_norm8(self.dconv5(self.relu(d4)))
        # state size is (ngf x 8) x 32 x 32
        d6 = self.batch_norm4(self.dconv6(self.relu(d5)))
        # state size is (ngf x 4) x 64 x 64
        d7 = self.batch_norm2(self.dconv7(self.relu(d6)))
        # state size is (ngf x 2) x 128 x 128
        d8 = self.batch_norm(self.dconv8(self.relu(d7)))
        # state size is (ngf) x 256 x 256
        d9 = self.dconv9(self.relu(d8))
        # state size is (nc) x 512 x 512
        output = self.tanh(d9)
        return output











