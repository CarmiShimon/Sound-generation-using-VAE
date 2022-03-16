import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class VAE_old(nn.Module):
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture with
    mirrored encoedr and decoder components
    """
    
    def __init__(self, 
                input_shape, 
                conv_filters,
                conv_kernels,
                conv_strides,
                conv_padding,
                deconv_padding,
                latent_space_dim):
        super().__init__()
        self.input_shape = input_shape # [c, w, h] = [1, 256, 256]
        self.conv_filters = conv_filters # i.e., [32, 64, 64, 64]
        self.conv_kernels = conv_kernels # i.e., [3, 3, 3, 3]
        self.conv_strides = conv_strides # i.e., [2, 2, 2, 2]
        self.latent_space_dim = latent_space_dim # 64
        self.conv_padding = conv_padding # [1, 1, 1 , 1]
        self.deconv_padding = deconv_padding # [0, 1, 1 , 1]
        
        
    def conv_block(self, input_shape, filters, kernel, stride, pad, x):
        x = nn.Conv2d(input_shape, filters, stride=stride, kernel_size=kernel, padding=pad)(x)
        x = nn.BatchNorm2d(filters)(x)
        x = nn.LeakyReLU(0.1, inplace=True)(x)
        x = nn.Dropout2d(0.25)(x)
        return x
        
    def deconv_block(self, filters_in, filters_out, kernel, stride, pad, x):
        x = nn.ConvTranspose2d(filters_in, filters_out, stride=stride, kernel_size=kernel, padding=pad)(x)
        x = nn.BatchNorm2d(filters_out)(x)
        x = nn.LeakyReLU(0.1, inplace=True)(x)
        x = nn.Dropout2d(0.25)(x)
        return x
        
    def encoder(self, x):
        idx = 0
        for filters_out, stride, kernel, pad in zip(self.conv_filters, self.conv_kernels, self.conv_strides, self.conv_padding):
            if idx == 0:
                input_shape = self.input_shape[0]
            else:
                input_shape = self.conv_filters[idx-1]
            x = self.conv_block(input_shape, filters_out, stride, kernel, pad, x)
            idx += 1
        
        return x
        
    def decoder(self, x, flatten_shape):
        x = nn.Linear(self.latent_space_dim, flatten_shape)(x)
        w = h = int(np.sqrt(flatten_shape//self.conv_filters[::-1][0]))
        x = torch.reshape(x, (-1, self.conv_filters[::-1][0], w, h))
        for filters_out, kernel, stride, pad in zip(self.conv_filters[:-1][::-1], self.conv_kernels[:-1][::-1], self.conv_strides[:-1][::-1], self.deconv_padding[:-1]):
            print(filters_out)
            x = self.deconv_block(x.shape[1], filters_out, kernel, stride, pad, x)
        
        x = nn.ConvTranspose2d(x.shape[1], self.input_shape[0], stride=self.conv_strides[0], kernel_size=self.conv_kernels[0], padding=self.deconv_padding[-1])(x)
        x = x[:, :, :-1, :-1]
        x = nn.Sigmoid()(x)
        return x
        
    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
            
    def bottleneck(self, x):
        x = torch.flatten(x, start_dim=1)
        flatten_shape = x.shape[1]
        z_mu = nn.Linear(flatten_shape, self.latent_space_dim)(x)
        z_log_var = nn.Linear(flatten_shape, self.latent_space_dim)(x)
        return z_mu, z_log_var, flatten_shape
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z
    
    def forward(self, x):

        x = self.encoder(x)
        z_mean, z_log_var, flatten_shape = self.bottleneck(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded, flatten_shape)
        return encoded, z_mean, z_log_var, decoded
    






class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return x, mu, logvar

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=4)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=8)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 1, 256, 256)
        return x

class VAE(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def forward(self, x):
        x, mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        d = self.decoder(z)
        return x, mean, logvar, d, z
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean