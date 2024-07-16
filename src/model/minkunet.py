# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.NCESoftmaxLoss
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from .resnet import ResNetBase

class MinkUNetcustom(ResNetBase):
    BLOCK = BasicBlock
    #PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (24,32,48,64,80,96,112,96,80,64,48,32,16)
    INIT_DIM = 24
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=1, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=1, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=1, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=1, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])
        self.conv5p16s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=1, dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])

        self.conv6p32s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=1, dimension=D)
        self.bn6 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])

        self.conv7p64s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=1, dimension=D)
        self.bn7 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])

        self.convtr7p128s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=3, stride=1, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.PLANES[5] * self.BLOCK.expansion
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.convtr8p64s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[8], kernel_size=3, stride=1, dimension=D)
        self.bntr8 = ME.MinkowskiBatchNorm(self.PLANES[8])

        self.inplanes = self.PLANES[8] + self.PLANES[4] * self.BLOCK.expansion
        self.block9 = self._make_layer(self.BLOCK, self.PLANES[8],
                                       self.LAYERS[8])
        self.convtr9p32s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[9], kernel_size=3, stride=1, dimension=D)
        self.bntr9 = ME.MinkowskiBatchNorm(self.PLANES[9])

        self.inplanes = self.PLANES[9] + self.PLANES[3] * self.BLOCK.expansion
        self.block10 = self._make_layer(self.BLOCK, self.PLANES[9],
                                       self.LAYERS[9])

        self.convtr10p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[10], kernel_size=3, stride=1, dimension=D)
        self.bntr10 = ME.MinkowskiBatchNorm(self.PLANES[10])

        self.inplanes = self.PLANES[10] + self.PLANES[2] * self.BLOCK.expansion
        self.block11 = self._make_layer(self.BLOCK, self.PLANES[10],
                                       self.LAYERS[10])
        self.convtr11p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[11], kernel_size=3, stride=1, dimension=D)
        self.bntr11 = ME.MinkowskiBatchNorm(self.PLANES[11])

        self.inplanes = self.PLANES[11] + self.PLANES[1] * self.BLOCK.expansion
        self.block12 = self._make_layer(self.BLOCK, self.PLANES[11],
                                       self.LAYERS[11])

        self.convtr12p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[12], kernel_size=3, stride=1, dimension=D)
        self.bntr12 = ME.MinkowskiBatchNorm(self.PLANES[12])

        self.inplanes = self.PLANES[12] + self.PLANES[0]
        self.block13 = self._make_layer(self.BLOCK, self.PLANES[12],
                                       self.LAYERS[12])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[12] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, coords, feats):

        x = ME.TensorField(
            features=feats,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device)

        x_sparse = x.sparse()
        out = self.conv0p1s1(x_sparse)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_b4p16 = self.block4(out)

        out = self.conv5p16s2(out_b4p16)
        out = self.bn5(out)
        out = self.relu(out)
        out_b5p32 = self.block5(out)

        out = self.conv6p32s2(out_b5p32)
        out = self.bn6(out)
        out = self.relu(out)
        out_b6p64 = self.block6(out)

        out = self.conv7p64s2(out_b6p64)
        out = self.bn7(out)
        out = self.relu(out)
        out = self.block7(out)

        out = self.convtr7p128s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_b6p64)
        out = self.block8(out)

        out = self.convtr8p64s2(out)
        out = self.bntr8(out)
        out = self.relu(out)

        out = ME.cat(out, out_b5p32)
        out = self.block9(out)

        out = self.convtr9p32s2(out)
        out = self.bntr9(out)
        out = self.relu(out)

        out = ME.cat(out, out_b4p16)
        out = self.block10(out)

        out = self.convtr10p16s2(out)
        out = self.bntr10(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block11(out)

        out = self.convtr11p8s2(out)
        out = self.bntr11(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block12(out)

        out = self.convtr12p4s2(out)
        out = self.bntr12(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block13(out)
        out_field = self.final(out)
        
        return (out_field.slice(x)).F, 

class MinkUNet34Encoder(ResNetBase):
    BLOCK = BasicBlock
    DILATIONS = (1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2)
    PLANES = (32, 64, 128, 256)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, device, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

        self.device = device

    def network_initialization(self, in_channels, out_channels, D):
    
        # Output of the first conv concated to conv6

        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.projection = ME.MinkowskiLinear(self.PLANES[3], 768, bias=False)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, coords, feats):

        x = ME.TensorField(
            features=feats,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device)

        x_sparse = x.sparse()
        
        out = self.conv0p1s1(x_sparse)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        #out = self.relu(out)
        out_feat = self.projection(out)

        return (out_feat.slice(x)).F

class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
    
        # Output of the first conv concated to conv6

        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=False,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x_sparse):
        
        out = self.conv0p1s1(x_sparse)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)
        
        out_field = self.final(out)
        return out_field 

class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


def mink_unet(in_channels=3, out_channels=20, D=3, arch='MinkUNet18A'):
    if arch == 'MinkUNet18A':
        return MinkUNet18A(in_channels, out_channels, D)
    elif arch == 'MinkUNet18B':
        return MinkUNet18B(in_channels, out_channels, D)
    elif arch == 'MinkUNet18D':
        return MinkUNet18D(in_channels, out_channels, D)
    elif arch == 'MinkUNet34A':
        return MinkUNet34A(in_channels, out_channels, D)
    elif arch == 'MinkUNet34B':
        return MinkUNet34B(in_channels, out_channels, D)
    elif arch == 'MinkUNet34C':
        return MinkUNet34C(in_channels, out_channels, D)
    elif arch == 'MinkUNet14A':
        return MinkUNet14A(in_channels, out_channels, D)
    elif arch == 'MinkUNet14B':
        return MinkUNet14B(in_channels, out_channels, D)
    elif arch == 'MinkUNet14C':
        return MinkUNet14C(in_channels, out_channels, D)
    elif arch == 'MinkUNet14D':
        return MinkUNet14D(in_channels, out_channels, D)
    else:
        raise Exception('architecture not supported yet'.format(arch))


if __name__ == '__main__':
    #from tests.python.common import data_loader
    # loss and network
    criterion = nn.CrossEntropyLoss()
    from torchsummary import summary as summary_
    model = MinkUNetcustom(in_channels=3, out_channels=10, D=3)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    '''
    # a data loader must return a tuple of coords, features, and labels.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    optimizer = SGD(net.parameters(), lr=1e-2)

    for i in range(10):
        optimizer.zero_grad()

        # Get new data
        coords, feat, label = data_loader(is_classification=False)
        input = ME.SparseTensor(feat, coordinates=coords, device=device)
        label = label.to(device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, label)
        print('Iteration: ', i, ', Loss: ', loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    # Saving and loading a network
    torch.save(net.state_dict(), 'test.pth')
    net.load_state_dict(torch.load('test.pth'))
    '''