import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BatchNormRelu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm3d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, dilation=1):
        super().__init__()

        self.br11 = BatchNormRelu(out_c)

        self.c11 = nn.Conv3d(in_channels=out_c, out_channels=out_c, kernel_size=kernel_size,
                             padding='same', dilation=dilation)
        self.br12 = BatchNormRelu(out_c)
        self.c12 = nn.Conv3d(in_channels=out_c, out_channels=out_c, kernel_size=kernel_size,
                             padding='same', dilation=dilation)
        
    def forward(self, inputs):
        x = self.br11(inputs)

        x = self.c11(x)
        x = self.br12(x)
        x = self.c12(x)
        s = inputs
        add1 = x + s

        return add1
class BuildResUNet(nn.Module):
    def __init__(self):
        super(BuildResUNet, self).__init__()
        kernel_size = (3, 3, 3)
        
        # Trunk part
        """ Encoders  """
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding='same')
        self.r1 = ResBlock(1, 8, kernel_size=kernel_size)
        self.mp1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding='same')
        self.r2 = ResBlock(8, 16, kernel_size=kernel_size)
        self.mp2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ## Now start U part.
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding='same')
        self.r3 = ResBlock(16, 32, kernel_size=(3, 3, 3))
        self.mp3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding='same')
        self.r4 = ResBlock(32, 64, kernel_size=(3, 3, 3), dilation=1)  

        self.r5 = ResBlock(64, 64, kernel_size=(3, 3, 3), dilation=1)  
        self.mp5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv6 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding='same')
        self.r6 = ResBlock(128, 128, kernel_size=(3, 3, 3), dilation=1)  

        self.r7 = ResBlock(128, 128, kernel_size=(3, 3, 3), dilation=1) 

        """ Transpose 1 """
        self.trans1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        """ Decoders 1 , 2 """
        self.conv7 = nn.Conv3d(128, 64, (3, 3, 3), padding='same')
        self.d1 = ResBlock(64, 64, kernel_size=(3, 3, 3))

        self.d2 = ResBlock(64, 64, kernel_size=(3, 3, 3))

        """ Transpose 2 """

        self.trans2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        """ Decoder 3 """
        self.conv8 = nn.Conv3d(64, 32, (3, 3, 3), padding='same')
        self.d3 = ResBlock(32, 32, kernel_size=(3, 3, 3))

        """ fin """
        self.f1 = nn.Conv3d(48, 16, kernel_size=1, padding='same')
        self.f2 = nn.Conv3d(16, 16, kernel_size=1, padding='same')
        self.f3 = nn.Conv3d(16, 12, kernel_size=1, padding='same')
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
    
        """ Encoders """
        conv1 = self.conv1(inputs)
        add1 = self.r1(conv1)
        mp1 = self.mp1(add1)  ## 8
        #print("mp1", mp1.size())

        conv2 = self.conv2(mp1)
        add2 = self.r2(conv2)
        mp2 = self.mp2(add2)  ## keep it (skip connection between encoder and decoder) -- 16
        #print("mp2", mp2.size())

        conv3 = self.conv3(mp2)
        add3 = self.r3(conv3)  ## keep it (skip connection between encoder and decoder) -- 32
        mp3 = self.mp3(add3)
        #print("mp3", mp3.size())

        conv4 = self.conv4(mp3)
        add4 = self.r4(conv4)  ## -- 64
        add5 = self.r5(add4)  ## keep it (skip connection between encoder and decoder)
        mp5 = self.mp5(add5)
        #print("mp5", mp5.size())

        conv6 = self.conv6(mp5)
        add6 = self.r6(conv6)  ## 128
        #print("add6", add6.size())

        add7 = self.r7(add6)
        #print("add7", add7.size())

        trans1 = self.trans1(add7)  ## 64
        #print("trans1", trans1.size())

        """ Concat"""
        concat1 = torch.cat([trans1, add5], axis=1)  ## 128
        #print("concat1", concat1.size())

        """ Decoders """
        conv7 = self.conv7(concat1)
        d1 = self.d1(conv7)  ## 64
        #print("d1", d1.size())

        d2 = self.d2(d1)
        #print("d2", d2.size())

        trans2 = self.trans2(d2)  ## 64
        #print("trans2", trans2.size())

        """ Concat"""
        concat2 = torch.cat([trans2, add3], axis=1)  ## 64
        #print("concat2", concat2.size())

        conv8 = self.conv8(concat2)
        d3 = self.d3(conv8)  ## 32
        #print("d3", d3.size())

        concat3 = torch.cat([d3, mp2], axis=1)  ## 48
        #print("concat3", concat3.size())

        """ output """
        f1 = self.f1(concat3)
        f1 = self.relu(f1)
        #print("f1", f1.size())

        f2 = self.f2(f1)
        f2 = self.relu(f2)
        #print("f2", f2.size())

        f3 = self.f3(f2)
        #print("f3", f3.size())
        output = f3
        return output
        
        
def get_res_unet():
    return BuildResUNet()

