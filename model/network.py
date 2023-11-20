import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm3d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x
  ######################################################################### 
class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm3d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()

        self.c0 = nn.Conv3d(in_channels=in_c,  out_channels=out_c, kernel_size = kernel_size , padding='same')
        self.br11 = batchnorm_relu(out_c)

        self.c11 = nn.Conv3d(in_channels=out_c,  out_channels=out_c, kernel_size = kernel_size , padding='same')
        self.br12 = batchnorm_relu(out_c)
        self.c12 = nn.Conv3d(in_channels=out_c,  out_channels=out_c, kernel_size = kernel_size , padding='same')


    def forward(self, inputs):
        x = self.c0(inputs)

        x = self.br11(x)

        x = self.c11(x)
        x = self.br12(x)
        x = self.c12(x)
        s = self.c0(inputs)
        add1 = x + s

        return add1

class residual_block_no_first_conv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.c1 = nn.Conv3d(in_c, out_c, kernel_size=(1,1,1), padding=0, stride=1)

        self.br11 = batchnorm_relu(in_c)
        self.c11 = nn.Conv3d(in_channels=out_c,  out_channels=out_c, kernel_size = kernel_size , padding='same')
        self.br12 = batchnorm_relu(in_c)
        self.c12 = nn.Conv3d(in_channels=out_c,  out_channels=out_c, kernel_size = kernel_size , padding='same')


    def forward(self, inputs):
        x = self.br11(inputs)

        x = self.c11(x)
        x = self.br12(x)
        x = self.c12(x)
        s = self.c1(inputs)
        add1 = x + s

        return add1
class build_resunet(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = (3,3,3)

        """ Encoders  """
        self.r1 = residual_block(1, 8, kernel_size=kernel_size )
        self.mp1 = nn.MaxPool3d(kernel_size= (2,2,2), stride= (2,2,2))

        self.r2 = residual_block(8, 16,kernel_size= kernel_size )
        self.mp2 = nn.MaxPool3d(kernel_size= (2,2,2), stride= (2,2,2))

        self.r3 = residual_block(16, 32,kernel_size= (3,3,3) )
        self.mp3 = nn.MaxPool3d(kernel_size= (2,2,2), stride= (2,2,2))

        self.r4 = residual_block(32, 64,kernel_size= (3,3,3) )

        self.r5 = residual_block_no_first_conv(64, 64,kernel_size= (3,3,3) )
        self.mp5 =  nn.MaxPool3d(kernel_size= (2,2,2), stride= (2,2,2))

        self.r6 = residual_block(64, 128,kernel_size= (3,3,3) )

        self.r7 = residual_block_no_first_conv(128, 128,kernel_size= (3,3,3) )

        """ Transpose 1 """
        self.trans1 = nn.ConvTranspose3d(in_channels=128,  out_channels=64, kernel_size = (2, 2, 2) , stride = (2,2,2))

        """ Decoders 1 , 2 """
        self.d1 = residual_block(128, 64,kernel_size= (3,3,3) )
        
        self.d2 = residual_block_no_first_conv(64, 64,kernel_size= (3,3,3) )

        """ Transpose 2 """

        self.trans2 = nn.ConvTranspose3d(in_channels=64,  out_channels=32, kernel_size = (2, 2, 2) , stride = (2,2,2))

        """ Decoder 3 """

        self.d3 = residual_block(64, 32 ,kernel_size= (3,3,3) )

        """ fin """
        self.f1 = nn.Conv3d(48, 16, kernel_size=(3,3,3), padding='same')
        self.f2 = nn.Conv3d(16, 16, kernel_size=(3,3,3), padding='same')
        self.f3 = nn.Conv3d(16, 12, kernel_size=(3,3,3), padding='same')
        self.relu = nn.ReLU()
        


    def forward(self, inputs):
        """ Encoders """
        add1 = self.r1(inputs)
        mp1 = self.mp1(add1) ## 8
        #print("mp1", mp1.size())

        add2 = self.r2(mp1)
        #print("add2", add2.size())

        mp2 = self.mp2(add2) ## keep it (skip connection between encoder and decoder) -- 16
        #print("mp2", mp2.size())

        add3 = self.r3(mp2) ## keep it (skip connection between encoder and decoder) -- 32
        #print("add3", add3.size())

        mp3 = self.mp3(add3)  
        #print("mp3", mp3.size())

        add4 = self.r4(mp3) ## -- 64
        #print("add4", add4.size())

        add5 = self.r5(add4) ## keep it (skip connection between encoder and decoder)
        #print("add5", add5.size())

        mp5 = self.mp5(add5)
        #print("mp5", mp5.size())

        add6 = self.r6(mp5) ## 128
        #print("add6", add6.size())

        add7 = self.r7(add6)
        #print("add7", add7.size())

        trans1 = self.trans1(add7) ## 64
        #print("trans1", trans1.size())

        """ Concat"""
        concat1 = torch.cat([trans1, add5], axis=1) ## 128
        #print("concat1", concat1.size())

        """ Decoders """
        d1 = self.d1(concat1) ## 64
        #print("d1", d1.size())

        d2 = self.d2(d1)
        #print("d2", d2.size())

        trans2 = self.trans2(d2) ## 64
        #print("trans2", trans2.size())

        """ Concat"""
        concat2 = torch.cat([trans2, add3], axis=1) ## 64
        #print("concat2", concat2.size())

        d3 = self.d3(concat2) ## 32
        #print("d3", d3.size())

        concat3 = torch.cat([d3 , mp2], axis=1) ## 48
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
    return build_resunet()

