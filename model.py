from torch import nn
import torch

import torch.nn.functional as F
import math
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
class Conv(nn.Module):
    def __init__(self, input_size, out_size, kernel_size, num_res_units = 0):
        super(Conv, self).__init__()

        self.num_res_units = num_res_units
        strides = 1
      
        self.conv_1 = ResidualUnit(
            1,
            input_size,
            out_size,
            strides=strides,
            kernel_size=kernel_size,
            subunits=num_res_units,
            norm='batch',
            dropout=0.1,
            bias=True,
        )

        
    def forward(self,x):
        x = self.conv_1(x)
        return x

class Up_sample(nn.Module):

    def __init__(self, input_size, out_size, kernel_size=5):
        super(Up_sample, self).__init__()

    
        self.upsample = Convolution(
                1,
                input_size,
                out_size,
                strides=2,
                kernel_size=kernel_size,
                norm='batch',
                dropout=0.1,
                bias=True,
                conv_only=False,
                is_transposed=True,

            )

  

        self.conv_1 = Convolution(
                1,
                input_size,
                out_size,
                strides=1,
                kernel_size=kernel_size,

                norm='batch',
                dropout=0.1,
                bias=True,

            )
    def forward(self, x1, x2):

        x1 = self.upsample(x1)

        out = torch.cat((x1, x2), dim=1)

        out = self.conv_1(out)


        return out



class FR_Net(nn.Module):
  def __init__(self,input_channel=4,layer=32,kernel_size=3):
    super(FR_Net, self).__init__()
    self.layer = input_channel
    num_res_units = 2
    self.encoder_1 = Conv(input_channel, layer, kernel_size,num_res_units= num_res_units)
    self.encoder_2 = Conv(layer, layer*2, kernel_size,num_res_units= num_res_units)
    self.encoder_3 = Conv(layer*2, layer*4, kernel_size,num_res_units= num_res_units)
    self.encoder_4 = Conv(layer*4, layer*8, kernel_size,num_res_units= num_res_units)
    self.encoder_5 = Conv(layer*8, layer*16, kernel_size,num_res_units= num_res_units)


    encoder_layers = nn.TransformerEncoderLayer(d_model=layer*16, nhead=8, dim_feedforward=128,#layer*16#input_size//16
                                        dropout=0.1,batch_first = True)#layer*8
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=2,)# norm=nn.LayerNorm(layer*8)

    Up_func = Up_sample

    self.decoder4 = Up_func(layer*16, layer*8, kernel_size)
    self.decoder3 = Up_func(layer*8, layer*4, kernel_size)
    self.decoder2 = Up_func(layer*4, layer*2, kernel_size)
    self.decoder1 = Up_func(layer*2, layer, kernel_size)

    self.maxpool = nn.MaxPool1d(kernel_size=2,stride=2,padding=0,ceil_mode=True)


    self.final = Convolution(
            1,
            layer,
            1,
            strides=1,
            kernel_size=kernel_size,
          
            norm='batch',
            dropout=0.1,
            bias=True,
            conv_only=True,
            is_transposed=True,
        
        )
      
  def forward(self,x):
    x = x.to(torch.float32)

    out_1 = self.encoder_1(x)
    x = self.maxpool(out_1)

    out_2 = self.encoder_2(x)
    x = self.maxpool(out_2)
    #print(x.shape)
    out_3 = self.encoder_3(x)
    x = self.maxpool(out_3)
    #print(x.shape)
    out_4 = self.encoder_4(x)
    x = self.maxpool(out_4)
    x = self.encoder_5(x)

    x_t = x.permute(0, 2, 1)

    x_t = self.transformer_encoder(x_t)
    x_t = x_t.permute(0, 2, 1)
    x = x_t+ x


    """ Expanding """
    x = self.decoder4(x, out_4)
    x = self.decoder3(x, out_3)
    #print(x.shape)
    x = self.decoder2(x, out_2)
    x = self.decoder1(x, out_1)
    x = self.final(x)

    return x