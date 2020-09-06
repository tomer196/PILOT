import torch
from torch import nn
from torch.nn import functional as F


class ComplexConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            ComplexConv2d(in_chans, out_chans, kernel_size=3, padding=1),
            ComplexInstanceNorm2d(out_chans),
            nn.ReLU(),
            # nn.Dropout2d(drop_prob),
            ComplexConv2d(out_chans, out_chans, kernel_size=3, padding=1),
            ComplexInstanceNorm2d(out_chans),
            nn.ReLU(),
            # nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class ComplexUnetModel(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.in_norm = ComplexInstanceNorm2d(in_chans)

        self.down_sample_layers = nn.ModuleList([ComplexConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ComplexConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ComplexConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ComplexConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ComplexConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            ComplexConv2d(ch, ch // 2, kernel_size=1),
            ComplexConv2d(ch // 2, out_chans, kernel_size=1),
            ComplexConv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        output = self.in_norm(output)
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = Complex_max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = Complex_interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)


def Complex_max_pool2d(input, kernel_size=2):
    real = F.max_pool2d(input[..., 0], kernel_size=kernel_size)
    imaginary = F.max_pool2d(input[..., 1], kernel_size=kernel_size)
    output = torch.stack((real, imaginary), dim=-1)
    return output


def Complex_interpolate(input, scale_factor=2, mode='bilinear', align_corners=False):
    real = F.interpolate(input[..., 0], scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    imaginary = F.interpolate(input[..., 1], scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    output = torch.stack((real, imaginary), dim=-1)
    return output


class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output

class ComplexInstanceNorm2d(nn.Module):
    def __init__(self, chans):
        super().__init__()
        self.in_re = nn.InstanceNorm2d(chans)
        self.in_im = nn.InstanceNorm2d(chans)

    def forward(self, x):
        real = self.in_re(x[..., 0])
        imag = self.in_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output
#
# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False,
#                  padding_mode="zeros"):
#         super().__init__()
#         if padding is None:
#             padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
#
#         if complex:
#             conv = ComplexConv2d
#             bn = ComplexBatchNorm2d
#         else:
#             conv = nn.Conv2d
#             bn = nn.BatchNorm2d
#
#         self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
#                          padding_mode=padding_mode)
#         self.bn = bn(out_channels)
#         self.relu = nn.LeakyReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
#
# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), complex=False):
#         super().__init__()
#         if complex:
#             tconv = ComplexConvTranspose2d
#             bn = ComplexBatchNorm2d
#         else:
#             tconv = nn.ConvTranspose2d
#             bn = nn.BatchNorm2d
#
#         self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.bn = bn(out_channels)
#         self.relu = nn.LeakyReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.transconv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
#
# class ComplexUNet(nn.Module):
#     def __init__(self, input_channels=1,
#                  complex=True,
#                  model_complexity=45,
#                  model_depth=20,
#                  padding_mode="zeros"):
#         super().__init__()
#
#         # if complex:
#         #     model_complexity = int(model_complexity // 1.414)
#
#         self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
#         self.encoders = []
#         self.model_length = model_depth // 2
#
#         for i in range(self.model_length):
#             module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
#                              stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=complex,
#                              padding_mode=padding_mode)
#             self.add_module("encoder{}".format(i), module)
#             self.encoders.append(module)
#
#         self.decoders = []
#
#         for i in range(self.model_length):
#             module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1],
#                              kernel_size=self.dec_kernel_sizes[i],
#                              stride=self.dec_strides[i], padding=self.dec_paddings[i], complex=complex)
#             self.add_module("decoder{}".format(i), module)
#             self.decoders.append(module)
#
#         if complex:
#             conv = ComplexConv2d
#         else:
#             conv = nn.Conv2d
#
#         linear = conv(self.dec_channels[-1], 1, 1)
#
#         self.add_module("linear", linear)
#         self.complex = complex
#         self.padding_mode = padding_mode
#
#         self.decoders = nn.ModuleList(self.decoders)
#         self.encoders = nn.ModuleList(self.encoders)
#
#     def forward(self, x):
#         # if self.complex:
#         #     x = bd['X']
#         # else:
#         #     x = bd['mag_X']
#         # go down
#         xs = []
#         for i, encoder in enumerate(self.encoders):
#             xs.append(x)
#             # print("x{}".format(i), x.shape)
#             x = encoder(x)
#         # xs : x0=input x1 ... x9
#
#         # print(x.shape)
#         p = x
#         for i, decoder in enumerate(self.decoders):
#             p = decoder(p)
#             if i == self.model_length - 1:
#                 break
#             # print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {xs[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
#             p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)
#
#         # print(p.shape)
#         mask = self.linear(p)
#         mask = torch.tanh(mask)
#         # bd['M_hat'] = mask
#         return mask
#
#     def set_size(self, model_complexity, model_depth=20, input_channels=1):
#         if model_depth == 10:
#             self.enc_channels = [input_channels,
#                                  model_complexity,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  ]
#             self.enc_kernel_sizes = [(7, 5),
#                                      (7, 5),
#                                      (5, 3),
#                                      (5, 3),
#                                      (5, 3)]
#             self.enc_strides = [(2, 2),
#                                 (2, 2),
#                                 (2, 2),
#                                 (2, 2),
#                                 (2, 1)]
#             self.enc_paddings = [None,
#                                  None,
#                                  None,
#                                  None,
#                                  None]
#
#             self.dec_channels = [0,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2]
#
#             self.dec_kernel_sizes = [(4, 3),
#                                      (4, 4),
#                                      (6, 4),
#                                      (6, 4),
#                                      (7, 5)]
#
#             self.dec_strides = [(2, 1),
#                                 (2, 2),
#                                 (2, 2),
#                                 (2, 2),
#                                 (2, 2)]
#
#             self.dec_paddings = [(1, 1),
#                                  (1, 1),
#                                  (2, 1),
#                                  (2, 1),
#                                  (2, 1)]
#
#         elif model_depth == 20:
#             self.enc_channels = [input_channels,
#                                  model_complexity,
#                                  model_complexity,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  128]
#
#             self.enc_kernel_sizes = [(7, 1),
#                                      (1, 7),
#                                      (6, 4),
#                                      (7, 5),
#                                      (5, 3),
#                                      (5, 3),
#                                      (5, 3),
#                                      (5, 3),
#                                      (5, 3),
#                                      (5, 3)]
#
#             self.enc_strides = [(1, 1),
#                                 (1, 1),
#                                 (2, 2),
#                                 (2, 1),
#                                 (2, 2),
#                                 (2, 1),
#                                 (2, 2),
#                                 (2, 1),
#                                 (2, 2),
#                                 (2, 1)]
#
#             self.enc_paddings = [(3, 0),
#                                  (0, 3),
#                                  None,
#                                  None,
#                                  None,
#                                  None,
#                                  None,
#                                  None,
#                                  None,
#                                  None]
#
#             self.dec_channels = [0,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2,
#                                  model_complexity * 2]
#
#             self.dec_kernel_sizes = [(4, 3),
#                                      (4, 2),
#                                      (4, 3),
#                                      (4, 2),
#                                      (4, 3),
#                                      (4, 2),
#                                      (6, 3),
#                                      (7, 5),
#                                      (1, 7),
#                                      (7, 1)]
#
#             self.dec_strides = [(2, 1),
#                                 (2, 2),
#                                 (2, 1),
#                                 (2, 2),
#                                 (2, 1),
#                                 (2, 2),
#                                 (2, 1),
#                                 (2, 2),
#                                 (1, 1),
#                                 (1, 1)]
#
#             self.dec_paddings = [(1, 1),
#                                  (1, 0),
#                                  (1, 1),
#                                  (1, 0),
#                                  (1, 1),
#                                  (1, 0),
#                                  (2, 1),
#                                  (2, 1),
#                                  (0, 3),
#                                  (3, 0)]
#         else:
#             raise ValueError("Unknown model depth : {}".format(model_depth))