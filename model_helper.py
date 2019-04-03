import torch.nn as nn


def gen_conv_module(in_ch, out_ch, ksize=3, stride=1, padding=1, dilation=1, activation=nn.ELU()):
    return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation),
                activation
    )


def gen_down_module(in_ch, out_ch, activation=nn.ELU()):
    layers = []
    layers.append(gen_conv_module(in_ch, out_ch, ksize=5))

    curr_dim = out_ch
    for i in range(2):
        layers.append(gen_conv_module(curr_dim, curr_dim * 2, ksize=3, stride=2))
        layers.append(gen_conv_module(curr_dim * 2, curr_dim * 2))
        curr_dim *= 2

    layers.append(gen_conv_module(curr_dim, curr_dim, activation=activation))

    return nn.Sequential(*layers)


def gen_dilation_module(in_ch, out_ch):
    layers = []
    dilation = 1
    for i in range(4):
        dilation *= 2
        layers.append(gen_conv_module(in_ch, out_ch, dilation=dilation, padding=dilation))
    return nn.Sequential(*layers)


def gen_up_module(in_ch, out_ch, is_refine=False):
    layers = []
    curr_dim = in_ch
    if is_refine:
        layers.append(gen_conv_module(curr_dim, curr_dim//2))
        curr_dim //= 2
    else:
        layers.append(gen_conv_module(curr_dim, curr_dim))

    for i in range(2):
        layers.append(gen_conv_module(curr_dim, curr_dim))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(gen_conv_module(curr_dim, curr_dim//2))
        curr_dim //= 2

    layers.append(gen_conv_module(curr_dim, curr_dim//2))
    layers.append(gen_conv_module(curr_dim//2, out_ch, activation=nn.Threshold(0, 0)))

    return nn.Sequential(*layers)


def gen_flatten_module(in_ch, out_ch, is_local=True):
    layers = []
    layers.append(gen_conv_module(in_ch, out_ch, ksize=5, stride=2, padding=2, activation=nn.LeakyReLU()))
    curr_dim = out_ch

    for i in range(2):
        layers.append(gen_conv_module(curr_dim, curr_dim*2, ksize=5, stride=2, padding=2, activation=nn.LeakyReLU()))
        curr_dim *= 2

    if is_local:
        layers.append(gen_conv_module(curr_dim, curr_dim*2, ksize=5, stride=2, padding=2, activation=nn.LeakyReLU()))
    else:
        layers.append(gen_conv_module(curr_dim, curr_dim, ksize=5, stride=2, padding=2, activation=nn.LeakyReLU()))

    return nn.Sequential(*layers)
