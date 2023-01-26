import math
from torch import nn

trunc_normal_correction = 0.87962566103423978


def fan_in(layer: nn.Linear):
    return layer.weight.size(1)


def fan_out(layer: nn.Linear):
    return layer.weight.size(0)


def linear_init_with_lecun_uniform(layer: nn.Linear):
    scale = math.sqrt(3 / fan_in(layer))
    layer.weight.data.uniform_(-scale, scale)
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


def linear_init_with_lecun_normal(layer: nn.Linear):
    stdev = math.sqrt(1 / fan_in(layer))/trunc_normal_correction
    nn.init.trunc_normal_(layer.weight.data,std = stdev)
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


def linear_init_with_glorot_uniform(layer: nn.Linear):
    scale = math.sqrt(6 / (fan_in(layer) + fan_out(layer)))
    layer.weight.data.uniform_(-scale, scale)
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


def linear_init_with_glorot_normal(layer: nn.Linear):
    stdev = math.sqrt(2 / (fan_in(layer) + fan_out(layer)))/trunc_normal_correction
    nn.init.trunc_normal_(layer.weight.data,std = stdev)
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


def linear_init_with_he_uniform(layer: nn.Linear):
    scale = math.sqrt(6 / fan_in(layer))
    layer.weight.data.uniform_(-scale, scale)
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


def linear_init_with_he_normal(layer: nn.Linear):
    stdev = math.sqrt(2 / fan_in(layer))/trunc_normal_correction
    nn.init.trunc_normal_(layer.weight.data,std = stdev)
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


def linear_init_with_zeros(layer: nn.Linear):
    layer.weight.data.zero_()
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer

