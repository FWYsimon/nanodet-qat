from torch import nn
# from torchvision.models.utils import load_state_dict_from_url
from .ghost_pan import GhostBlocks, GhostPAN
from ..backbone.ghostnet import GhostBottleneck, GhostModule
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

from ..module.conv import ConvModule, DepthwiseConvModule

import torch
from typing import List


class QuantizableGhostModule(GhostModule):
    def __init__(self, *args, **kwargs):
        super(QuantizableGhostModule, self).__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # out = torch.cat([x1, x2], dim=1)
        out = self.skip_add.cat([x1, x2], dim=1)
        return out
    
    def fuse_model(self):
        if self.activation:
            fuse_modules(self.primary_conv, ["0", "1", "2"], inplace=True)
            fuse_modules(self.cheap_operation, ["0", "1", "2"], inplace=True)
        else:
            fuse_modules(self.primary_conv, ["0", "1"], inplace=True)
            fuse_modules(self.cheap_operation, ["0", "1"], inplace=True)
        


class QuantizableGhostBottleneck(GhostBottleneck):
    def __init__(self, *args, **kwargs):
        super(QuantizableGhostBottleneck, self).__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        # print(self.stride)
        # if self.stride > 1:
        #     x = self.conv_dw(x)
        #     x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        # x += self.shortcut(residual)
        x = self.skip_add.add(x, self.shortcut(residual))
        return x

    def fuse_model(self):
        for idx in range(len(self.shortcut)):
            if type(self.shortcut[idx]) == nn.Conv2d:
                fuse_modules(self.shortcut, [str(idx), str(idx + 1)], inplace=True)
        # fuse_modules(self.primary_conv, ["0", "1", "2"], inplace=True)
        # fuse_modules(self.cheap_operation, ["0", "1", "2"], inplace=True)

class QuantizableGhostBlocks(GhostBlocks):
    def __init__(self, *args, **kwargs):
        super(QuantizableGhostBlocks, self).__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.blocks(x)
       #  print(self.use_res)

        # if self.use_res:
        #     out = self.skip_add.add(out, self.reduce_conv(x))
            # out = out + self.reduce_conv(x)
        return out

    # def fuse_model(self):
    #     for idx in range(len(self.conv)):
    #         if type(self.conv[idx]) == nn.Conv2d:
    #             fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)
    # def fuse_model(self):
    #     for idx in range(len(self.shortcut)):
    #         if type(self.shortcut[idx]) == nn.Conv2d:
    #             fuse_modules(self.shortcut, [str(idx), str(idx + 1)], inplace=True)

class QuantizableGhostPAN(GhostPAN):
    def __init__(self, *args, **kwargs):
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super(QuantizableGhostPAN, self).__init__(block=QuantizableGhostBlocks, bottleneck=QuantizableGhostBottleneck, module=QuantizableGhostModule, *args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, inputs:List[torch.Tensor]):
    # def forward(self, inputs):        
        # x = self._forward_impl(x)
        # print(inputs.size())
        # assert len(inputs) == len(self.in_channels)
        # inputs = []
        # for i, (input_x, reduce) in enumerate(zip(inputs, self.reduce_layers)):
        #     inputs.append[reduce(input_x)]
        outputs = []
        # b = inputs.shape[0]
        # shapes = [[b,32,40,40],[b,96,20,20],[b,1280,10,10]]
        # inputs = inputs.split([32*40*40,96*20*20,1280*10*10], dim=1)
        for i, reduce in enumerate(self.reduce_layers):
            # input_x = inputs[i].view(shapes[i])
            input_x = inputs[i]
            outputs.append(reduce(input_x))
        # inputs = [
        #     reduce(input_x) for input_x, reduce in zip(inputs, self.reduce_layers)
        # ]
        # top-down path
        inputs = outputs
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            for i, top_down_block in enumerate(self.top_down_blocks):
                if i == len(self.in_channels) - 1 - idx:
                    inner_out = top_down_block(self.skip_add.cat([upsample_feat, feat_low], 1))
                    inner_outs.insert(0, inner_out)

            # inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
            #     # torch.cat([upsample_feat, feat_low], 1)
            #     self.skip_add.cat([upsample_feat, feat_low], 1)
            # )
            # inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]

            # downsample_feat = self.downsamples[idx](feat_low)
            # out = self.bottom_up_blocks[idx](
            #     # torch.cat([downsample_feat, feat_height], 1)
            #     self.skip_add.cat([downsample_feat, feat_height], 1)
            # )
            # outs.append(out)

            for i, (downsample, bottom_up_block)in enumerate(zip(self.downsamples, self.bottom_up_blocks)):
                if i == idx:
                    downsample_feat = downsample(feat_low)
                    out = bottom_up_block(self.skip_add.cat([downsample_feat, feat_height], 1))
                    outs.append(out)

        # extra layers
        for extra_in_layer, extra_out_layer in zip(
            self.extra_lvl_in_conv, self.extra_lvl_out_conv
        ):
            # out = self.skip_add.add(out, self.reduce_conv(x))
            # outs.append(extra_in_layer(inputs[-1]) + extra_out_layer(outs[-1]))
            outs.append(self.skip_add.add(extra_in_layer(inputs[-1]),  extra_out_layer(outs[-1])))

        return outs

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvModule:
                fuse_modules(m, ['conv', 'norm_layer', 'act'], inplace=True)
            if type(m) == DepthwiseConvModule:
                fuse_modules(m, ['depthwise', 'dwnorm'], inplace=True)
                fuse_modules(m, ['pointwise', 'pwnorm'], inplace=True)
            if type(m) == QuantizableGhostBottleneck:
                m.fuse_model()
            if type(m) == QuantizableGhostModule:
                m.fuse_model()
            
            # if type(m) == QuantizableGhostBlocks:
            #     m.fuse_model()
