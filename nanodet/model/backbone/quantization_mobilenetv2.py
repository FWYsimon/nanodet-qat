from torch import nn
from .mobilenetv2 import InvertedResidual, ConvBNReLU, MobileNetV2
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
# from .utils import _replace_relu, quantize_model


__all__ = ['QuantizableMobileNetV2', 'mobilenet_v2']




class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super(QuantizableInvertedResidual, self).__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)
            # if type(self.conv[idx]) == ConvBNReLU:
            #     fuse_modules(self.conv[idx], ['0', '1', '2'], inplace=True)


class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, *args, **kwargs):
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super(QuantizableMobileNetV2, self).__init__(block=QuantizableInvertedResidual, *args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        # x = self.quant(x)
        x = self.first_layer(x)
        output = []
        x = self.stage0(x)
        if 0 in self.out_stages:
            output.append(x)
        x = self.stage1(x)
        if 1 in self.out_stages:
            output.append(x)
        x = self.stage2(x)
        if 2 in self.out_stages:
            output.append(x)
        x = self.stage3(x)
        if 3 in self.out_stages:
            output.append(x)
        x = self.stage4(x)
        if 4 in self.out_stages:
            output.append(x)
        x = self.stage5(x)
        if 5 in self.out_stages:
            output.append(x)
        x = self.stage6(x)
        if 6 in self.out_stages:
            output.append(x)
        return output

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == QuantizableInvertedResidual:
                m.fuse_model()
