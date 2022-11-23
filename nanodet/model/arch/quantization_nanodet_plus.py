# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import torch

# from .nanodet_plus import NanoDetPlus
from .one_stage_detector import OneStageDetector
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch import nn

from ..head import build_head
from ..head.quantization_nanodet_plus_head import QuantizableLoss

from ..module.conv import ConvModule, DepthwiseConvModule

class QuantizableNanoDetPlus(OneStageDetector):
    def __init__(
        self, 
        backbone,
        fpn,
        aux_head,
        head,
        detach_epoch=0,
    ):
        super(QuantizableNanoDetPlus, self).__init__(
            backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()

        # self.loss = QuantizableLoss(**head)
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head = build_head(aux_head)
        self.detach_epoch = detach_epoch

    def forward(self, x):
        x = self.quant(x)
        x = self.backbone(x)
        # if hasattr(self, "fpn"):
        x = self.fpn(x)
        # if hasattr(self, "head"):
        x = self.head(x)
        x = self.dequant(x)
        return x

    def forward_train(self, gt_meta):
        img = gt_meta["img"]
        img = self.quant(img)
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        if self.epoch >= self.detach_epoch:
            # aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            fd = [f.detach() for f in feat]
            aux_fpn_feat = self.aux_fpn(fd[0], fd[1], fd[2])
            dual_fpn_feat = (
                # torch.cat([f.detach(), aux_f], dim=1)
                # for f, aux_f in zip(fpn_feat, aux_fpn_feat)

                self.skip_add.cat([f.detach(), aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            dual_fpn_feat = (
                # torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
                self.skip_add.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        head_out = self.head(fpn_feat)
        head_out = self.dequant(head_out)
        aux_head_out = self.aux_head(dual_fpn_feat)
        aux_head_out = self.dequant(aux_head_out)
        # loss, loss_states = self.loss(head_out, gt_meta, aux_preds=aux_head_out)
        # loss, loss_states = None, None
        return head_out, aux_head_out
    
    def fuse_model(self):
        self.backbone.fuse_model()
        self.fpn.fuse_model()
        self.head.fuse_model()
        # for m in self.modules():
        #     if type(m) == ConvModule:
        #         if m.norm_name == "bn":
        #             fuse_modules(m, ['conv', 'norm_layer', 'act'], inplace=True)
        #     if type(m) == DepthwiseConvModule:
        #         fuse_modules(m, ['depthwise', 'dwnorm'], inplace=True)
        #         fuse_modules(m, ['pointwise', 'pwnorm'], inplace=True)
            # if type(m) == QuantizableInvertedResidual:
            #     m.fuse_model()
