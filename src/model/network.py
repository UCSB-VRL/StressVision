#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 2 13:11:19 2020

@author: satish
"""
import torch
import torch.nn as nn
import time
import torchvision

import torch
import torch.nn as nn

print("Loading ResNet Model")
start_time = time.time()
model = torch.hub.load("facebookresearch/detr:main", "detr_resnet50", pretrained=True)


class NestedTensor(object):
    def __init__(self, tensors, mask=None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class Merge_LSTM(nn.Module):
    def __init__(self, *args):
        super().__init__()

        self.spatial_backbone = model.backbone
        self.encoder = model.transformer.encoder
        self.proj = model.input_proj
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        for name, module in self.classifier.named_children():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                nn.init.normal_(module.bias)

    def forward(self, x):
        x = x.squeeze(0)
        masks = torch.ones(x.shape).squeeze(1).to(x.device)
        x = x.repeat(1, 3, 1, 1)
        nested = NestedTensor(x, masks)
        # frames = [NestedTensor(frame.repeat(1, 1, 3, 1, 1), torch.ones(frame.shape[-2:]).to(frame.device))  for frame in x]
        features, pos_embeddings = self.spatial_backbone(nested)
        src, masks = features[-1].decompose()
        src = model.input_proj(src)  # condense down to 256 channels
        src = self.avgpool(src)  # condense further spatially
        masks = self.maxpool(masks.to(dtype=src.dtype))
        src = src.flatten(2).permute(0, 2, 1)

        cls_token = nn.Parameter(torch.zeros((1, 1, src.shape[-1]))).to(src.device)
        src = torch.cat([cls_token, src], dim=0)
        masks = torch.cat([torch.ones(1, 1, 1).to(masks.device), masks], dim=0)
        pos_embeddings = nn.Parameter(torch.zeros(src.shape)).to(
            src.device
        )  # generate our own positional embedding
        masks = masks.flatten(1).permute(1, 0)

        src = self.encoder(src, src_key_padding_mask=masks, pos=pos_embeddings)
        final = self.classifier(src[0, :])  # only operate on very first token

        return final


if __name__ == "__main__":
    lstm = Merge_LSTM(256, 6, 3).cuda()
    print(lstm)
    inputs = torch.rand(1, 499, 3, 240, 200).float().cuda()
    # import pdb; pdb.set_trace()
    out = lstm(inputs)
