from typing import Tuple
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import mul
from functools import reduce
import math

from .CLIP_vision_encoder_utils import QuickGELU, LayerNorm, TransformerEncoderLayer, ImagePatchEmbed2D


class CLIPVisionEncoder(nn.Module):

    def __init__(
            self,
            # data shape
            device,
            input_size: Tuple[int, int] = (224, 224),
            num_frames: int = 20,
            # model def
            feature_dim: int = 768,
            patch_size: Tuple[int, int] = (16, 16),
            num_heads: int = 12,
            num_layers: int = 12,
            mlp_factor: float = 4.0,
            act: nn.Module = QuickGELU,
            embed_dim: int = 512,

    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.patch_embed = ImagePatchEmbed2D(img_size=input_size[0], patch_size=patch_size[0], in_chans=3,
                                             embed_dim=feature_dim)
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))


        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, feature_dim]))
        self.time_embed = nn.Parameter(torch.zeros([num_frames, feature_dim]))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                device,
                in_feature_dim=feature_dim, qkv_dim=feature_dim, num_heads=num_heads,
                mlp_factor=mlp_factor, act=act
            ) for _ in range(num_layers)
        ])

        self.ln_pre = LayerNorm(feature_dim)
        self.ln_post = LayerNorm(feature_dim)
        self.ln_at = LayerNorm(feature_dim)
        scale = feature_dim ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(feature_dim, embed_dim))


        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)



    def _initialize_global_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.global_prompts.data, -val, val)

    def temporal_encoding(self, x, T, B):
        ## Time Embeddings
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)

        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(0):
            time_embed = self.time_embed.unsqueeze(0).transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2).squeeze(0)
            x = x + new_time_embed
        else:
            x = x + self.time_embed

        x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T)
        return x

    def forward(self, x: torch.Tensor, face):

        B, T, C, H, W = x.size()
        x = x.flatten(0, 1)

        x = self.patch_embed(x)

        x = torch.cat([self.cls_token.view(1, 1, -1).repeat(x.size(0), 1, 1), x], dim=1)

        x = x + self.pos_embed

        x = torch.cat((x, face), dim=1)

        x = self.temporal_encoding(x, T, B)
        x = self.ln_pre(x)


        for blk in self.blocks:
            x = blk(x)
        scene_feature = x[:,1:-2,:]

        cls_x = self.ln_post(x[:, 0, :])

        at_1 = self.ln_at(x[:, -2, :])

        at_2 = self.ln_at(x[:, -1, :])


        return cls_x, at_1, at_2,scene_feature