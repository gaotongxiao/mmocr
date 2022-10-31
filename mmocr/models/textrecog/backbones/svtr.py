# Copyright (c) OpenMMLab. All rights reserved.
# Modified from <https://arxiv.org/abs/2205.00159>
# Adapted from <https://github.com/PaddlePaddle/PaddleOCR>
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule


class OverlapPatchEmbed(BaseModule):
    """Image to the progressive overlapping Patch Embedding.

    Args:
        img_size (int or tuple): The size of input, which will be used to
            calculate the out size. Defaults to [32, 100].
        in_channels (int): Number of input channels. Defaults to 3.
        embed_dims (int): The dimensions of embedding. Defaults to 768.
        num_layers (int, optional): Number of Conv_BN_Layer. Defaults to 2 and
            limit to [2, 3].
    """

    def __init__(self,
                 img_size: Union[int, Tuple[int, int]] = [32, 100],
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 num_layers: int = 2,
                 init_cfg: Optional[Dict] = None):

        super().__init__(init_cfg=init_cfg)

        # num_patches = (img_size[1] // (2 ** num_layers)) * \
        #               (img_size[0] // (2 ** num_layers))

        assert num_layers in [2, 3], \
            'The number of layers must belong to [2, 3]'
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.norm = None
        self.net = nn.Sequential()
        for num in range(num_layers, 0, -1):
            if (num == num_layers):
                _input = in_channels
            _output = embed_dims // (2**(num - 1))
            self.net.add_module(
                f'ConvModule{str(num_layers - num)}',
                ConvModule(
                    in_channels=_input,
                    out_channels=_output,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='GELU')))
            _input = _output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): A Tensor of shape :math:`(N, C, H, W)`.

        Returns:
            Tensor: A tensor of shape math:`(N, HW//16, C)`.
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model \
                ({self.img_size[0]}*{self.img_size[1]})."

        x = self.net(x).flatten(2).permute(0, 2, 1)
        return x


class ConvMixer(BaseModule):
    """The conv Mixer.

    Args:
        dim (int): Number of character components.
        num_heads (int, optional): Number of heads. Defaults to 8.
        HW (Tuple[int, int], optional): Number of H x W. Defaults to [8, 25].
        local_k (Tuple[int, int], optional): Window size. Defaults to [3, 3].
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 HW: Tuple[int, int] = [8, 25],
                 local_k: Tuple[int, int] = [3, 3],
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)
        self.HW = HW
        self.embed_dims = embed_dims
        self.local_mixer = ConvModule(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=local_k,
            stride=1,
            padding=[local_k[0] // 2, local_k[1] // 2],
            groups=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, HW, C)`.

        Returns:
            torch.Tensor: Tensor: A tensor of shape math:`(N, HW, C)`.
        """
        h, w = self.HW[0], self.HW[1]
        x = x.permute(0, 2, 1).reshape([-1, self.embed_dims, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class AttnMixer(BaseModule):
    """One of mixer of {'Global', 'Local'}. Defaults to Global Mixer.

    Args:
        embed_dims (int): Number of character components.
        num_heads (int, optional): Number of heads. Defaults to 8.
        mixer (str, optional): The mixer type. Defaults to 'Global'.
        HW (Tuple[int, int], optional): Number of H x W. Defaults to [8, 25].
        local_k (Tuple[int, int], optional): Window size. Defaults to [7, 11].
        qkv_bias (bool, optional): Whether a additive bias is required.
            Defaults to False.
        qk_scale (float, optional): A scaling factor. Defaults to None.
        attn_drop (float, optional): A Dropout layer. Defaults to 0.0.
        proj_drop (float, optional): A Dropout layer. Defaults to 0.0.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 mixer: str = 'Global',
                 HW: Tuple[int, int] = [8, 25],
                 local_k: Tuple[int, int] = [7, 11],
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg)
        assert mixer in {'Global', 'Local'}, \
            "The type of mixer must belong to {'Global', 'Local'}"
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H, W = HW[0], HW[1]
            self.N = H * W
            self.C = embed_dims
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones([H * W, H + hk - 1, W + wk - 1], dtype='float32')
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * w + w, h:h + hk, w:w + wk] = 0.
            local_mask = mask[:, hk // 2:H + hk // 2,
                              wk // 2:W + wk // 2].flatten(1)
            mask_inf = torch.full([H * W, H * W], -10000, dtype='float32')
            mask = torch.where(local_mask < 1, local_mask, mask_inf)
            self.mask = mask.unsqueeze([0, 1])
        self.mixer = mixer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H, W, C)`.
        """
        if self.HW is not None:
            N, C = self.N, self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x).reshape(
            (-1, N, 3, self.num_heads, C // self.num_heads)).permute(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q.matmul(k.permute(0, 1, 3, 2)))
        if self.mixer == 'Local':
            attn += self.mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn.matmul(v).permute(0, 2, 1, 3).reshape(-1, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(BaseModule):
    """The MLP block.

    Args:
        in_features (int): The input features.
        hidden_features (int, optional): The hidden features.
            Defaults to None.
        out_features (int, optional): The output features.
            Defaults to None.
        drop (float, optional): cfg of dropout function. Defaults to 0.0.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int = None,
                 out_features: int = None,
                 drop: float = 0.,
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H, W, C)`.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixingBlock(BaseModule):
    """The Mixing block.

    Args:
        embed_dims (int): Number of character components.
        num_heads (int): Number of heads
        mixer (str, optional): The mixer type. Defaults to 'Global'.
        local_mixer (Tuple[int ,int], optional): Local window size.
            Defaults to [7, 11].
        HW (Tuple[int, int], optional): The size of [H, W].
            Defaults to [8, 25].
        mlp_ratio (float, optional): The ratio of hidden features to input.
            Defaults to 4.0.
        qkv_bias (bool, optional): Whether a additive bias is required.
            Defaults to False.
        qk_scale (float, optional): A scaling factor. Defaults to None.
        drop (float, optional): _description_. Defaults to 0..
        attn_drop (float, optional): A Dropout layer. Defaults to 0.0.
        drop_path (_type_, optional): The probability of drop path.
            Defaults to 0.0.
        pernorm (bool, optional): Mixing layer before norm. Defaults to True.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mixer: str = 'Global',
                 local_mixer: Tuple[int, int] = [7, 11],
                 HW: Tuple[int, int] = [8, 25],
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path=0.,
                 prenorm: bool = True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        if mixer in {'Global', 'Local'}:
            self.mixer = AttnMixer(
                embed_dims,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                embed_dims, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=embed_dims, hidden_features=mlp_hidden_dim, drop=drop)
        self.prenorm = prenorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H*W, C)`.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H*W, C)`.
        """
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DownSample(BaseModule):
    """The last layer of any stage, one of the {'Merging', 'Combing'}.

    Args:
        in_channels (int): The channels of input.
        out_channels (int): The channels of output.
        types (str, optional): Which operation of ['Merging', 'Combing'].
            Defaults to 'Merging'.
        stride (Union[int, Tuple[int, int]], optional): Stride of the Conv.
            Defaults to [2, 1].
        act (bool, optional): activation function. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 types: str = 'Merging',
                 stride: Union[int, Tuple[int, int]] = [2, 1],
                 act: bool = None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.types = types
        if types == 'Combing':
            self.avgpool = nn.AvgPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        elif types == 'Merging':
            self.conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1)
        self.norm = nn.LayerNorm(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H, W, C)`.

        Returns:
            torch.Tensor: A Tensor of shape :math:`(N, H/2, W, 2C)`.
        """
        if self.types == 'Combing':
            x = (self.avgpool(x) + self.maxpool(x)) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))

        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out
