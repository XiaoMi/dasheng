import torch
import torch.nn as nn
from einops import rearrange
import torch
from torch.cuda.amp import autocast
from functools import partial
from typing import Optional, Tuple, Union
import torchaudio.transforms as audio_transforms
from einops import rearrange
from einops.layers.torch import Rearrange
from itertools import repeat
import collections


def _ntuple(n):

    def parse(x) -> Tuple:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class MAELoss(torch.nn.Module):

    def __init__(self, norm_pix_loss: bool = True):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    @autocast(enabled=False)
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if self.norm_pix_loss is True:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        elif self.norm_pix_loss == 'global':
            mean = target.mean()
            var = target.var()
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


class AudioPatchEmbed(nn.Module):

    def __init__(self,
                 input_size: Union[int, Tuple[int, int]] = (64, 100),
                 patch_size: Tuple[int, int] = (64, 4),
                 patch_stride: Tuple[int, int] = (64, 4),
                 in_chans=1,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.input_size: Tuple[int, int] = to_2tuple(input_size)
        self.patch_size: Tuple[int, int] = to_2tuple(patch_size)
        self.patch_stride: Tuple[int, int] = to_2tuple(patch_stride)
        self.grid_size = (self.input_size[0] // self.patch_stride[0],
                          self.input_size[1] // self.patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, 'b c f t -> b (f t) c')
        x = self.norm(x)
        return x


class LayerScale(nn.Module):

    def __init__(self, dim: int, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        act_layer: torch.Callable = nn.GELU,
        norm_layer: torch.Callable = nn.LayerNorm,
        attention_type='Attention',
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        attn_type = globals()[attention_type]
        self.attn = attn_type(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class AudioTransformerMAE_Encoder(nn.Module):

    def __init__(self,
                 patch_size: Tuple[int, int] = (64, 4),
                 patch_stride: Tuple[int, int] = (64, 4),
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=None,
                 act_layer=None,
                 init_values=None,
                 target_length=1008,
                 pooling='mean',
                 time_patch_out: Optional[float] = None,
                 freq_patch_out: Optional[float] = None,
                 block_type='Block',
                 attention_type='Attention',
                 eval_avg='cat',
                 n_fft: int = 512,
                 n_mels: int = 64,
                 hop_size: int = 160,
                 win_size: int = 512,
                 f_min: int = 0,
                 f_max: int = 8000,
                 center: bool = True,
                 **kwargs):
        super().__init__()
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels = n_mels
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out

        self.front_end = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=f_min,
                                            sample_rate=16000,
                                            win_length=win_size,
                                            center=center,
                                            n_fft=n_fft,
                                            f_max=f_max,
                                            hop_length=hop_size,
                                            n_mels=self.n_mels),
            audio_transforms.AmplitudeToDB(top_db=kwargs.get('top_db', 120)))

        self.init_bn = nn.Sequential(
            Rearrange('b c f t -> b f c t'),
            nn.BatchNorm2d(self.n_mels, momentum=0.01),
            Rearrange('b f c t -> b c f t'))

        self.target_length = target_length
        self.patch_embed = AudioPatchEmbed(input_size=(self.n_mels,
                                                       target_length),
                                           embed_dim=self.embed_dim,
                                           patch_size=self.patch_size,
                                           flatten=False,
                                           patch_stride=self.patch_stride)
        self.num_patches = self.patch_embed.num_patches

        if pooling == 'token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.token_pos_embed = nn.Parameter(
                torch.randn(1, embed_dim) * .02)

        self.time_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, self.patch_embed.grid_size[1]) * .02)
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * .02)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.pos_drop = nn.Dropout(p=drop_rate)
        block_function = globals()[block_type]
        self.blocks = nn.Sequential(*[
            block_function(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                act_layer=act_layer,
                attention_type=attention_type,
            ) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.apply(self.init_weights)
        if hasattr(self, 'cls_token') and self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        group_masking = kwargs.get('group_masking', False)
        if isinstance(group_masking, bool):
            if group_masking is True:
                self.masking_func = self.random_masking_group
            else:
                self.masking_func = self.random_masking
        elif isinstance(group_masking, int):
            self.masking_func = partial(self.random_masking_group,
                                        group_factor=group_masking)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'time_pos_embed', 'cls_token', 'freq_pos_embed', 'token_pos_embed'
        }

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def random_masking_group(self, x, mask_ratio, group_factor: int = 2):
        """                                                                                                                                                                                       
        Perform per-sample random masking by per-sample shuffling.                                                                                                                                
        Per-sample shuffling is done by argsort random noise.                                                                                                                                     
        x: [N, L, D], sequence                                                                                                                                                                    
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L // group_factor,
                           device=x.device)  # noise in [0, 1]
        # indices = torch.arange(L).view(1, 5, 4).repeat(N, 1, 1)
        indices = torch.arange(L, device=x.device).view(-1, group_factor)

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_shuffle = indices[ids_shuffle].flatten(-2)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x,
                                dim=1,
                                index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x,
                                dim=1,
                                index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_features(self, x, mask_ratio):
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]  # Just for sin pos embed
        x = rearrange(x, 'b c f t -> b (f t) c')
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        x, mask, ids_restore = self.masking_func(x, mask_ratio)
        if self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed[:, :]
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if 'time_pos_embed' in state_dict and self.time_pos_embed.shape != state_dict[
                'time_pos_embed'].shape:
            print(
                "Positional Embedding shape not the same with model, resizing!"
            )
            self.change_pos_embedding(state_dict)
        super().load_state_dict(state_dict, strict=strict, **kwargs)

    def change_pos_embedding(self, state_dict):
        target_time_pos_embed_length = self.time_pos_embed.shape[-1]
        target_freq_pos_embed_length = self.freq_pos_embed.shape[-2]

        pretrained_time_pos_embed = state_dict['time_pos_embed']
        pretrained_freq_pos_embed = state_dict['freq_pos_embed']

        if target_freq_pos_embed_length <= pretrained_time_pos_embed.shape[-1]:
            state_dict['time_pos_embed'] = pretrained_time_pos_embed[
                ..., :target_time_pos_embed_length]
        else:
            state_dict['time_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_time_pos_embed,
                size=(1, target_time_pos_embed_length),
                align_corners=False,
                mode='bilinear')
        if target_freq_pos_embed_length <= pretrained_freq_pos_embed.shape[-2]:
            state_dict[
                'freq_pos_embed'] = pretrained_freq_pos_embed[:, :, :
                                                              target_freq_pos_embed_length, :]
        else:
            state_dict['freq_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_freq_pos_embed,
                size=(target_freq_pos_embed_length, 1),
                align_corners=False,
                mode='bilinear')

    def forward_to_spec(self, x):
        # Do not use fp16 for feature extraction, that is likely to get nan
        with autocast(enabled=False):
            X = self.front_end(x)
            X = rearrange(X, 'b f t -> b 1 f t')
            X = self.init_bn(X)
        return X

    def forward(self, x, mask_ratio: float = 0.75):
        x = self.forward_to_spec(x)
        x, mask, restore_idxs = self.forward_features(x, mask_ratio=mask_ratio)
        return x, mask, restore_idxs


class AudioTransformerMAE_Decoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 outputdim: int,
                 patch_size: int = 16,
                 patch_stride: int = 16,
                 embed_dim: int = 768,
                 num_patches: int = 100,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 norm_layer: Optional[torch.nn.Module] = None,
                 act_layer: Optional[torch.nn.Module] = None,
                 cls_token: bool = False,
                 attention_type='Attention',
                 init_values=None,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * .02)
        _norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        _act_layer = act_layer or nn.GELU
        self.use_cls = cls_token
        num_patches_total = num_patches + 1 if not cls_token else num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_total, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=_norm_layer,
                act_layer=_act_layer,
                attention_type=attention_type,
            ) for i in range(depth)
        ])
        self.norm = _norm_layer(embed_dim)
        self.outputlayer = nn.Linear(self.embed_dim, outputdim)
        self.apply(self.init_weights)
        torch.nn.init.normal_(self.mask_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'time_pos_embed', 'cls_token', 'freq_pos_embed', 'token_pos_embed'
        }

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x, ids_restore):
        x = self.input_proj(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        if self.use_cls:
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        else:
            x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
        x_ = torch.gather(x_,
                          dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(
                              1, 1, x.shape[2]))  # unshuffle
        if self.use_cls:
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x = x_
        t = x.shape[1]

        x = x + self.pos_embed[:, :t, :]
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x, restore_idxs):
        x = self.forward_features(x, restore_idxs)
        x = self.outputlayer(x)
        return x


class AudioTransformerMAE(nn.Module):

    def __init__(self,
                 encoder: AudioTransformerMAE_Encoder,
                 decoder: AudioTransformerMAE_Decoder,
                 loss_fn: Optional[torch.nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.unfold = nn.Unfold(
            kernel_size=self.encoder.patch_embed.patch_size,
            stride=self.encoder.patch_embed.patch_size)
        self.loss_fn = MAELoss() if loss_fn is None else loss_fn

    def forward(self,
                x: torch.Tensor,
                mask_ratio: float = 0.75,
                return_loss: bool = False):
        latent, mask, restore_ids = self.encoder(x, mask_ratio=mask_ratio)
        pred = self.decoder(latent, restore_ids)
        with autocast(enabled=False):
            targets = self.encoder.front_end(x)
        targets = self.patchify(targets)
        if return_loss:
            return self.loss_fn(pred, targets, mask)
        return pred, targets, mask

    def patchify(self, x):
        return self.unfold(x.unsqueeze(1)).transpose(-2, -1)


def dasheng_base(**kwargs):
    encoder_kwargs = dict(embed_dim=768,
                          depth=12,
                          num_heads=12,
                          target_length=1008,
                          patch_size=[64, 4],
                          patch_stride=[64, 4])
    encoder_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(encoder_kwargs))
    encoder_kwargs = {**encoder_kwargs, **kwargs}
    encoder = AudioTransformerMAE_Encoder(**encoder_kwargs)

    decoder_kwargs = dict(embed_dim=512,
                          depth=8,
                          num_heads=16,
                          input_dim=encoder_kwargs['embed_dim'],
                          outputdim=encoder.patch_embed.patch_size[0] *
                          encoder.patch_embed.patch_size[1],
                          num_patches=encoder.patch_embed.num_patches)
    decoder = AudioTransformerMAE_Decoder(**decoder_kwargs)
    return AudioTransformerMAE(encoder, decoder)


def dasheng_06B(**kwargs):
    encoder_kwargs = dict(
        patch_size=[64, 4],
        patch_stride=[64, 4],
        embed_dim=1536,
        depth=24,
        num_heads=24,
        mlp_ratio=4,
    )
    encoder_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(encoder_kwargs))
    encoder_kwargs = {**encoder_kwargs, **kwargs}
    encoder = AudioTransformerMAE_Encoder(**encoder_kwargs)

    decoder_kwargs = dict(embed_dim=512,
                          depth=8,
                          num_heads=16,
                          input_dim=encoder_kwargs['embed_dim'],
                          outputdim=encoder.patch_embed.patch_size[0] *
                          encoder.patch_embed.patch_size[1],
                          num_patches=encoder.patch_embed.num_patches)
    decoder = AudioTransformerMAE_Decoder(**decoder_kwargs)
    return AudioTransformerMAE(encoder, decoder)


def dasheng_12B(**kwargs):
    encoder_kwargs = dict(
        patch_size=[64, 4],
        patch_stride=[64, 4],
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
    )
    encoder_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(encoder_kwargs))
    encoder_kwargs = {**encoder_kwargs, **kwargs}
    encoder = AudioTransformerMAE_Encoder(**encoder_kwargs)

    decoder_kwargs = dict(embed_dim=768,
                          depth=8,
                          num_heads=24,
                          input_dim=encoder_kwargs['embed_dim'],
                          outputdim=encoder.patch_embed.patch_size[0] *
                          encoder.patch_embed.patch_size[1],
                          num_patches=encoder.patch_embed.num_patches)
    decoder = AudioTransformerMAE_Decoder(**decoder_kwargs)
    return AudioTransformerMAE(encoder, decoder)
