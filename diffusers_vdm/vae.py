# video VAE with many components from lots of repos
# collected by lvmin


import torch
import xformers.ops
import torch.nn as nn

from einops import rearrange, repeat
from diffusers_vdm.basics import default, exists, zero_module, conv_nd, linear, normalization
from diffusers_vdm.unet import Upsample, Downsample
from huggingface_hub import PyTorchModelHubMixin


def chunked_attention(q, k, v, batch_chunk=0):
    # if batch_chunk > 0 and not torch.is_grad_enabled():
    #     batch_size = q.size(0)
    #     chunks = [slice(i, i + batch_chunk) for i in range(0, batch_size, batch_chunk)]
    #
    #     out_chunks = []
    #     for chunk in chunks:
    #         q_chunk = q[chunk]
    #         k_chunk = k[chunk]
    #         v_chunk = v[chunk]
    #
    #         out_chunk = torch.nn.functional.scaled_dot_product_attention(
    #             q_chunk, k_chunk, v_chunk, attn_mask=None
    #         )
    #         out_chunks.append(out_chunk)
    #
    #     out = torch.cat(out_chunks, dim=0)
    # else:
    #     out = torch.nn.functional.scaled_dot_product_attention(
    #         q, k, v, attn_mask=None
    #     )
    out = xformers.ops.memory_efficient_attention(q, k, v)
    return out


def nonlinearity(x):
    return x * torch.sigmoid(x)


def GroupNorm(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self, noise=None):
        if noise is None:
            noise = torch.randn(self.mean.shape)

        x = self.mean + self.std * noise.to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean


class EncoderDownSampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = GroupNorm(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = GroupNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(Attention(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = EncoderDownSampleBlock(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = Attention(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = GroupNorm(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, return_hidden_states=False):
        # timestep embedding
        temb = None

        # print(f'encoder-input={x.shape}')
        # downsampling
        hs = [self.conv_in(x)]

        ## if we return hidden states for decoder usage, we will store them in a list
        if return_hidden_states:
            hidden_states = []
        # print(f'encoder-conv in feat={hs[0].shape}')
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(f'encoder-down feat={h.shape}')
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if return_hidden_states:
                hidden_states.append(h)
            if i_level != self.num_resolutions - 1:
                # print(f'encoder-downsample (input)={hs[-1].shape}')
                hs.append(self.down[i_level].downsample(hs[-1]))
                # print(f'encoder-downsample (output)={hs[-1].shape}')
        if return_hidden_states:
            hidden_states.append(hs[0])
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        # print(f'encoder-mid1 feat={h.shape}')
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print(f'encoder-mid2 feat={h.shape}')

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # print(f'end feat={h.shape}')
        if return_hidden_states:
            return h, hidden_states
        else:
            return h


class ConvCombiner(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1, padding=0)

        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, context):
        ## x: b c h w, context: b c 2 h w
        b, c, l, h, w = context.shape
        bt, c, h, w = x.shape
        context = rearrange(context, "b c l h w -> (b l) c h w")
        context = self.conv(context)
        context = rearrange(context, "(b l) c h w -> b c l h w", l=l)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=bt // b)
        x[:, :, 0] = x[:, :, 0] + context[:, :, 0]
        x[:, :, -1] = x[:, :, -1] + context[:, :, -1]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class AttentionCombiner(nn.Module):
    def __init__(
            self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op = None

        self.norm = GroupNorm(query_dim)
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(
            self,
            x,
            context=None,
            mask=None,
    ):
        bt, c, h, w = x.shape
        h_ = self.norm(x)
        h_ = rearrange(h_, "b c h w -> b (h w) c")
        q = self.to_q(h_)

        b, c, l, h, w = context.shape
        context = rearrange(context, "b c l h w -> (b l) (h w) c")
        k = self.to_k(context)
        v = self.to_v(context)

        t = bt // b
        k = repeat(k, "(b l) d c -> (b t) (l d) c", l=l, t=t)
        v = repeat(v, "(b l) d c -> (b t) (l d) c", l=l, t=t)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = chunked_attention(
            q, k, v, batch_chunk=1
        )

        if exists(mask):
            raise NotImplementedError

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        out = self.to_out(out)
        out = rearrange(out, "bt (h w) c -> bt c h w", h=h, w=w, c=c)
        return x + out


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = GroupNorm(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )

        out = chunked_attention(
            q, k, v, batch_chunk=1
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class VideoDecoder(nn.Module):
    def __init__(
            self,
            *,
            ch,
            out_ch,
            ch_mult=(1, 2, 4, 8),
            num_res_blocks,
            attn_resolutions,
            dropout=0.0,
            resamp_with_conv=True,
            in_channels,
            resolution,
            z_channels,
            give_pre_end=False,
            tanh_out=False,
            use_linear_attn=False,
            attn_level=[2, 3],
            video_kernel_size=[3, 1, 1],
            alpha: float = 0.0,
            merge_strategy: str = "learned",
            **kwargs,
    ):
        super().__init__()
        self.video_kernel_size = video_kernel_size
        self.alpha = alpha
        self.merge_strategy = merge_strategy
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.attn_level = attn_level
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = VideoResBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            video_kernel_size=self.video_kernel_size,
            alpha=self.alpha,
            merge_strategy=self.merge_strategy,
        )
        self.mid.attn_1 = Attention(block_in)
        self.mid.block_2 = VideoResBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            video_kernel_size=self.video_kernel_size,
            alpha=self.alpha,
            merge_strategy=self.merge_strategy,
        )

        # upsampling
        self.up = nn.ModuleList()
        self.attn_refinement = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    VideoResBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        video_kernel_size=self.video_kernel_size,
                        alpha=self.alpha,
                        merge_strategy=self.merge_strategy,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(Attention(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

            if i_level in self.attn_level:
                self.attn_refinement.insert(0, AttentionCombiner(block_in))
            else:
                self.attn_refinement.insert(0, ConvCombiner(block_in))
        # end
        self.norm_out = GroupNorm(block_in)
        self.attn_refinement.append(ConvCombiner(block_in))
        self.conv_out = DecoderConv3D(
            block_in, out_ch, kernel_size=3, stride=1, padding=1, video_kernel_size=self.video_kernel_size
        )

    def forward(self, z, ref_context=None, **kwargs):
        ## ref_context: b c 2 h w, 2 means starting and ending frame
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if ref_context:
                h = self.attn_refinement[i_level](x=h, context=ref_context[i_level])
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        if ref_context:
            # print(h.shape, ref_context[i_level].shape) #torch.Size([8, 128, 256, 256]) torch.Size([1, 128, 2, 256, 256])
            h = self.attn_refinement[-1](x=h, context=ref_context[-1])
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class TimeStackBlock(torch.nn.Module):
    def __init__(
            self,
            channels: int,
            emb_channels: int,
            dropout: float,
            out_channels: int = None,
            use_conv: bool = False,
            use_scale_shift_norm: bool = False,
            dims: int = 2,
            use_checkpoint: bool = False,
            up: bool = False,
            down: bool = False,
            kernel_size: int = 3,
            exchange_temb_dims: bool = False,
            skip_t_emb: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            # print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class VideoResBlock(ResnetBlock):
    def __init__(
            self,
            out_channels,
            *args,
            dropout=0.0,
            video_kernel_size=3,
            alpha=0.0,
            merge_strategy="learned",
            **kwargs,
    ):
        super().__init__(out_channels=out_channels, dropout=dropout, *args, **kwargs)
        if video_kernel_size is None:
            video_kernel_size = [3, 1, 1]
        self.time_stack = TimeStackBlock(
            channels=out_channels,
            emb_channels=0,
            dropout=dropout,
            dims=3,
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=True,
            skip_t_emb=True,
        )

        self.merge_strategy = merge_strategy
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned":
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, bs):
        if self.merge_strategy == "fixed":
            return self.mix_factor
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        else:
            raise NotImplementedError()

    def forward(self, x, temb, skip_video=False, timesteps=None):
        assert isinstance(timesteps, int)

        b, c, h, w = x.shape

        x = super().forward(x, temb)

        if not skip_video:
            x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            x = self.time_stack(x, temb)

            alpha = self.get_alpha(bs=b // timesteps)
            x = alpha * x + (1.0 - alpha) * x_mix

            x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class DecoderConv3D(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, video_kernel_size=3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        if isinstance(video_kernel_size, list):
            padding = [int(k // 2) for k in video_kernel_size]
        else:
            padding = int(video_kernel_size // 2)

        self.time_mix_conv = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=video_kernel_size,
            padding=padding,
        )

    def forward(self, input, timesteps, skip_video=False):
        x = super().forward(input)
        if skip_video:
            return x
        x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
        x = self.time_mix_conv(x)
        return rearrange(x, "b c t h w -> (b t) c h w")


class VideoAutoencoderKL(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 double_z=True,
                 z_channels=4,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[],
                 num_res_blocks=2,
                 attn_resolutions=[],
                 dropout=0.0,
                 ):
        super().__init__()
        self.encoder = Encoder(double_z=double_z, z_channels=z_channels, resolution=resolution, in_channels=in_channels,
                               out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout)
        self.decoder = VideoDecoder(double_z=double_z, z_channels=z_channels, resolution=resolution,
                                    in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult,
                                    num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.scale_factor = 0.18215

    def encode(self, x, return_hidden_states=False, **kwargs):
        if return_hidden_states:
            h, hidden = self.encoder(x, return_hidden_states)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior, hidden
        else:
            h = self.encoder(x)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior, None

    def decode(self, z, **kwargs):
        if len(kwargs) == 0:
            z = self.post_quant_conv(z)
        dec = self.decoder(z, **kwargs)
        return dec

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
