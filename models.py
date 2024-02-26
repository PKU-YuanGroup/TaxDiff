import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from transformers import AutoModel, AutoTokenizer


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

# class TaxStr_Embedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """
#     def __init__(self, num_classes, hidden_size):
#         super().__init__()
#         self.bge_model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5').embeddings
#         self.embedding_table = nn.Embedding(num_classes, hidden_size)
#         self.num_classes = num_classes
#         # self.emb_linear2 = nn.Linear(1024, hidden_size, bias=True)
#         self.emb_2_linear = nn.Sequential(
#             nn.Linear(1024, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size, bias=True)
#         )
#     def forward(self, tax_str):
#         # print(tax_str)
#         with torch.no_grad():
#             model_output = self.bge_model(tax_str.long())
#         sentence_embeddings = model_output[:,0,:]
#         sentence_embeddings = nn.functional.normalize(sentence_embeddings, p=2, dim=1)
#         out_emb = self.emb_2_linear(sentence_embeddings)
#         return out_emb

#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class MyDiTBlock_conta(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio, slice_size,**block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # local attn Processing
        self.norm3 = nn.LayerNorm(2*hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.local_nat = Attention(hidden_size, num_heads=1, **block_kwargs)
        self.slice_size = slice_size
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=2*hidden_size, hidden_features=mlp_hidden_dim, out_features=hidden_size,act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * 2 * hidden_size, bias=True)
        )
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_nat, scale_nat, gate_nat = self.adaLN_modulation(c).chunk(9, dim=1)
        
        shift_mlp, scale_mlp, _ = self.adaLN_modulation2(c).chunk(3, dim=1)
        b, l, d = x.shape
        x0 = self.norm4(x)
        x1 = rearrange(x0, 'b (i j) d -> b i j d', i=self.slice_size, j=l//self.slice_size)
        x1 = rearrange(x1, 'b i j d -> (b i) j d')
        x1 = self.local_nat(x1)
        x1 = rearrange(x1, '(b i) j d -> b i j d', b=b,i=self.slice_size)
        x1 = rearrange(x1, 'b i j d -> b (i j) d')
        x1 = x0 + gate_nat.unsqueeze(1) * (modulate(self.norm2(x1), shift_nat, scale_nat))
        x2 = x0 + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x0), shift_msa, scale_msa))
        x_concat = torch.cat((x1, x2), dim=2)    
        
        x =  x0 + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x_concat), shift_mlp, scale_mlp))
        # x = self.mlp2(x)
        return x

class MyFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 2*hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    DiT_XL_2(**kwargs): DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,               
        in_channels=4,
        hidden_size=1152,   
        depth=28, 
        num_heads=16,               
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        slice_size=1,
    ):
        super().__init__()
        self.embeding = nn.Embedding(21,hidden_size)
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        # self.tax_str_embedder  = TaxStr_Embedder(num_classes, hidden_size)
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.x_embedder.num_patches = 256
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            MyDiTBlock_conta(hidden_size, num_heads, mlp_ratio=mlp_ratio, slice_size=slice_size) for _ in range(depth)
        ])
        self.final_layer = MyFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # x = self.embeding(x)
        b, ch, h, w = x.shape #
        x = x.reshape(b, ch, h*w).transpose(1,2) 
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c) # B, HW, 2C               # (N, T, patch_size ** 2 * out_channels)
        x = x.transpose(1,2).reshape(b, 2*ch, h, w)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.      
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_pro_12_h6_L4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, slice_size=4, num_heads=6, **kwargs)

def DiT_pro_12_h6_L8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, slice_size=8, num_heads=6, **kwargs)

def DiT_pro_12_h6_L16(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, slice_size=16, num_heads=6, **kwargs)

def DiT_pro_12_h6_L32(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, slice_size=32, num_heads=6, **kwargs)

def DiT_pro_12_h6_L64(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, slice_size=64, num_heads=6, **kwargs)
    
DiT_models = {
    'DiT-pro-12-h6-L4':  DiT_pro_12_h6_L4,
    'DiT-pro-12-h6-L8':  DiT_pro_12_h6_L8,
    'DiT-pro-12-h6-L16':  DiT_pro_12_h6_L16,
    'DiT-pro-12-h6-L32':  DiT_pro_12_h6_L32,
    'DiT-pro-12-h6-L64':  DiT_pro_12_h6_L64, 
}