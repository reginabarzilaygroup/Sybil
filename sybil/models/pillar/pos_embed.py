import einops
import numpy as np


def gen_3d_sincos_pos_embed(d, img_seqlen, frame_seqlen, num_blank_tokens=0):
    """
    Args:
        img_size: int of the grid height and width
        frame_size: int of the temporal size
    Returns:
        pos_embed: (w/ or w/o cls_token)
    """
    assert d % 4 == 0
    embed_dim_s = d // 4 * 3
    embed_dim_t = d // 4

    # position encoding along spatial and temporal dimensions
    loc_h = loc_w = np.arange(img_seqlen, dtype=np.float32)
    loc_s = np.stack(np.meshgrid(loc_w, loc_h), axis=0)  # here w goes first

    loc_s = loc_s.reshape([2, 1, img_seqlen, img_seqlen])
    pos_embed_s = gen_2d_sincos_pos_embed_from_grid(embed_dim_s, loc_s)

    loc_t = np.arange(frame_seqlen, dtype=np.float32)
    pos_embed_t = gen_1d_sincos_pos_embed_from_grid(embed_dim_t, loc_t)

    # concate: [T, H, W] order

    pos_embed_t = einops.repeat(pos_embed_t, "t d -> t s d", s=img_seqlen**2)
    pos_embed_s = einops.repeat(pos_embed_s, "s d -> t s d", t=frame_seqlen)

    pos_embed = np.concatenate([pos_embed_t, pos_embed_s], axis=-1).reshape([-1, d])

    # add special tokens
    if num_blank_tokens > 0:
        pos_embed = np.concatenate([np.zeros([num_blank_tokens, d]), pos_embed], axis=0)
    return pos_embed


def gen_2d_sincos_pos_embed(embed_dim, grid_size, num_blank_tokens=0):
    """
    grid_size: int of the grid height and width

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = gen_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # add special tokens
    if num_blank_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([num_blank_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def gen_2d_sincos_pos_embed_from_grid(d, grid):
    """
    generate 2d sinusoidal position embedding from grid tensor.

    Args:
        d (int): embedding dimension
        grid (Tensor): grid tensor
    Returns:
        pos_emb (Tensor): position embedding, shape (H*W, D)
    """
    assert d % 2 == 0

    # use half of dimensions to encode grid_h
    pos_emb_h = gen_1d_sincos_pos_embed_from_grid(d // 2, grid[0])
    pos_emb_w = gen_1d_sincos_pos_embed_from_grid(d // 2, grid[1])

    pos_emb = np.concatenate([pos_emb_h, pos_emb_w], axis=1)  # (H*W, d)
    return pos_emb


def gen_1d_sincos_pos_embed_from_grid(d, loc):
    """
    Generate 1d sinusoidal position embedding from location tensor.

    Args:
        d (int): embedding dimension
        loc (Tensor): location tensor, shape (N, d)
    """
    MAX_TIMESCALE = 1.0e4

    assert d % 2 == 0, "embed_dim must be even for 1d sin/cos embedding"

    inv_time = np.arange(d // 2, dtype=np.float32)
    inv_time = 1.0 / MAX_TIMESCALE ** (2 * inv_time / d)

    loc = loc.reshape(-1)
    scaled_time = np.einsum("l,d->ld", loc, inv_time)

    pos_emb = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return pos_emb
