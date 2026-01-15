import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TypedDict, List, Dict, Callable, Tuple
from typing_extensions import Unpack
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import math


class AutoencoderKwargs(TypedDict):
    batch_size: int
    block_num_local_encoder: int
    block_num_local_decoder: int
    block_num_latent_encoder: int
    block_num_latent_decoder: int
    downsize: int
    device: str
    embedding_dim: int
    grad_clipping: float
    r: int
    lr: float
    max_len: int
    mlp_ratio: float
    name: str
    n_layers: int
    num_heads: int
    training_iterations: int
    use_wandb: bool
    vocab_size: int
    token_merging: bool
    warmup_iterations: int
    window_size: int
    down_weight_factor: float
    token_merging_latent_encoder: bool


def adjacent_cosine_sim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: [B, L, D]
    returns sim: [B, L-1] where sim[:, i] = cosine(x_i, x_{i+1})
    """
    x0 = x[:, :-1, :]
    x1 = x[:, 1:, :]
    x0 = x0 / (x0.norm(dim=-1, keepdim=True) + eps)
    x1 = x1 / (x1.norm(dim=-1, keepdim=True) + eps)
    return (x0 * x1).sum(dim=-1)


def merge_key_padding_mask(key_padding_mask: torch.Tensor, merge_fn):
    """
    key_padding_mask: [B, L] bool, True where PAD
    merge_fn: returned by bipartite_soft_matching(...)
    returns: [B, L-r] bool
    """
    # 1 for valid, 0 for pad
    valid = (~key_padding_mask).to(torch.float32).unsqueeze(-1)  # [B, L, 1]

    # apply same merge operator
    valid_merged = merge_fn(valid)  # [B, L-r, 1]

    # back to bool PAD mask: PAD if "mostly invalid"
    key_padding_mask_merged = valid_merged.squeeze(-1) < 0.5  # [B, L-r] bool
    return key_padding_mask_merged


def _build_local_attn_mask(
    seq_len: int, window: int, device: torch.device
) -> torch.Tensor:
    """
    Returns an attention mask of shape [seq_len, seq_len]:
      - True entries mean "disallow attention" (masked out).
      - We allow attention only if |i - j| <= window.
    """
    idx = torch.arange(seq_len, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    disallow = dist > window  # True = mask out
    return disallow


class TransformerBlockLocalEncode(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        window_size: int = 16,
        token_merging: bool = False,
        r: int = 2,
        protected: int = 1,
        use_prop_attn: bool = True,
        build_local_attn_mask: bool = True,
    ):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        )  # for multi-head attention to work correctly

        self.ln1 = nn.LayerNorm(embedding_dim)

        self.ln2 = nn.LayerNorm(embedding_dim)
        hidden = int(embedding_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),  # commonly used in transformer architecture and prevent dead neurons
            nn.Linear(hidden, embedding_dim),
        )
        self.window_size = window_size
        self.build_local_attn_mask = build_local_attn_mask
        self.token_merging = token_merging
        self.r = r
        self.protected = protected
        self.use_prop_attn = use_prop_attn
        if self.token_merging:
            self.attn = ToMeAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        else:
            self.attn = FlashSelfAttention(embedding_dim, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        token_sizes=None,
        return_info: bool = False,
    ):
        """
        x: [B, L, D]
        key_padding_mask: [B, L] with True for PAD positions (these are ignored in attention)
        token_sizes: [B, L', 1]

        Returns:
            x: [B, L' , C]
            token_sizes: [B, L', 1]
            key_padding_mask: [B, L'] (if provided)
            info: (unm_idx, src_idx, dst_idx, T_old) or None
        """
        B, L, D = x.shape
        if token_sizes is None:
            token_sizes = x.new_ones(B, L, L)

        if self.build_local_attn_mask:
            attn_mask = _build_local_attn_mask(
                L, self.window_size, x.device
            )  # [L, L], bool
        else:
            attn_mask = None

        # Pre-norm attention
        h = self.ln1(x)

        info = None

        if self.token_merging:
            attn_out, k_merge = self.attn(
                h,
                token_sizes=token_sizes if self.use_prop_attn else None,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            print('attn_out shape: ', attn_out.shape)
            print('k_merge shape: ', k_merge.shape)
            merge_fn, info = bipartite_soft_matching(
                k_merge, r=self.r, protected=self.protected
            )
            print('info: ', info)
            x, token_sizes = merge_with_sizes(x + attn_out, token_sizes, merge_fn)

            key_padding_mask = merge_key_padding_mask(key_padding_mask, merge_fn)
        else:
            attn_out = self.attn(x, key_padding_mask=key_padding_mask)

            x = x + attn_out
        h = self.ln2(x)
        x = x + self.mlp(h)

        if return_info:
            return x, token_sizes, key_padding_mask, info
        else:
            return x, token_sizes, key_padding_mask, None


class localEncoder(nn.Module):
    """
    Local Transformer encoder (no token merging).
    Input: token ids [B, L]
    Output: contextualized embeddings [B, L, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float,
        window_size: int,
        vocab_size: int,
        n_layers: int,
        max_len: int,
        r: int,
        pad_token_id: int = 0,
        token_merging: bool = False,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        self.token_merging = token_merging
        self.r = r

        self.blocks = nn.ModuleList(
            [
                TransformerBlockLocalEncode(
                    embedding_dim, num_heads, mlp_ratio, window_size, token_merging, r
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(embedding_dim)
        self.pad_token_id = pad_token_id

    def forward(self, x: torch.Tensor, token_sizes=None):
        """
        x: [B, L] (int64)
        """
        B, L = x.shape
        L0 = L

        # key_padding_mask: True where padding
        key_padding_mask = x == self.pad_token_id  # [B, L], bool

        if token_sizes is None:
            token_sizes = x.new_ones(B, L, 1)  # init: each token has a size of 1

        # mapping from original token index -> current token index
        owner_idx = torch.arange(L0, device=x.device)[None, :].expand(B, L0)  # [B, L0]

        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)  # [B, L]
        x = self.tok_emb(x) + self.pos_emb(pos)

        num_tokens_merged = 0
        merge_maps = []  # store (old_to_new, L_old, L_new) per merge step
        # print("x len before block:", x.shape[1])
        for i, block in enumerate(self.blocks):
            x, token_sizes, key_padding_mask, info = block(
                x,
                key_padding_mask=key_padding_mask,
                token_sizes=token_sizes,
                return_info=True,
            )

            print('local encoder forward block ', i)
            print('x shape after block: ',  x.shape)
            print('token_sizes shape after block: ',  token_sizes.shape)

            if info is not None:
                unm_idx, src_idx, dst_idx, L_old = info
                print('umn_idx shape: ', unm_idx.shape)
                print('src_idx shape: ', src_idx.shape)
                print('dst_idx shape: ', dst_idx.shape)
                print('L_old: ', L_old)
                old_to_new = build_old_to_new(
                    L_old, unm_idx, src_idx, dst_idx
                )  # [B, L_old]
                merge_maps.append(old_to_new)
                num_tokens_merged += src_idx.shape[1]

                print('num_tokens_merged so far: ', num_tokens_merged)

            # print("x len after block :", x.shape[1])  # if you have it
        x = self.ln(x)
        if self.token_merging:
            return (
                x,
                token_sizes,
                key_padding_mask,
                owner_idx,
                num_tokens_merged,
                merge_maps,
            )
        else:
            return x, key_padding_mask


def compose_old_to_new(maps, L0):
    # maps: list of old_to_new tensors from early->late
    # Start with identity mapping for original positions
    composed = torch.arange(L0, device=maps[0].device)[None, :].expand(
        maps[0].size(0), L0
    )

    for old_to_new in maps:
        # old_to_new maps positions of "current old length" -> "current new length"
        composed = old_to_new.gather(1, composed)
    return composed  # [B, L0], maps original pos -> final merged pos


def bipartite_soft_matching(
    k: torch.Tensor,
    r: int,
    protected: int = 1,  # number of leading tokens to never merge (e.g., cls=1)
):
    """
    Build a merge operator based on bipartite soft matching.

    Args:
        k: attention keys, shape [B, L, D] (typically averaged over heads already)
        r: number of merges to perform in this layer (reduces token count by r)
        protected: first `protected` tokens are never merged (e.g., CLS token at index 0)

    Returns:
        merge_fn(x): merges any tensor x with shape [B, L, D] into [B, L - r, D]
    """

    B, L, D = k.shape
    if r <= 0:
        return lambda x: x, None  # no merging

    # We can merge at most min(|A| - protected_in_A, |A|) tokens.
    # With even/odd split: A has ceil(L/2), B has floor(L/2)
    # A is even positions. B is odd positions after partitioning
    tA = (L + 1) // 2
    max_mergeable_in_A = max(0, tA - math.ceil(protected / 2))

    r = min(r, max_mergeable_in_A)

    # Cosine similarity on keys (normalize last dim)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    # Alternating partition: A = even positions, B = odd positions
    a = k[:, ::2, :]  # [B, tA, D]
    b = k[:, 1::2, :]  # [B, tB, D]

    # Similarity scores between A and B
    scores = a @ b.transpose(-1, -2)  # [B, tA, tB]

    # Prevent protected tokens (by absolute index) from being merged by setting their rows to -inf.
    # Applying abs_idx // 2 because only tokens in A (even indices) can be sources for merging
    for abs_idx in range(protected):
        if abs_idx % 2 == 0:
            scores[:, abs_idx // 2, :] = -math.inf  # don't allow merges for that A node

    node_max, node_idx = scores.max(dim=-1)  # best B match per A token
    edge_order = node_max.argsort(dim=-1, descending=True)  # [B, tA]

    # Pick top-r A tokens to merge; the rest stay unmerged
    src_idx = edge_order[:, :r]  # [B, r]
    unm_idx = edge_order[:, r:]  # [B, tA-r]

    dst_idx = node_idx.gather(dim=-1, index=src_idx)  # [B, r]

    # Sort unmerged A indices so token order is stable-ish (CLS returns to front)
    unm_idx, _ = unm_idx.sort(dim=-1)

    print('src_idx: ', src_idx)
    print('dst_idx: ', dst_idx)
    print('unm_idx: ', unm_idx)

    def merge_fn(x: torch.Tensor) -> torch.Tensor:
        """
        Merge any x of shape [B, L, D] using the precomputed indices.
        """
        Bx, Lx, D = x.shape
        assert Bx == B and Lx == L, f"Expected x shape [B={B}, L={L}, *], got {x.shape}"

        src = x[:, ::2, :]  # [B, tA, D]
        dst = x[:, 1::2, :]  # [B, tB, D]

        # Gather unmerged A tokens
        unm = src.gather(dim=1, index=unm_idx.unsqueeze(-1).expand(B, tA - r, D))

        # Gather the A tokens that will be merged (sources)
        src_m = src.gather(dim=1, index=src_idx.unsqueeze(-1).expand(B, r, D))

        # Add sources into their matched destinations (scatter_add over token dim)
        dst = dst.scatter_add(
            dim=1, index=dst_idx.unsqueeze(-1).expand(B, r, D), src=src_m
        )

        # Concatenate: [unmerged A tokens, updated B tokens]
        return torch.cat([unm, dst], dim=1)  # [B, (tA-r)+tB = L-r, D]

    return merge_fn, (unm_idx, src_idx, dst_idx, L)


def build_old_to_new(
    L: int, unm_idx: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor
) -> torch.Tensor:
    """
    L: old length (before merge)
    unm_idx: [B, tA-r] indices in A-space that remain
    src_idx: [B, r] indices in A-space that merged away
    dst_idx: [B, r] indices in B-space destinations
    Returns:
      old_to_new: [B, L] mapping old token index -> new token index
    """
    B = unm_idx.shape[0]
    device = unm_idx.device
    tA = (L + 1) // 2
    tB = L // 2

    print('src_idx: ', src_idx)
    print('tA: ', tA)

    # sanity checks
    assert (src_idx >= 0).all() and (src_idx < tA).all()
    assert (unm_idx >= 0).all() and (unm_idx < tA).all()
    assert (dst_idx >= 0).all() and (dst_idx < tB).all()

    r = src_idx.shape[1]
    len_unm = tA - r  # number of unmerged A tokens in output

    old_to_new = torch.empty(B, L, device=device, dtype=torch.long)

    # new index for each A token (A-space length tA)
    newA = torch.empty(B, tA, device=device, dtype=torch.long)

    # unmerged A tokens go to positions [0 .. len_unm-1] in output, in the order of unm_idx
    newA.scatter_(
        1, unm_idx, torch.arange(len_unm, device=device)[None, :].expand(B, len_unm)
    )

    # merged A tokens map to their destination B token position in output: len_unm + dst
    newA.scatter_(1, src_idx, len_unm + dst_idx)

    # even absolute indices are A tokens
    old_to_new[:, ::2] = newA

    # odd absolute indices are B tokens; they always map to len_unm + j
    old_to_new[:, 1::2] = len_unm + torch.arange(tB, device=device)[None, :].expand(
        B, tB
    )

    return old_to_new


def merge_with_sizes(
    x: torch.Tensor,  # [B, L, D]
    s: torch.Tensor,  # [B, L, 1] or [B, L] token sizes
    merge_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply merge_fn to x using size-weighted averaging.
    Also updates token sizes consistently.

    Returns:
        x_merged: [B, L-r, D]
        s_merged: [B, L-r, 1]
    """
    if s.dim() == 2:
        s = s.unsqueeze(-1)  # [B,T,1]

    # Weighted sum then divide by merged sizes.
    # Each token contributes proportionally to how many tokens it represents (x * s)
    x_sum = merge_fn(x * s)  # [B, L-r, D]
    s_new = merge_fn(s)  # [B, L-r, 1]
    x_new = x_sum / (s_new + eps)  # size-weighted average
    return x_new, s_new


def proportional_attention_logits(
    attn_logits: torch.Tensor,  # [B, H, L, L]
    s: torch.Tensor,  # [B, L, 1] or [B, L]
) -> torch.Tensor:
    """
    Implements Eq. (1): softmax(QK^T/sqrt(d) + log s)
    Specifically adds log(s_k) to each key column.
    """
    if s.dim() == 3:
        s = s.squeeze(-1)  # [B,L]
    # broadcast to [B, 1, 1, L] to add per-key
    return attn_logits + torch.log(s.clamp_min(1e-8))[:, None, None, :]


class ToMeAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        token_sizes=None,
        attn_mask=None,  # [L, L] or [B, L, L] or [B, H, L, L]
        key_padding_mask=None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            token_sizes: [B, L, 1] or [B, L] or None

        Returns:
            out: [B, L, D]
            k_merge: [B, L, head_dim]  (averaged over heads)
        """

        B, L, D = x.shape

        qkv = self.qkv(x)  # [B, L, 3D]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q,k,v: [B, H, L, d]
        assert q.shape == (B, self.num_heads, L, self.head_dim)
        assert k.shape == (B, self.num_heads, L, self.head_dim)
        assert v.shape == (B, self.num_heads, L, self.head_dim)

        # Attention logits
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        if attn_mask is not None:
            m = attn_mask[None, None, :, :]
            attn_logits = attn_logits.masked_fill(m, float("-inf"))

        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :]  # True = masked
            attn_logits = attn_logits.masked_fill(kpm, float("-inf"))

        print('token_sizes shape: ', None if token_sizes is None else token_sizes.shape)
        print('attn_logits shape before token sizes: ', attn_logits.shape)

        if token_sizes is not None:
            if token_sizes.dim() == 3:
                token_sizes = token_sizes.squeeze(-1)  # [B,L]
            attn_logits = (
                attn_logits + torch.log(token_sizes.clamp_min(1e-8))[:, None, None, :]
            )

        attn = attn_logits.softmax(dim=-1)
        out = attn @ v  # [B, H, L, d]

        print('B: ', B)
        print('L: ', L)
        print('D: ', D)
        print('out shape: ', out.shape)

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)

        # --- Keys for ToMe matching ---
        # Average over heads
        k_merge = k.mean(dim=1)  # [B, L, d]

        return out, k_merge


class FlashSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        x: [B, L, D]
        key_padding_mask: [B, L] bool, True for PAD (masked out)
        """
        B, L, D = x.shape
        qkv = self.qkv(x)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, L, D]

        # -> [B, H, L, Hd]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: True for PAD => disallow attending to those keys
            # We make additive mask of shape [B, 1, 1, L] to broadcast over heads and query length
            # allowed = ~pad
            allowed = ~key_padding_mask  # [B, L]
            attn_mask = torch.zeros((B, 1, 1, L), device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(~allowed[:, None, None, :], float("-inf"))

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [B, H, L, Hd]

        y = y.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        return self.proj(y)

class LatentEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        n_layers: int,
        mlp_ratio: float,
        token_merging: bool,
        r: int,
    ):
        super().__init__()
        self.token_merging = token_merging
        self.ln = nn.LayerNorm(embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockLocalEncode(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    build_local_attn_mask=False,
                    token_merging=token_merging,
                    r=r,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, key_padding_mask=None, token_sizes: Optional[torch.Tensor] = None):
        # for block in self.blocks:
        #     x, token_sizes, key_padding_mask, info = block(
        #         x,
        #         key_padding_mask=key_padding_mask,
        #         token_sizes=token_sizes,
        #         return_info=self.token_merging,
        #     )
        # return x

        num_latent_tokens_merged = 0
        latent_tokens_merge_maps = []  # store (old_to_new, L_old, L_new) per merge step
        # print("x len before block:", x.shape[1])
        for i, block in enumerate(self.blocks):
            x, token_sizes, key_padding_mask, info = block(
                x,
                key_padding_mask=key_padding_mask,
                token_sizes=token_sizes,
                return_info=True,
            )

            if info is not None:
                unm_idx, src_idx, dst_idx, L_old = info
                old_to_new = build_old_to_new(
                    L_old, unm_idx, src_idx, dst_idx
                )  # [B, L_old]
                latent_tokens_merge_maps.append(old_to_new)
                num_latent_tokens_merged += src_idx.shape[1]

            # print("x len after block :", x.shape[1])  # if you have it
        x = self.ln(x)
        if self.token_merging:
            return (
                x,
                token_sizes,
                key_padding_mask,
                num_latent_tokens_merged,
                latent_tokens_merge_maps,
            )
        else:
            return x, key_padding_mask


class LatentDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        n_layers: int,
        mlp_ratio: float,
        r: int,
        token_merging: bool = False,
    ):
        super().__init__()
        self.token_merging = token_merging
        self.blocks = nn.ModuleList(
            [
                TransformerBlockLocalEncode(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    build_local_attn_mask=False,
                    token_merging=self.token_merging,
                    r=r,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, key_padding_mask=None, token_sizes: Optional[torch.Tensor] = None):
        for block in self.blocks:
            x, _, key_padding_mask, info = block(
                x,
                key_padding_mask=key_padding_mask,
                token_sizes=token_sizes,
                return_info=self.token_merging,
            )
        return x


class TransformerBlockLocalDecode(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        window_size: int = 16,
    ):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        )  # for multi-head attention to work correctly
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = FlashSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
        hidden = int(embedding_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embedding_dim),
        )
        self.window_size = window_size

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.ln1(x)
        attn_out = self.attn(h, key_padding_mask=key_padding_mask)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


class localDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float,
        window_size: int,
        vocab_size: int,
        n_layers: int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlockLocalDecode(
                    embedding_dim, num_heads, mlp_ratio, window_size
                )
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.out_proj = nn.Linear(
            embedding_dim, vocab_size, bias=False
        )  # no bias as it is used in layernorm

    def forward(
        self,
        z_bar: torch.Tensor,
        L0: int,
        key_padding_mask: Optional[torch.Tensor] = None,
        merge_maps: list = None,
    ) -> torch.Tensor:
        B, L, D = z_bar.shape

        if merge_maps is None:
            # create an identity function of size [L, L] and broadcast to B such that the resulting shape is [B, L, L]
            U = torch.eye(L, device=z_bar.device).unsqueeze(0).expand(B, L, L)  # [B, L, L]

        else:
            final_map = compose_old_to_new(merge_maps, L0)  # [B, L0]

            # convert to binary source tensor
            U = torch.nn.functional.one_hot(final_map, num_classes=L).to(torch.float32)
        x = U @ z_bar

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.final_ln(x)
        logits = self.out_proj(x)
        return logits


class Autoencoder:
    def __init__(self, **kwargs: Unpack[AutoencoderKwargs]):
        self._kwargs = kwargs
        self._embed_dim = kwargs["embedding_dim"]
        self._num_heads = kwargs["num_heads"]
        self._mlp_ratio = kwargs["mlp_ratio"]
        self._window_size = kwargs["window_size"]
        self._vocab_size = kwargs["vocab_size"]
        self._n_layers = kwargs["n_layers"]
        self._max_len = kwargs["max_len"]
        self.device = self._kwargs["device"]
        self.lr = self._kwargs["lr"]
        self.grad_clipping = self._kwargs["grad_clipping"]
        self.training_iterations = self._kwargs["training_iterations"]
        self.warmup_iterations = self._kwargs["warmup_iterations"]
        self.r = self._kwargs["r"]
        self.token_merging = self._kwargs["token_merging"]
        self.down_weight_factor = self._kwargs["down_weight_factor"]
        self.token_merging_latent_encoder = self._kwargs["token_merging_latent_encoder"]

        self._downsize = kwargs["downsize"]  # ratio to reduce the size of the model
        # embedding_dim = 64 # mergeDNA use 1024
        self._block_num_local_encoder = 4 // self._downsize
        self._block_num_latent_encoder = 20 // self._downsize
        self._block_num_latent_decoder = 4 // self._downsize
        self._block_num_local_decoder = 2 // self._downsize

        self.localEncoder = localEncoder(
            embedding_dim=self._embed_dim,
            num_heads=self._num_heads,
            mlp_ratio=self._mlp_ratio,
            window_size=self._window_size,
            vocab_size=self._vocab_size,
            n_layers=self._block_num_local_encoder,
            # n_layers=self._n_layers,
            max_len=self._max_len,
            token_merging=self.token_merging,
            r=self.r,
        ).to(self.device)

        self.latentEncoder = LatentEncoder(
            embedding_dim=self._embed_dim,
            num_heads=self._num_heads,
            mlp_ratio=self._mlp_ratio,
            n_layers=self._block_num_latent_encoder,
            # n_layers=self._n_layers,
            token_merging=self.token_merging_latent_encoder,
            r=self.r,
        ).to(self.device)

        self.latentDecoder = LatentDecoder(
            embedding_dim=self._embed_dim,
            num_heads=self._num_heads,
            mlp_ratio=self._mlp_ratio,
            n_layers=self._block_num_latent_decoder,
            # n_layers=self._n_layers,
            token_merging=False,    # never merge tokens when decoding
            r=self.r,
        ).to(self.device)

        self.localDecoder = localDecoder(
            embedding_dim=self._embed_dim,
            num_heads=self._num_heads,
            mlp_ratio=self._mlp_ratio,
            window_size=self._window_size,
            vocab_size=self._vocab_size,
            n_layers=self._block_num_local_decoder,
            # n_layers=self._n_layers,
        ).to(self.device)

        # combine all parameters
        self.all_params = (
            list(self.localEncoder.parameters())
            + list(self.localDecoder.parameters())
            + list(self.latentEncoder.parameters())
            + list(self.latentDecoder.parameters())
        )

        self.optimizer = torch.optim.AdamW(
            self.all_params, lr=self.lr, weight_decay=1e-8, betas=(0.9, 0.95)
        )

        self.warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-3,  # start at ~0
            end_factor=1.0,  # reach base_lr
            total_iters=self.warmup_iterations,
        )

        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.training_iterations - self.warmup_iterations,
            eta_min=1e-6,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[self.warmup_iterations],
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        ID_PAD: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass. Returns logits [B, L, vocab_size]."""
        print("In forward function")
        B, L0 = input_ids.shape
        merge_maps = None
        num_tokens_merged = 0
        token_sizes = None
        if self.token_merging:
            (
                z,
                token_sizes,
                key_padding_mask,
                orig_to_cur,
                num_tokens_merged,
                merge_maps,
            ) = self.localEncoder.forward(input_ids)
        else:
            (
                z,
                key_padding_mask,
            ) = self.localEncoder.forward(input_ids)

        if self.token_merging_latent_encoder:
            assert self.token_merging, "token_sizes is required for latent encoder merging"
            (
                z,
                token_sizes,
                key_padding_mask,
                num_tokens_merged,
                merge_maps,
            ) = self.latentEncoder.forward(z, key_padding_mask=key_padding_mask, token_sizes=token_sizes)
        else:
            z, key_padding_mask = self.latentEncoder.forward(z, key_padding_mask=key_padding_mask)
        z = self.latentDecoder.forward(z, key_padding_mask=key_padding_mask)
        # change key_padding_mask to match input shape
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = input_ids == ID_PAD
        logits = self.localDecoder.forward(
            z, key_padding_mask=key_padding_mask, merge_maps=merge_maps, L0=L0
        )
        return logits, num_tokens_merged

    def forward_no_grad_local_encoder(
        self,
        input_ids: torch.Tensor,
        ID_PAD: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass. Returns logits [B, L, vocab_size]."""
        print("In forward no grad local encoder function")
        B, L0 = input_ids.shape
        merge_maps = None
        num_tokens_merged = 0
        token_sizes = None
        # if self.token_merging:
        #     with torch.no_grad():
        #         (
        #             z,
        #             token_sizes,
        #             key_padding_mask,
        #             orig_to_cur,
        #             num_tokens_merged,
        #             merge_maps,
        #         ) = self.localEncoder.forward(input_ids)
        # else:
        #     with torch.no_grad():
        #         (
        #             z,
        #             key_padding_mask,
        #         ) = self.localEncoder.forward(input_ids)

        if self.token_merging:
            (
                z,
                token_sizes,
                key_padding_mask,
                orig_to_cur,
                num_tokens_merged,
                merge_maps,
            ) = self.localEncoder.forward(input_ids)
        else:
            (
                z,
                key_padding_mask,
            ) = self.localEncoder.forward(input_ids)

        if self.token_merging_latent_encoder:
            assert self.token_merging, "token_sizes is required for latent encoder merging"
            (
                z,
                token_sizes,
                key_padding_mask,
                num_tokens_merged,
                merge_maps,
            ) = self.latentEncoder.forward(z, key_padding_mask=key_padding_mask, token_sizes=token_sizes)
        else:
            z, key_padding_mask = self.latentEncoder.forward(z, key_padding_mask=key_padding_mask)
        z = self.latentDecoder.forward(z, key_padding_mask=key_padding_mask)
        # change key_padding_mask to match input shape
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = input_ids == ID_PAD
        logits = self.localDecoder.forward(
            z, key_padding_mask=key_padding_mask, merge_maps=merge_maps, L0=L0
        )
        return logits, num_tokens_merged

    def compute_loss_mtr(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        ID_PAD: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Token-level cross entropy, ignoring PAD positions."""
        # logits: [B, L, V], targets: [B, L]
        B, L, V = logits.shape
        logits_flat = logits.view(B * L, V)
        targets_flat = target_ids.view(B * L)

        if attention_mask is not None:
            valid = attention_mask.view(B * L).bool()
        else:
            valid = targets_flat != ID_PAD

        # filter
        logits_flat = logits_flat[valid]
        targets_flat = targets_flat[valid]

        return F.cross_entropy(logits_flat, targets_flat)

    def update(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ID_PAD: int
    ) -> Dict[str, float]:
        """
        :param batch: input batch
        :return:
        """
        self.localEncoder.train()
        self.localDecoder.train()
        self.latentEncoder.train()
        self.latentDecoder.train()

        # target is the original sequence
        target_ids = input_ids

        self.optimizer.zero_grad(set_to_none=True)

        logits, num_tokens_merged = self.forward(input_ids, ID_PAD)
        logits_no_grad_local_encoder, _ = self.forward_no_grad_local_encoder(
            input_ids.clone(), ID_PAD
        )

        loss = self.compute_loss_mtr(logits, target_ids, ID_PAD)
        loss_no_grad_local_encoder = self.compute_loss_mtr(
            logits_no_grad_local_encoder, target_ids, ID_PAD
        )

        total_loss = loss + self.down_weight_factor * loss_no_grad_local_encoder
        total_loss.backward()

        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.all_params, self.grad_clipping
        ).item()
        self.optimizer.step()
        self.step_scheduler()

        with torch.no_grad():
            # token accuracy (excluding PAD)
            pred = logits.argmax(dim=-1)
            if attention_mask is not None:
                valid = attention_mask.bool()
            else:
                valid = target_ids != ID_PAD
            correct = ((pred == target_ids) & valid).sum().item()
            total = valid.sum().item()
            acc = float(correct) / float(total) if total > 0 else 0.0

        logs = {"loss_mtr": float(loss.item()), "acc": acc}
        logs["loss_mtr_no_local_encoder"] = float(loss_no_grad_local_encoder.item())
        logs["total_loss"] = float(total_loss.item())
        if grad_norm is not None:
            logs["grad_norm"] = float(grad_norm)

        # for debugging LR schedules
        logs["lr"] = float(self.optimizer.param_groups[0]["lr"])

        logs["num_tokens_merged"] = num_tokens_merged

        return logs

    def eval(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ID_PAD: int):
        self.localEncoder.eval()
        self.localDecoder.eval()
        self.latentEncoder.eval()
        self.latentDecoder.eval()

        # target is the original sequence
        target_ids = input_ids

        self.optimizer.zero_grad(set_to_none=True)

        logits, num_tokens_merged = self.forward(input_ids, ID_PAD)
        loss = self.compute_loss_mtr(logits, target_ids, ID_PAD)

        pred = logits.argmax(dim=-1)
        if attention_mask is not None:
            valid = attention_mask.bool()
        else:
            valid = (target_ids != ID_PAD)

        correct = ((pred == target_ids) & valid).sum().item()
        total = valid.sum().item()
        acc = float(correct) / float(total) if total > 0 else 0.0

        return {"loss": float(loss.item()), "acc": acc}


    def step_scheduler(self):
        self.scheduler.step()

    @property
    def name(self):
        return self._kwargs["name"]
