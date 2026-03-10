"""
Set Transformer implementation with Pre-LayerNorm.

This module implements a Set Transformer architecture for processing sets of embeddings
with two types of token embeddings. Uses Pre-LayerNorm for improved training stability.

Reference: Lee et al., "Set Transformer: A Framework for Attention-based
           Permutation-Invariant Neural Networks", ICML 2019.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_q, embed_dim]
            key: [batch, seq_k, embed_dim]
            value: [batch, seq_k, embed_dim]
            key_padding_mask: [batch, seq_k] - True for positions to mask

        Returns:
            output: [batch, seq_q, embed_dim]
        """
        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]

        # Project Q, K, V
        q = self.q_proj(query)  # [batch, seq_q, embed_dim]
        k = self.k_proj(key)  # [batch, seq_k, embed_dim]
        v = self.v_proj(value)  # [batch, seq_k, embed_dim]

        # Reshape to [batch, num_heads, seq, head_dim]
        q = q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # [batch, heads, seq_q, seq_k]

        # Apply mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_k] -> [batch, 1, 1, seq_k]
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, heads, seq_q, head_dim]

        # Reshape back
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_q, self.embed_dim)
        )
        output = self.out_proj(output)

        return output


class FeedForward(nn.Module):
    """
    Feed-Forward Network with GELU activation.

    Args:
        embed_dim: Input/output dimension
        hidden_dim: Hidden layer dimension (default: 4 * embed_dim)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PreNormTransformerBlock(nn.Module):
    """
    Transformer block with Pre-LayerNorm.

    Order of operations:
    1. LayerNorm -> Multi-Head Attention -> Residual
    2. LayerNorm -> Feed-Forward -> Residual

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        ffn_dim: Hidden dimension of feed-forward network
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            key_padding_mask: [batch, seq_len] - True for positions to mask

        Returns:
            output: [batch, seq_len, embed_dim]
        """
        # Pre-Norm Self-Attention + Residual
        x_norm = self.norm1(x)
        x = x + self.dropout(self.attn(x_norm, x_norm, x_norm, key_padding_mask))

        # Pre-Norm FFN + Residual
        x_norm = self.norm2(x)
        x = x + self.dropout(self.ffn(x_norm))

        return x


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention Block (MAB) from Set Transformer paper.
    Used for cross-attention between two sets.

    MAB(X, Y) = LayerNorm(H + FFN(H))
    where H = LayerNorm(X + MultiHead(X, Y, Y))

    With Pre-Norm variant:
    MAB(X, Y) = X + Attention(LN(X), LN(Y)) -> + FFN(LN(.))

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        ffn_dim: Hidden dimension of feed-forward network
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_q, embed_dim] - queries
            key_value: [batch, seq_kv, embed_dim] - keys and values
            key_padding_mask: [batch, seq_kv] - True for positions to mask

        Returns:
            output: [batch, seq_q, embed_dim]
        """
        # Pre-Norm Cross-Attention + Residual
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        x = query + self.dropout(self.attn(q_norm, kv_norm, kv_norm, key_padding_mask))

        # Pre-Norm FFN + Residual
        x_norm = self.norm2(x)
        x = x + self.dropout(self.ffn(x_norm))

        return x


class InducedSetAttentionBlock(nn.Module):
    """
    Induced Set Attention Block (ISAB) from Set Transformer paper.

    Uses inducing points to reduce complexity from O(n^2) to O(nm) where
    n is sequence length and m is number of inducing points.

    ISAB_m(X) = MAB(X, MAB(I, X))
    where I is a set of m learnable inducing points.

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        num_inducing_points: Number of inducing points (m)
        ffn_dim: Hidden dimension of feed-forward network
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_inducing_points: int = 32,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.inducing_points = nn.Parameter(
            torch.randn(1, num_inducing_points, embed_dim)
        )
        nn.init.xavier_uniform_(self.inducing_points)

        self.mab1 = MultiHeadAttentionBlock(embed_dim, num_heads, ffn_dim, dropout)
        self.mab2 = MultiHeadAttentionBlock(embed_dim, num_heads, ffn_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            key_padding_mask: [batch, seq_len] - True for positions to mask

        Returns:
            output: [batch, seq_len, embed_dim]
        """
        batch_size = x.shape[0]

        # Expand inducing points for batch
        inducing_points = self.inducing_points.expand(batch_size, -1, -1)

        # H = MAB(I, X) - inducing points attend to input
        h = self.mab1(inducing_points, x, key_padding_mask)

        # Output = MAB(X, H) - input attends to inducing points
        output = self.mab2(x, h)

        return output


class PoolingByMultiheadAttention(nn.Module):
    """
    Pooling by Multihead Attention (PMA) from Set Transformer paper.

    Aggregates set to fixed-size output using learnable seed vectors.

    PMA_k(Z) = MAB(S, Z)
    where S is a set of k learnable seed vectors.

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        num_outputs: Number of output vectors (k)
        ffn_dim: Hidden dimension of feed-forward network
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_outputs: int = 1,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.seed_vectors = nn.Parameter(torch.randn(1, num_outputs, embed_dim))
        nn.init.xavier_uniform_(self.seed_vectors)

        self.mab = MultiHeadAttentionBlock(embed_dim, num_heads, ffn_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            key_padding_mask: [batch, seq_len] - True for positions to mask

        Returns:
            output: [batch, num_outputs, embed_dim]
        """
        batch_size = x.shape[0]

        # Expand seed vectors for batch
        seed_vectors = self.seed_vectors.expand(batch_size, -1, -1)

        # Seeds attend to input
        output = self.mab(seed_vectors, x, key_padding_mask)

        return output


class SetTransformer(nn.Module):
    """
    Set Transformer with Pre-LayerNorm and dual token embeddings.

    Architecture:
    1. Input embeddings + Token Embedding 1 + Token Embedding 2
    2. N x Transformer Blocks (ISAB or standard self-attention)
    3. Pooling (PMA) to aggregate set
    4. Output projection

    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        ffn_dim: Hidden dimension of FFN (default: 4 * embed_dim)
        dropout: Dropout probability
        num_token_types_1: Vocabulary size for first token type
        num_token_types_2: Vocabulary size for second token type
        use_isab: Whether to use ISAB (induced) or standard self-attention
        num_inducing_points: Number of inducing points for ISAB
        num_outputs: Number of output vectors from pooling
        output_dim: Final output dimension (if different from embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_isab: bool = False,
        num_inducing_points: int = 32,
        num_outputs: int = 1,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_outputs = num_outputs

        # Dropout after embedding combination
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_isab:
                self.layers.append(
                    InducedSetAttentionBlock(
                        embed_dim, num_heads, num_inducing_points, ffn_dim, dropout
                    )
                )
            else:
                self.layers.append(
                    PreNormTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                )

        # Pooling layer
        self.pooling = PoolingByMultiheadAttention(
            embed_dim, num_heads, num_outputs, ffn_dim, dropout
        )

        # Final layer norm (for pre-norm architecture)
        self.final_norm = nn.LayerNorm(embed_dim)

        # Output projection
        output_dim = output_dim or embed_dim
        self.output_proj = nn.Linear(embed_dim * num_outputs, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard transformer practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_input_proj(self, input_dim: int):
        """Set input projection if input dimension differs from embed_dim."""
        if input_dim != self.embed_dim:
            self.input_proj = nn.Linear(input_dim, self.embed_dim)
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through Set Transformer.

        Order of operations:
        1. Apply dropout
        2. Pass through N transformer blocks
        3. Pool to fixed-size output
        4. Final layer norm
        5. Project to output dimension

        Args:
            embeddings: [batch, seq_len, input_dim] - input embeddings
            padding_mask: [batch, seq_len] - True for padded positions to ignore

        Returns:
            dict with keys:
                - 'output': [batch, output_dim] - final output
                - 'pooled': [batch, num_outputs, embed_dim] - pooled representations
                - 'hidden': [batch, seq_len, embed_dim] - last hidden states
        """
        # Step 1: Apply dropout
        x = self.embed_dropout(x)

        # Step 2: Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, padding_mask)

        # Store hidden states before pooling
        hidden = x

        # Step 3: Pool to fixed-size output
        pooled = self.pooling(x, padding_mask)  # [batch, num_outputs, embed_dim]

        # Step 4: Final layer norm
        pooled = self.final_norm(pooled)

        # Step 5: Project to output dimension
        # Flatten pooled outputs if multiple
        pooled_flat = pooled.view(
            pooled.shape[0], -1
        )  # [batch, num_outputs * embed_dim]
        output = self.output_proj(pooled_flat)  # [batch, output_dim]

        return {
            "output": output,
            "pooled": pooled,
            "hidden": hidden,
        }


class SetTransformerEncoder(nn.Module):
    """
    Set Transformer Encoder (without pooling) for use as a backbone.

    Returns the transformed set embeddings without aggregation.
    Useful when you want to process a set and then apply custom pooling.

    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        ffn_dim: Hidden dimension of FFN
        dropout: Dropout probability
        num_token_types_1: Vocabulary size for first token type
        num_token_types_2: Vocabulary size for second token type
        use_isab: Whether to use ISAB or standard self-attention
        num_inducing_points: Number of inducing points for ISAB
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_isab: bool = False,
        num_inducing_points: int = 32,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_isab:
                self.layers.append(
                    InducedSetAttentionBlock(
                        embed_dim, num_heads, num_inducing_points, ffn_dim, dropout
                    )
                )
            else:
                self.layers.append(
                    PreNormTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                )

        # Final norm for pre-norm architecture
        self.final_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_input_proj(self, input_dim: int):
        if input_dim != self.embed_dim:
            self.input_proj = nn.Linear(input_dim, self.embed_dim)
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, seq_len, input_dim]
            padding_mask: [batch, seq_len] - True for padded positions

        Returns:
            hidden: [batch, seq_len, embed_dim] - transformed embeddings
        """
        x = self.embed_dropout(x)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, padding_mask)

        # Final norm
        x = self.final_norm(x)

        return x
