import numpy as np
from .tensor import tensor_from_numpy
from .module import Module
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    softmax,
    GELU,
)
datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd 
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        ### BEGIN ASSIGN3_3
        assert (
            self.attn_hidden_dim * self.n_head == self.n_embd
        ), "n_embd must be divisible by n_head"

        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        x_flat = x.contiguous().view(batch_size * seq_len, n_embd)

        q = self.q_projection(x_flat)
        k = self.k_projection(x_flat)
        v = self.v_projection(x_flat)

        q = q.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        k = k.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        v = v.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        kT = k.permute(0, 1, 3, 2)
        ### END ASSIGN3_3
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.
        
        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        ### BEGIN ASSIGN3_3
        scale = datatype(np.sqrt(self.attn_hidden_dim))
        attn_scores = (q @ kT) / scale

        if self.causal:
            causal_mask = self.create_causal_mask(queries_len)
            attn_scores = attn_scores + causal_mask

        attn_weights = softmax(attn_scores, dim=3)
        context = attn_weights @ v

        context = context.permute(0, 2, 1, 3).contiguous()
        result = context.view(batch_size, queries_len, self.n_embd)
        ### END ASSIGN3_3

        return result

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        q, kT, v = self.project_to_query_key_value(x)
        attn_output = self.self_attention(q, kT, v)

        attn_flat = attn_output.contiguous().view(batch_size * seq_len, n_embd)
        projected = self.out_projection(attn_flat)
        projected = projected.view(batch_size, seq_len, n_embd)
        projected = self.dropout(projected)

        return projected
        ### END ASSIGN3_3


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.backend = backend
        self.linear_in = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with GELU activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        x_flat = x.contiguous().view(batch_size * seq_len, n_embd)
        hidden = self.linear_in(x_flat)
        hidden = GELU(hidden)
        hidden = self.linear_out(hidden)
        hidden = hidden.view(batch_size, seq_len, n_embd)
        hidden = self.dropout(hidden)
        ### END ASSIGN3_3

        return hidden
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        self.backend = backend
        self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            causal=True,
            p_dropout=p_dropout,
            bias=bias,
            backend=backend,
        )
        self.ff = FeedForward(
            n_embd=n_embd,
            middle_dim=256,
            p_dropout=p_dropout,
            bias=bias,
            backend=backend,
        )
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        residual = x

        ln1_in = x.contiguous().view(batch_size * seq_len, n_embd)
        ln1_out = self.ln_1(ln1_in).view(batch_size, seq_len, n_embd)
        attn_out = self.attention(ln1_out)
        x = residual + attn_out

        ln2_in = x.contiguous().view(batch_size * seq_len, n_embd)
        ln2_out = self.ln_2(ln2_in).view(batch_size, seq_len, n_embd)
        ff_out = self.ff(ln2_out)
        x = x + ff_out

        return x
        ### END YOUR SOLUTION


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        ### BEGIN ASSIGN3_3
        self.token_embeddings = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        self.t_layer_1 = TransformerLayer(
            n_embd=n_embd,
            n_head=n_head,
            p_dropout=p_dropout,
            ln_eps=ln_eps,
            bias=bias,
            backend=backend,
        )
        self.t_layer_2 = TransformerLayer(
            n_embd=n_embd,
            n_head=n_head,
            p_dropout=p_dropout,
            ln_eps=ln_eps,
            bias=bias,
            backend=backend,
        )
        self.t_layer_3 = TransformerLayer(
            n_embd=n_embd,
            n_head=n_head,
            p_dropout=p_dropout,
            ln_eps=ln_eps,
            bias=bias,
            backend=backend,
        )
        self.t_layer_4 = TransformerLayer(
            n_embd=n_embd,
            n_head=n_head,
            p_dropout=p_dropout,
            ln_eps=ln_eps,
            bias=bias,
            backend=backend,
        )
        self.dropout = Dropout(p_dropout)
        self.ln = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)
        ### END ASSIGN3_3
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        ### BEGIN ASSIGN3_3
        token_embeds = self.token_embeddings(idx)

        position_ids_np = np.arange(seq_len, dtype=np.float32).reshape(1, seq_len)
        position_ids = tensor_from_numpy(position_ids_np, backend=self.backend)
        position_embeds = self.position_embeddings(position_ids)
        position_embeds = position_embeds.view(1, seq_len, self.n_embd)

        x = token_embeds + position_embeds
        x = self.dropout(x)

        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)

        x_flat = x.contiguous().view(batch_size * seq_len, self.n_embd)
        x_norm = self.ln(x_flat).view(batch_size, seq_len, self.n_embd)

        logits_flat = self.lm_head(x_norm.contiguous().view(batch_size * seq_len, self.n_embd))
        logits = logits_flat.view(batch_size, seq_len, self.n_vocab)

        return logits
        ### END ASSIGN3_3

