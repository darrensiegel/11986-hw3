"""
For additional transformer related

Sequential
Embedding

"""
import math
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        weight_init = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        weight_tensor = tensor_from_numpy(weight_init, backend=backend, requires_grad=True)
        self.weights = Parameter(weight_tensor)
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        one_hot_x = one_hot(x, self.num_embeddings)
        flat = one_hot_x.contiguous().view(bs * seq_len, self.num_embeddings)
        embeddings = flat @ self.weights.value
        return embeddings.view(bs, seq_len, self.embedding_dim)
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN ASSIGN3_2
        if not self.training or self.p_dropout == 0.0:
            return x

        keep_prob = 1.0 - self.p_dropout
        if keep_prob <= 0.0:
            return zeros(x.shape, backend=x.backend)

        mask_np = np.random.binomial(1, keep_prob, size=x.shape).astype(np.float32)
        mask = tensor_from_numpy(mask_np, backend=x.backend)
        scaled_mask = mask / keep_prob
        return x * scaled_mask
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        self.backend = backend
        self.in_size = in_size
        bound = 1.0 / math.sqrt(in_size)
        weight_init = np.random.uniform(-bound, bound, size=(in_size, out_size)).astype(np.float32)
        weight_tensor = tensor_from_numpy(weight_init, backend=backend, requires_grad=True)
        self.weights = Parameter(weight_tensor)

        self.use_bias = bias
        if bias:
            bias_init = np.random.uniform(-bound, bound, size=(out_size,)).astype(np.float32)
            bias_tensor = tensor_from_numpy(bias_init, backend=backend, requires_grad=True)
            self.bias = Parameter(bias_tensor)
        else:
            self.bias = None
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        ### BEGIN ASSIGN3_2
        result = x @ self.weights.value
        if self.use_bias and self.bias is not None:
            result = result + self.bias.value.view(1, self.out_size)
        return result
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        self.backend = backend
        weight_tensor = ones((dim,), backend=backend)
        weight_tensor.requires_grad_(True)
        self.weights = Parameter(weight_tensor)

        bias_tensor = zeros((dim,), backend=backend)
        bias_tensor.requires_grad_(True)
        self.bias = Parameter(bias_tensor)
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        mean = x.mean(dim=1)
        variance = x.var(dim=1)
        std = (variance + self.eps) ** 0.5
        normalized = (x - mean) / std

        weight = self.weights.value.view(1, self.dim)
        bias = self.bias.value.view(1, self.dim)
        return normalized * weight + bias
        ### END ASSIGN3_2
