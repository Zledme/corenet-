#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Tuple

import torch
from torch import Tensor


def _negate_half(x: Tensor) -> Tensor:
    """
    Computes the negative half of the input tensor along the last dimension.

    Args:
        x: Input tensor.

    Returns:
        Tensor with the negative second half preceding the first half along the last dimension.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoMaFunctions:
    @staticmethod
    def e_to_q(roll, pitch, yaw):
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
    
        w = cr * cp * cy - sr * sp * sy
        x = sr * cp * cy + cr * sp * sy
        y = cr * sp * cy - sr * cp * sy
        z = cr * cp * sy + sr * sp * cy
    
        return torch.stack([w, x, y, z], dim=0)

    @staticmethod
    def quat_conj(quaternion):
        return torch.stack([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]], dim=0)

    @staticmethod
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w_res = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x_res = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y_res = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z_res = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w_res, x_res, y_res, z_res], dim=0)

    @staticmethod
    def qvq_multiply(quaternion, a, b, c):
        v_quaternion = torch.stack([ torch.zeros_like(a), a,b,c], dim = 0)
        q_conjugate = RoMaFunctions.quat_conj(quaternion)
        intermediate = RoMaFunctions.quaternion_multiply(quaternion, v_quaternion)
        rotated_vector_quaternion = RoMaFunctions.quaternion_multiply(intermediate, q_conjugate)
        # breakpoint()
        x_res, y_res, z_res = rotated_vector_quaternion[1:]

        return torch.stack([x_res, y_res, z_res], dim = -1).flatten(3)

# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     print("0..dim", dim, end)
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 3)[: (dim // 3)].float() / dim))
#     print("1...",freqs.shape)
#     t = torch.arange(end, device=freqs.device)  # type: ignore
#     print("2...",t.shape)
#     freqs = torch.outer(t, freqs).float()  # type: ignore
#     print("3....",freqs.shape)
#     freqs_cos = torch.cos(freqs)  # real part
#     freqs_sin = torch.sin(freqs)  # imaginary part
#     return freqs

# def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#     ndim = x.ndim
#     print("freqs.shape", freqs_cis.shape)
#     assert 0 <= 1 < ndim
#     # breakpoint()
#     assert freqs_cis.shape == (x.shape[1], x.shape[-1])
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#     return freqs_cis.view(shape)



class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings (aka RoPE) from `RoFormer <https://arxiv.org/abs/2104.09864>`_.

    RoPE encodes the position information of tokens using a rotation matrix, and is able to capture
    explicit relative positional dependencies.

    Args:
        model_dim: The dimensionality of the model's hidden state.
        max_seq_length: Maximum sequence length.
        freq_constant: A constant used for computing frequencies.
    """

    def __init__(
        self, model_dim: int, max_seq_length: int, freq_constant: int = 10000
    ) -> None:
        inv_freq = 1.0 / (
            freq_constant
            ** (torch.arange(0, model_dim, 3, dtype=torch.float32) / model_dim)
        )
        super().__init__()

        self.model_dim = model_dim
        self.freq_constant = freq_constant
        self.max_seq_length = max_seq_length

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = max_seq_length
        self._compute_sin_cos_embeddings(max_seq_length)

    def extra_repr(self) -> str:
        return f"\tmodel_dim={self.model_dim}, max_seq_length={self.max_seq_length}, freq_constant={self.freq_constant}"

    def _compute_sin_cos_embeddings(
        self,
        key_len: int,
        key_device: torch.device = torch.device("cpu"),
        key_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Compute sine and cos embeddings.

        Args:
            key_len: Number of tokens in the key embeddings in the transformer model.
            device: Device where the key embeddings are stored.
            key_dtype: Data type of the key embeddings.

        Returns:
            None

        ...note:
            We recalculate the sine and cosine embeddings if any of the following conditions are met:
                1. The number of tokens in key embeddings are greater than the cached sequence length.
                2. Sine and cosine caches are empty.
                3. The device and data type of sine and cosine embeddings does not match with the key embeddings.
        """
        if (
            key_len > self._cached_seq_length
            or self._cached_cos is None
            or (self._cached_cos is not None and self._cached_cos.device != key_device)
            or (self._cached_cos is not None and self._cached_cos.dtype != key_dtype)
            or self._cached_sin is None
            or (self._cached_sin is not None and self._cached_sin.device != key_device)
            or (self._cached_sin is not None and self._cached_sin.dtype != key_dtype)
        ):
            self._cached_seq_length = max(key_len, self._cached_seq_length)

            # The shape of 'pos_index' is [number of key tokens]
            pos_index = torch.arange(
                self._cached_seq_length,
                dtype=torch.float32,
                device=self.inv_freq.device,
            )
            #print("pos_In",pos_index.shape)
            # The shape of 'pos_index_theta' is [number of key tokens, model dimension]
            emb = torch.einsum("i,j->ij", pos_index, self.inv_freq)
            # The shape of 'emb' is [number of key tokens, model dimension]
            #emb = torch.cat((pos_index_theta, pos_index_theta, pos_index_theta), dim=-1)
            #print("emb:",emb.shape)
           # sin embeddings is [number of key tokens, model_dim]
            cos_emb = emb.to(dtype=key_dtype, device=key_device)
            sin_emb = emb.to(dtype=key_dtype, device=key_device)

            # the shape of cached cos and sin embeddings is [1, 1, number of key tokens, model_dim]
            self._cached_cos = cos_emb[None, None, :, :]
            self._cached_sin = sin_emb[None, None, :, :]

    

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function of RoPE embeddings.

        Args:
            query: Query embeddings in the transformer model. The shape of query embeddings is
                [Batch, number of query heads, number of query tokens, model dimension].
            key: Key embeddings in the transformer model. The shape of key embeddings is
                [Batch, number of key heads, number of key tokens, model dimension].

        Returns:
            A tuple containing the query and key embeddings with positional information. The shape of the returned query
            and key embeddings is the same as the input query and key embeddings respectively.

        ...note:
            The RoPE embedding computation is done in full-precision. After the computation, input query and key tensors
            are casted to original input datatype.
        """
        dim = xk.shape[-1]
        key_len = xk.shape[2]
        query_len = xq.shape[2]

        assert dim == self.model_dim
        assert xk.device == xq.device
        assert xk.dtype == xq.dtype

        # In the context of self-attention, the lengths of keys and queries are equal.
        # However, in generation tasks, such as predicting the next token in a sequence, the lengths of keys and queries
        # can differ. For instance, when employing key-value (KV) caching for sequence prediction, the keys
        # represent embeddings of previous tokens and the current token, while the query corresponds
        # to the embedding of the current token only.
        assert (
            key_len >= query_len
        ), "Number of keys has to be greater than or equal to number of queries."


        xq_a, xq_b, xq_c = xq.float().reshape(xq.shape[:-1] + (-1, 3)).unbind(-1)
        xk_a, xk_b, xk_c = xk.float().reshape(xk.shape[:-1] + (-1, 3)).unbind(-1)

        #print("shape",xk.shape)
        self._compute_sin_cos_embeddings(
            xk_a.shape[2], key_device=xk_a.float().device, key_dtype=xk_a.float().dtype
        )


        #print(self._cached_cos.shape, xk.shape,xk_a.shape) 
        # print("moel_im", self.model_dim)
        # freqs = precompute_freqs_cis(self.model_dim, self.max_seq_length)      
        # print("freqs.shape",freqs.shape, xk_a.shape)
        # freqs = reshape_for_broadcast(self._cached_cos, xq_a)
        freqs = self._cached_cos
        #print("freqs.shape",freqs.shape, xk_a.shape)


        #breakpoint()
        rot_axis = torch.tensor([1.0], device=freqs.device) / torch.sqrt(torch.tensor(3.0))
        axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        half_angle = freqs / 2.0
        sin_half = torch.sin(half_angle)

        
        quaternion = torch.stack([torch.cos(half_angle), rot_axis * sin_half, rot_axis * sin_half, rot_axis * sin_half], dim=0)

        # print("quat.shape",quaternion.shape)

        xq_out = RoMaFunctions.qvq_multiply(quaternion, xq_a, xq_b, xq_c)
        xk_out = RoMaFunctions.qvq_multiply(quaternion, xk_a, xk_b, xk_c)
        
        return xq_out.type_as(xq), xk_out.type_as(xk)
