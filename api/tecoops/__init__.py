import torch

from ._torch_ext import (flatten_rays, morton3D_invert, reshape_and_cache, rms_norm,
                         flash_attn_varlen_func, causal_conv1d_fn_torch,
                         ms_deform_attn_forward, ms_deform_attn_backward)

__all__ = ['flatten_rays', 'morton3D_invert', 'reshape_and_cache', 'rms_norm',
           'flash_attn_varlen_func', 'causal_conv1d_fn_torch',
           'ms_deform_attn_forward', 'ms_deform_attn_backward', 'ms_deform_attn']


class _MSDeformAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, spatial_shapes, sampling_locations, attention_weights):
        if value.device.type != "sdaa":
            raise ValueError("ms_deform_attn requires SDAA tensors")
        ctx.save_for_backward(value, spatial_shapes, sampling_locations, attention_weights)
        return ms_deform_attn_forward(value, spatial_shapes, sampling_locations, attention_weights)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        value, shapes, locations, weights = ctx.saved_tensors
        gv, gl, gw = ms_deform_attn_backward(value, shapes, locations, weights, grad_output.contiguous())
        return gv, None, gl, gw


def ms_deform_attn(value, spatial_shapes, sampling_locations, attention_weights):
    """Differentiable SDAA MSDA. First order only; list traversal order is not deterministic.

    Inputs follow the raw forward contiguous layout and FP32/FP16 contract.
    spatial_shapes must contain positive H/W pairs summing to value.size(1).
    No CPU fallback or gradient for spatial_shapes is provided.
    """
    return _MSDeformAttn.apply(value, spatial_shapes, sampling_locations, attention_weights)
