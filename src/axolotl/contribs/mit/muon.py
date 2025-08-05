"""
Muon optimizer extending the torchao low bit optimizer to support FSDP2, TP, and torch.compile.
"""

# adapted from the following sources:
# - https://github.com/samsja/muon_fsdp_2/blob/main/src/muon_fsdp2/muon_fsdp2/__init__.py
# - https://gist.github.com/rka97/c5266b71569bc0b03bbc27159fe7e1d8

from typing import Optional, List, Union, Tuple
import torch
from torch import Tensor
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed._tensor.placement_types import Placement
from torch.ops import aten
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from torchao.optim.adam import single_param_adam
from torchao.optim.quant_utils import _fp32_to_bf16_sr
from torchao.optim.subclass_8bit import OptimState8bit

from axolotl.integrations.base import BaseOptimizerFactory


@OptimState8bit.implements(aten.lerp_.Scalar)
def _(func, types, args, kwargs):
    """Handle in-place tensor.lerp_(end, weight)"""
    dst = args[0]  # The tensor being modified in-place
    end = args[1]  # The end value for interpolation
    weight = args[2]  # The interpolation weight

    # Dequantize, perform lerp operation, and copy back (re-quantizing)
    dst_f32 = dst.dequantize()
    end_f32 = end.dequantize() if isinstance(end, OptimState8bit) else end

    # Perform lerp: dst = dst + weight * (end - dst) = (1-weight)*dst + weight*end
    result = dst_f32.lerp(end_f32, weight)
    dst.copy_(result)
    return dst

@OptimState8bit.implements(aten.mul_.Tensor)
def _(func, types, args, kwargs):
    """Handle in-place tensor *= tensor"""
    dst = args[0]
    src = args[1]

    # Dequantize, perform operation, and copy back (re-quantizing)
    result = dst.dequantize() * (src.dequantize() if isinstance(src, OptimState8bit) else src)
    dst.copy_(result)
    return dst

# ==============================================================================
# Muon Utilities (FSDP/TP helpers and Newton-Schulz)
# ==============================================================================

# Sharding Utilities
def fsdp_unshard_tensor(x: Union[DTensor, Tensor]) -> Union[DTensor, Tensor]:
    """Remove FSDP dim-0 sharding (AllGather) while preserving TP sharding."""
    if not isinstance(x, DTensor):
        return x

    # Assuming mesh structure: [DP, TP]. We want to Replicate on the DP dimension.
    # Check if the first placement (DP/FSDP dimension) is sharded
    if len(x.placements) > 0 and x.placements[0].is_shard():
        # Change DP shard to Replicate(), keep TP placements
        non_dp = x.placements[1:]
        # Redistribute performs the AllGather
        return x.redistribute(placements=(Replicate(), *non_dp))
    return x

def fsdp_reshard_tensor(x: Union[DTensor, Tensor], orig_placements: Tuple[Placement, ...]) -> Union[DTensor, Tensor]:
    """Restore FSDP dim-0 sharding (Scatter/Slice)."""
    if not isinstance(x, DTensor):
        return x

    if x.placements != orig_placements:
        # Redistribute performs the Scatter/Slice
        return x.redistribute(placements=orig_placements)
    return x

@torch.compile(fullgraph=True)
def nsloop_torch(X: torch.Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    """
    When compiled down, inductor produces the following steps:
    1. A = matmul X with reinterpret_tensor(X)
    2. (triton) read A -> write b*A and c*A
    3. B = addmm(b*A, c*A, A)
    4. (triton) read X -> write a*X (this is stupid)
    5. X = addmm(a*X, B, X)
    """
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

# @torch.compile(fullgraph=True)
def zeropower_via_newtonschulz(G: Union[DTensor, Tensor], steps: int = 5, eps=1e-7) -> Tensor:
    """
    Newton-Schulz iteration for orthogonalization (zeroth power).
    Works with DTensor; matmuls automatically handle TP communication.
    """
    assert G.ndim >= 2

    # Use bfloat16 for computation stability and speed
    X = G.bfloat16()

    # Handle tall matrices (M > N)
    is_tall = G.size(-2) > G.size(-1)
    if is_tall:
        X = X.mT

    # Normalize spectral norm to ≤ 1 to ensure convergence
    # X.norm involves distributed communication if TP is active.
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norm + eps)

    # Newton-Schulz iterations (Coupled iteration optimized by Inductor)
    X = nsloop_torch(X, steps)

    return X.mT if is_tall else X

# ==============================================================================
# Muon Core Implementation
# ==============================================================================

def single_param_muon(
        p: DTensor,
        grad: DTensor,
        m: Tensor, # Sharded momentum buffer (can be quantized or FP32/BF16)
        lr: Tensor,
        momentum: float,
        wd: float,
        ns_steps: int,
        USE_EMA: bool,
        BF16_STOCHASTIC_ROUND: bool,
        # Placements are passed for DTensor resharding. Dynamo treats tuples as static arguments.
        original_placements: Tuple[Placement, ...],
):
    """
    Functional implementation of the Muon update step, compatible with torch.compile,
    FSDP2, TP, and quantized optimizer states.
    """

    # Sharded Momentum Update (Element-wise operations)
    m_f32 = m.float() # Dequantize sharded m if necessary
    grad_f32 = grad.float() # Sharded grad

    if USE_EMA:
        # Exponential Moving Average (like Adam)
        m_f32.lerp_(grad_f32, 1 - momentum)
    else:
        # Standard momentum (Original Muon paper)
        m_f32.mul_(momentum).add_(grad_f32)

    m.copy_(m_f32) # Quantize back sharded m

    # Nesterov Momentum (Sharded)
    # Calculate the input for the NS iteration.
    if USE_EMA:
        # Nesterov approximation commonly used with EMA (as in Snippet 1)
        g_ns_input_sharded = grad_f32.lerp(m_f32, momentum)
    else:
        # Standard Nesterov approximation
        g_ns_input_sharded = grad_f32.add(m_f32, alpha=momentum)

    # Unshard for NS
    # AllGather on the FSDP dimension. Preserves TP sharding.
    g_ns_input_unsharded = fsdp_unshard_tensor(g_ns_input_sharded)

    # Handle ND tensors (e.g., Convolutions) by flattening dimensions 1 to N
    original_shape = g_ns_input_unsharded.shape
    if g_ns_input_unsharded.ndim > 2:
        g_ns_input_unsharded = g_ns_input_unsharded.flatten(1)

    # Newton-Schulz Orthogonalization
    # Operates on the unsharded (potentially TP-sharded) DTensor.
    update_unsharded = zeropower_via_newtonschulz(g_ns_input_unsharded, steps=ns_steps)

    # Reshape back if necessary
    if update_unsharded.shape != original_shape:
        update_unsharded = update_unsharded.view(original_shape)

    # Reshard the update
    # Scatter/Slice on the FSDP dimension.
    update_sharded = fsdp_reshard_tensor(update_unsharded, original_placements)

    # Final Update
    p_f32 = p.float()

    # Apply weight decay (decoupled style, AdamW style)
    p_f32.mul_(1 - lr * wd)

    # Apply update with scaling factor
    # Scaling factor stabilizes training for rectangular matrices.
    # We use the global shape (p.size) for the aspect ratio.
    H, W = p.size(-2), p.size(-1)
    lr_scale = max(1.0, float(H) / float(W)) ** 0.5

    p_f32.add_(update_sharded.float(), alpha=-lr * lr_scale)

    # 7. Update parameter (with stochastic rounding if needed)
    if BF16_STOCHASTIC_ROUND:
        p.copy_(_fp32_to_bf16_sr(p_f32))
    else:
        p.copy_(p_f32.to(p.dtype))


class _MuonBase(Optimizer):
    """Base class for Muon optimizer following the torchao design pattern."""
    def __init__(
            self,
            params: ParamsT,
            adamw_params: Optional[Union[List[Tensor], Tensor]] = None,
            # Default Muon args
            lr: float = 0.02,
            betas: Tuple[float, float] = (0.9, 0.95),
            momentum: float | None = None,
            ns_steps: int = 5,
            weight_decay: float = 0.01,
            use_ema: bool = True,
            # Default AdamW args
            adam_lr: float | None = None,
            eps: float = 1e-8,
            adam_wd: float | None = None,
            # Quantization args
            block_size: int = 256,
            bf16_stochastic_round: bool = False,
    ):
        adam_lr = lr if adam_lr is None else adam_lr
        adam_wd = weight_decay if adam_wd is None else adam_wd
        momentum = betas[1] if momentum is None else momentum
        groups = self._structure_groups(params, adamw_params)

        self._set_defaults(groups, lr, momentum, ns_steps, weight_decay, use_ema,
                           adam_lr, betas, eps, adam_wd)

        # Initialize the optimizer
        super().__init__(groups, {})

        self.block_size = block_size
        self.bf16_stochastic_round = bf16_stochastic_round

        # Convert LRs to tensors eagerly for torch.compile compatibility
        for group in self.param_groups:
            self._convert_lr_to_tensor(group)

    def _structure_groups(self, params, adamw_params):
        """Organizes parameters into the standard optimizer group structure."""

        def normalize_params(p_input, default_muon=True):
            if isinstance(p_input, torch.Tensor):
                p_input = [p_input]

            # Handle generators and lists
            try:
                p_list = list(p_input)
            except TypeError:
                raise ValueError(f"Invalid parameter format provided: {type(p_input)}")

            if not p_list:
                return []

            if all(isinstance(p, torch.Tensor) for p in p_list):
                return [{'params': p_list, 'use_muon': default_muon}]
            if all(isinstance(p, dict) for p in p_list):
                return p_list

            raise ValueError("Invalid or mixed parameter format provided.")

        groups = normalize_params(params, default_muon=True)

        if adamw_params:
            adam_groups = normalize_params(adamw_params, default_muon=False)
            groups.extend(adam_groups)

        return groups

    def _set_defaults(self, groups, lr, momentum, ns_steps, wd, use_ema,
                      adam_lr, adam_betas, adam_eps, adam_wd):
        """Sets the default hyperparameters for Muon and AdamW groups."""
        for group in groups:
            # Ensure use_muon is set if not provided
            group.setdefault("use_muon", True)

            if group["use_muon"]:
                # Muon defaults
                if group.get("lr") is None: group["lr"] = lr
                group.setdefault("momentum", momentum)
                group.setdefault("ns_steps", ns_steps)
                group.setdefault("wd", wd)
                group.setdefault("use_ema", use_ema)
            else:
                # AdamW defaults
                if group.get("lr") is None: group["lr"] = adam_lr
                group.setdefault("betas", adam_betas)
                group.setdefault("eps", adam_eps)
                group.setdefault("wd", adam_wd)
                # AdamW specific flags needed by single_param_adam
                group.setdefault("amsgrad", False)
                group.setdefault("is_adamw", True)

    def _convert_lr_to_tensor(self, group):
        """Ensures the learning rate is a tensor."""
        if not isinstance(group.get("lr"), Tensor):
            # Determine device. If params exist, use the first param's device as a hint.
            device = group["params"][0].device if group.get("params") else None
            group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device=device)

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        # Ensure the newly added group has LR converted
        self._convert_lr_to_tensor(self.param_groups[-1])

    # Methods for quantization (to be overridden by subclasses)
    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        # Default to full precision if not overridden
        return torch.zeros_like(p)

    def _new_buffer(self, p: Tensor, signed: bool):
        """Initializes a new optimizer state buffer, potentially quantized and distributed."""
        # Logic adapted from _AdamBase._new_buffer (Snippet 4)

        # Get the local component if it's a DTensor
        local_p = p.to_local()

        # Handle infinite/zero block size (FP32 mode)
        if self.block_size == float('inf') or self.block_size == 0:
            out = torch.zeros_like(local_p)
        else:
            # Check if the subclass provides a quantized type
            default_zeros = torch.zeros_like(local_p)
            # We call the static method on the instance's class (e.g., Muon8bit._subclass_zeros)
            quantized_zeros = self._subclass_zeros(local_p, signed, self.block_size)

            # Check if the returned type is different from a standard tensor
            is_quantized_type = (type(quantized_zeros) != type(default_zeros))

            # Heuristic from bitsandbytes/torchao: Quantize large tensors divisible by block_size
            should_quantize = (is_quantized_type) and \
                              (local_p.numel() >= 4096) and \
                              (self.block_size > 0 and local_p.numel() % self.block_size == 0)

            if should_quantize:
                out = quantized_zeros
            else:
                out = default_zeros

        # Wrap subclass/local tensor in DTensor if the parameter is a DTensor
        if isinstance(p, DTensor) and hasattr(DTensor, 'from_local'):
            out = DTensor.from_local(
                local_tensor=out,
                device_mesh=p.device_mesh,
                placements=p.placements,
                run_check=False,
                shape=p.shape,
                stride=p.stride(),
            )

        # Handle potential CPU offload mismatch (ensure buffer device matches param device)
        if hasattr(out, 'device') and hasattr(p, 'device') and out.device != p.device:
            out = out.to(p.device)
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Optimization recommended by torchao for compiled optimizers
        try:
            # Disable dynamo cache limit to avoid recompilations for different shapes/groups
            dynamo_ctx = torch._dynamo.utils.disable_cache_limit()
        except (AttributeError, ImportError):
            # Fallback context manager if dynamo is not available
            class NoOpContext:
                def __enter__(self): pass
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            dynamo_ctx = NoOpContext()

        # Define the compilation wrapper
        try:
            # Use dynamic=False as optimizer shapes are generally static
            compile_fn = torch.compile(fullgraph=True, dynamic=False)
        except Exception:
            compile_fn = lambda f: f # Fallback if compile is unavailable

        with dynamo_ctx:
            for group in self.param_groups:
                use_muon = group["use_muon"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    if p.grad.is_sparse:
                        raise RuntimeError("Sparse gradient is not supported")

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        if use_muon:
                            if p.ndim < 2:
                                raise ValueError(f"Muon requires 2D+ params, got {p.ndim}D. Use AdamW for this parameter (use_muon=False).")
                            # Muon only needs one momentum buffer 'm' (signed)
                            state["m"] = self._new_buffer(p, signed=True)
                        else:
                            # AdamW state initialization
                            state["step"] = torch.tensor(0.0, device=p.device)
                            # exp_avg (m) is signed, exp_avg_sq (v) is unsigned (>=0)
                            state["exp_avg"] = self._new_buffer(p, signed=True)
                            state["exp_avg_sq"] = self._new_buffer(p, signed=False)
                            if group.get("amsgrad", False):
                                state["max_exp_avg_sq"] = self._new_buffer(p, signed=False)

                    # Ensure LR tensor is on the correct device before calling compiled function
                    lr_tensor = group["lr"].to(p.device)

                    use_stochastic_rounding = self.bf16_stochastic_round and p.dtype == torch.bfloat16

                    # Dispatch to the correct update function
                    # p.detach() is recommended for FSDP2 compatibility with torch.compile
                    if use_muon:
                        # Muon Update
                        # Get placements if it's a DTensor. This is passed as a static arg to the compiled function.
                        original_placements = p.placements if isinstance(p, DTensor) else ()

                        torch.compile(single_param_muon, fullgraph=True, dynamic=False)(
                            p.detach(),
                            p.grad,
                            state["m"],
                            lr_tensor,
                            group["momentum"],
                            group["wd"],
                            group["ns_steps"],
                            group["use_ema"],
                            use_stochastic_rounding,
                            original_placements,
                        )
                    else:
                        # AdamW Update
                        state["step"] += 1

                        torch.compile(single_param_adam, fullgraph=True, dynamic=False)(
                            p.detach(),
                            p.grad,
                            state["step"],
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            state.get("max_exp_avg_sq", None),
                            lr_tensor,
                            group["betas"][0],
                            group["betas"][1],
                            group["wd"],
                            group["eps"],
                            group["is_adamw"],
                            use_stochastic_rounding,
                        )
        return loss

# ==============================================================================
# Concrete Implementations
# ==============================================================================

class Muon8bit(_MuonBase):
    """
    8-bit Muon optimizer (Hybrid with AdamW8bit).

    Uses 8-bit block-wise quantization for the momentum buffer(s), reducing memory footprint.
    Relies on OptimState8bit
    """
    def __init__(self, params, **kwargs):
        # Set default block_size for 8bit if not provided
        kwargs.setdefault("block_size", 256)
        super().__init__(params, **kwargs)

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        # Use OptimState8bit for quantization
        return OptimState8bit.zeros(p.shape, signed, block_size, p.device)


class Muon(_MuonBase):
    """
    Full precision Muon optimizer (Hybrid with AdamW).
    Supports FSDP2, TP, and torch.compile.
    """
    def __init__(self, params, **kwargs):
        # Set block_size to infinity to disable quantization logic in _new_buffer
        kwargs["block_size"] = float('inf')
        super().__init__(params, **kwargs)


# ==============================================================================
# Integrate with Axolotl
# ==============================================================================


class MuonOptimizerFactory(BaseOptimizerFactory):
    optim_cls = Muon

    def __call__(self, opt_model, training_args, **optimizer_kwargs) -> "_MuonBase":
        muon_params = []
        adamw_params = []

        for name, param in opt_model.named_parameters():
            if not param.requires_grad or param.ndim < 2:
                continue
            if name.endswith("modules_to_save.default.weight") or any(
                    embed_name in name for embed_name in ["embed_tokens", "lm_head"]
            ):
                adamw_params.append(param)
            else:
                muon_params.append(param)

        return self.optim_cls(
            muon_params,
            adamw_params=adamw_params,
            **optimizer_kwargs,
        )


class Muon8bitOptimizerFactory(BaseOptimizerFactory):
    optim_cls = Muon8bit
