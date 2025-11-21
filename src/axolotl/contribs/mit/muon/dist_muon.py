"""
Distributed muon optimizer for FSDP2.

Based on microsoft/dion muon implementation
https://github.com/microsoft/dion
"""

import math
import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from axolotl.integrations.base import BaseOptimizerFactory

from ..dion.opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from ..dion.scalar_opts import adamw_update_foreach, lion_update_foreach


class DistMuon(Optimizer):
    """
    distributed muon optimizer for pytorch fsdp2. also compatible with ddp.

    args:
        params: parameters for the optimizer.
        distributed_mesh: devicemesh or processgroup for distributed training.
            use devicemesh for fsdp2 (1d mesh only) and processgroup for distributeddataparallel.
        lr: base learning rate. for muon, this will be scaled based on the matrix dimensions.
            for element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: momentum factor for muon algorithm.
        betas: tuple of (beta1, beta2) for adamw and lion algorithms.
        weight_decay: weight decay factor.
        epsilon: small value to avoid division by zero.
        nesterov: whether to use nesterov momentum.
        adjust_lr: how to adjust the learning rate for muon updates ("spectral_norm" or "rms_norm" or none).
            "spectral_norm": adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": adjust based on rms norm, for learning rate compatibility with adam/adamw.
            none: do not adjust the learning rate.
        flatten: whether to flatten 3d+ tensors to 2d for muon updates.
            true: tensors with 3+ dimensions are flattened to 2d. use this for convolutional layers.
            false: tensors are not flattened. 3d+ tensors are treated as batches of 2d matrices.

    note: tensor parallelism is not currently supported. only 1d data parallel sharding is supported.
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"invalid adjust_lr value: {adjust_lr}. must be 'spectral_norm', 'rms_norm', or none."
            )

        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)

        # distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"only 1d devicemesh is supported, but got {distributed_mesh.ndim}d. for hsdp, provide the 1d sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"invalid distributed_mesh type: {type(distributed_mesh)}. expected devicemesh or processgroup."
            )
        self._distributed_mesh = distributed_mesh

        self._newton_schulz_func = zeropower_via_newtonschulz5

    @torch.no_grad()
    def step(self, closure=None):
        """perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1

            algo = group["algorithm"]
            if algo == "muon":
                muon_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"unknown algorithm: {algo}")

        muon_tasks = self._create_muon_tasks(muon_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(muon_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """get optimizer state for the given parameter tensor, or lazy-initialize it if it doesn't exist."""
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _create_muon_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "muon",
    ) -> Generator["AsyncTask", None, None]:
        """helper function to create batches of muon matrices and generate asynctask objects"""
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            muon_update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
            )

            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]

                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "must create optimizer with devicemesh if using dtensor parameters."
                        )

                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(
                            p.dim not in matrix_dims for _, p in shard_placements
                        )
                        shard_placements = [
                            (i, p) for i, p in shard_placements if p.dim in matrix_dims
                        ]

                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "muon does not support parameters with multiple sharded dimensions."
                        )

                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"got dtensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh."
                        )

                if is_batch_sharded and not is_matrix_sharded:
                    for x, g, m in zip(params, gradients, momentums):
                        yield AsyncTask(
                            muon_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                shard_dim=None,
                                **muon_update_args,
                            )
                        )
                else:
                    yield AsyncTask(
                        muon_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            **muon_update_args,
                        )
                    )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """helper function to generate asynctask objects for lion updates."""
        for group in param_groups:
            assert group["algorithm"] == algo_name

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """helper function to generate asynctask objects for adamw updates."""
        for group in param_groups:
            assert group["algorithm"] == algo_name

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )


def muon_update_batch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    momentum: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """batched version of muon update."""

    assert len(X) == len(G)
    assert len(X) == len(M)

    U = muon_update_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

    if shard_dim is not None:
        assert len(X) == world_size, "batch size must equal world size"
        assert (
            process_group is not None
        ), "process_group must be provided for sharded dtensors"
        assert isinstance(X[0], DTensor), "x should contain dtensors"
        assert not isinstance(U[0], DTensor), "u should contain local shards"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), f"shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."

        single_matrix_shards = [torch.empty_like(u) for u in U]

        work = dist.all_to_all(
            single_matrix_shards, U, group=process_group, async_op=True
        )
        yield
        work.wait()

        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]

        work = dist.all_to_all(
            U, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()

    elif len(U) > 1:
        assert len(U) == world_size, "batch size must equal world size"
        assert process_group is not None

        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        U = [torch.empty_like(u) for u in U]

        work = dist.all_gather(
            U, single_matrix.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    else:
        assert len(U) == 1
        U[0] = muon_update_newton_schulz(
            U[0],
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"unknown adjust_lr value: {adjust_lr}")

    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def adamw_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    """async wrapper around foreach adamw update."""
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


def lion_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
) -> Generator[None, None, None]:
    """async wrapper around foreach lion update."""
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)
    yield


@torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    """update momentum with gradient and compute the input to orthogonalization."""
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


@torch.compile(fullgraph=True)
def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """apply weight decay and weight update after orthogonalization."""
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """flatten the input tensor if needed and call the newton-schulz function."""
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


def adjust_lr_rms_norm(lr, param_shape, flatten):
    """adjust learning rate for constant element-wise rms norm"""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_ratio = 0.2 * math.sqrt(max(fan_out, fan_in))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


def adjust_lr_spectral_norm(lr, param_shape, flatten):
    """adjust from spectral norm 1 to rms operator norm 1"""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr


@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """newton-schulz iteration to approximate the orthogonalization of x."""
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class DistMuonOptimizerFactory(BaseOptimizerFactory):
    """factory for creating distributed muon optimizer with axolotl integration"""

    optim_cls = DistMuon

    def __call__(self, opt_model, training_args, **optimizer_kwargs):
        """create muon optimizer with parameter grouping for axolotl"""

        device_mesh: DeviceMesh | None = optimizer_kwargs.pop("device_mesh", None)

        distributed_mesh = None
        if device_mesh is not None:
            if "dp_shard" in device_mesh.mesh_dim_names:
                distributed_mesh = device_mesh["dp_shard"]
            elif device_mesh.ndim == 1:
                distributed_mesh = device_mesh

        lr = optimizer_kwargs.pop("lr")
        weight_decay = optimizer_kwargs.pop("weight_decay", 0.0)

        decay_parameters: list[str] = self.get_decay_parameter_names(opt_model)

        muon_params = {
            "to_weight_decay": {},
            "no_weight_decay": {},
        }
        adamw_params = {
            "to_weight_decay": {},
            "embeddings": {},
            "no_weight_decay": {},
        }

        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue

            if name.endswith("modules_to_save.default.weight") or any(
                embed_name in name for embed_name in ["embed_tokens", "lm_head"]
            ):
                adamw_params["embeddings"][name] = param
                continue

            if param.ndim < 2:
                if name in decay_parameters:
                    adamw_params["to_weight_decay"][name] = param
                else:
                    adamw_params["no_weight_decay"][name] = param
            else:
                if name in decay_parameters:
                    muon_params["to_weight_decay"][name] = param
                else:
                    muon_params["no_weight_decay"][name] = param

        optimizer_grouped_parameters = []

        if muon_params["to_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": list(muon_params["to_weight_decay"].values()),
                    "algorithm": "muon",
                    "weight_decay": weight_decay,
                }
            )
        if muon_params["no_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": list(muon_params["no_weight_decay"].values()),
                    "algorithm": "muon",
                    "weight_decay": 0.0,
                }
            )

        if adamw_params["to_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": list(adamw_params["to_weight_decay"].values()),
                    "algorithm": "adamw",
                    "weight_decay": weight_decay,
                }
            )
        if adamw_params["no_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": list(adamw_params["no_weight_decay"].values()),
                    "algorithm": "adamw",
                    "weight_decay": 0.0,
                }
            )
        if adamw_params["embeddings"]:
            optimizer_grouped_parameters.append(
                {
                    "params": list(adamw_params["embeddings"].values()),
                    "algorithm": "adamw",
                    "weight_decay": 0.0,
                }
            )

        return self.optim_cls(
            optimizer_grouped_parameters,
            distributed_mesh=distributed_mesh,
            lr=lr,
            **optimizer_kwargs,
        )
