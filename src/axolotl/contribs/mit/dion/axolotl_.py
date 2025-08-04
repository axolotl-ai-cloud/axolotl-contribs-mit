# ==============================================================================
# Integrate with Axolotl
# ==============================================================================

from axolotl.integrations.base import BaseOptimizerFactory
from torch.distributed.device_mesh import DeviceMesh

from .dion import Dion


class DionOptimizerFactory(BaseOptimizerFactory):
    optim_cls = Dion
    mp_8bit = False

    def __call__(self, opt_model, training_args, dion_lr, dion_mu, **optimizer_kwargs):

        device_mesh: DeviceMesh | None = optimizer_kwargs.pop("device_mesh", None)
        replicate_mesh = None
        outer_shard_mesh = None
        inner_shard_mesh = None
        if device_mesh is not None:
            if "dp_replicate" in device_mesh.mesh_dim_names:
                replicate_mesh = device_mesh["dp_replicate"]
            if "dp_shard" in device_mesh.mesh_dim_names:
                outer_shard_mesh = device_mesh["dp_shard"]
            if "tp_shard" in device_mesh.mesh_dim_names:
                inner_shard_mesh = device_mesh["tp"]

        weight_decay = optimizer_kwargs.get("weight_decay", None)
        dion_params = {
            "to_weight_decay": {},  # LayerNorm and bias
            "no_weight_decay": {},
        }
        adamw_params = {
            "to_weight_decay": {},  # LayerNorm and bias
            "embeddings": {},  # lm_head, embed_tokens,
            "no_weight_decay": {},
        }

        decay_parameters: list[str] = self.get_decay_parameter_names(opt_model)

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
                    dion_params["to_weight_decay"][name] = param
                else:
                    dion_params["no_weight_decay"][name] = param

        optimizer_grouped_parameters = []
        wd_kwargs = {} if weight_decay is None else {"weight_decay": weight_decay}
        if adamw_params["to_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "adamw",
                    "params": list(adamw_params["to_weight_decay"].values()),
                    "lr": optimizer_kwargs["lr"],
                    **wd_kwargs,
                }
            )
        if adamw_params["no_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "adamw",
                    "params": list(adamw_params["no_weight_decay"].values()),
                    "lr": optimizer_kwargs["lr"],
                    "weight_decay": 0.0,
                }
            )
        if adamw_params["embeddings"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "adamw",
                    "params": list(adamw_params["embeddings"].values()),
                    "lr": optimizer_kwargs["lr"],
                    "weight_decay": 0.0,
                }
            )
        if dion_params["to_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "dion",
                    "params": list(dion_params["to_weight_decay"].values()),
                    "lr": dion_lr,
                    "mu": dion_mu,
                    **wd_kwargs,
                }
            )
        if dion_params["no_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "dion",
                    "params": list(dion_params["no_weight_decay"].values()),
                    "lr": dion_lr,
                    "mu": dion_mu,
                    "weight_decay": 0.0,
                }
            )

        if self.mp_8bit:
            raise NotImplementedError(
                "8-bit optimizer is not supported yet"
            )

        return self.optim_cls(
            optimizer_grouped_parameters,
            replicate_mesh = replicate_mesh,
            outer_shard_mesh = outer_shard_mesh,
            inner_shard_mesh = inner_shard_mesh,
            **optimizer_kwargs,
        )


class Dion8bitOptimizerFactory(DionOptimizerFactory):
    mp_8bit = False
