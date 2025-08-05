# ==============================================================================
# Integrate with Axolotl
# ==============================================================================

from axolotl.integrations.base import BaseOptimizerFactory
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP

from .muon import Muon


class MuonOptimizerFactory(BaseOptimizerFactory):
    def __call__(self, opt_model, training_args, **optimizer_kwargs) -> "Muon":
        lr = optimizer_kwargs.pop("lr")
        wd = optimizer_kwargs.pop("weight_decay")

        adam_beta1 = optimizer_kwargs.pop("adam_beta1", 0.9)
        adam_beta2 = optimizer_kwargs.pop("adam_beta2", 0.95)
        adamw_betas = optimizer_kwargs.pop("betas", (adam_beta1, adam_beta2))

        adamw_eps = optimizer_kwargs.pop("adam_epsilon", 1e-8)

        # Muon-specific parameters
        nesterov = optimizer_kwargs.pop("nesterov", True)
        use_triton = optimizer_kwargs.pop("use_triton", True)
        adjust_lr = optimizer_kwargs.pop("adjust_lr", "spectral_norm")
        flatten = optimizer_kwargs.pop("flatten", False)
        mu = optimizer_kwargs.pop("mu", 0.95)

        # Get decay parameter names
        decay_parameters = self.get_decay_parameter_names(opt_model)

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

            # Check if this is an embedding or lm_head parameter
            if name.endswith("modules_to_save.default.weight") or any(
                embed_name in name for embed_name in ["embed_tokens", "lm_head"]
            ):
                adamw_params["embeddings"][name] = param
                continue

            # Parameters with ndim < 2 go to AdamW
            if param.ndim < 2:
                if name in decay_parameters:
                    adamw_params["to_weight_decay"][name] = param
                else:
                    adamw_params["no_weight_decay"][name] = param
            else:
                # Matrix parameters go to Muon
                if name in decay_parameters:
                    muon_params["to_weight_decay"][name] = param
                else:
                    muon_params["no_weight_decay"][name] = param

        # Build parameter groups
        optimizer_grouped_parameters = []

        # AdamW groups
        if adamw_params["to_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "adamw",
                    "params": list(adamw_params["to_weight_decay"].values()),
                    "lr": lr,
                    "betas": adamw_betas,
                    "eps": adamw_eps,
                    "weight_decay": wd,
                }
            )

        if adamw_params["no_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "adamw",
                    "params": list(adamw_params["no_weight_decay"].values()),
                    "lr": lr,
                    "betas": adamw_betas,
                    "eps": adamw_eps,
                    "weight_decay": 0.0,
                }
            )

        if adamw_params["embeddings"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "adamw",
                    "params": list(adamw_params["embeddings"].values()),
                    "lr": lr,
                    "betas": adamw_betas,
                    "eps": adamw_eps,
                    "weight_decay": 0.0,
                }
            )

        # Muon groups
        if muon_params["to_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "muon",
                    "params": list(muon_params["to_weight_decay"].values()),
                    "weight_decay": wd,
                }
            )

        if muon_params["no_weight_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "algorithm": "muon",
                    "params": list(muon_params["no_weight_decay"].values()),
                    "weight_decay": 0.0,
                }
            )

        device_mesh: DeviceMesh | None = optimizer_kwargs.pop("device_mesh", None)
        distributed_mesh = None

        if device_mesh is not None:
            replicate_mesh = None
            outer_shard_mesh = None
            inner_shard_mesh = None
            if "dp_replicate" in device_mesh.mesh_dim_names:
                replicate_mesh = device_mesh["dp_replicate"]
            if "dp_shard" in device_mesh.mesh_dim_names:
                outer_shard_mesh = device_mesh["dp_shard"]
            if "tp" in device_mesh.mesh_dim_names:
                inner_shard_mesh = device_mesh["tp"]

            # Ensure that we have a supported device mesh configuration for Muon
            if inner_shard_mesh is not None and inner_shard_mesh.size() > 1:
                raise ValueError("Tensor parallel is not supported by Muon.")

            # Determine which mesh to use for distributed communication
            if outer_shard_mesh and outer_shard_mesh.size() > 1:
                distributed_mesh = outer_shard_mesh
            elif replicate_mesh and replicate_mesh.size() > 1:
                distributed_mesh = replicate_mesh

        elif isinstance(opt_model, DDP):
            distributed_mesh = opt_model.process_group  # using ProcessGroup for DDP

        return Muon(
            params=optimizer_grouped_parameters,
            distributed_mesh=distributed_mesh,
            lr=lr,
            mu=mu,
            betas=adamw_betas,
            epsilon=adamw_eps,
            weight_decay=wd,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            flatten=flatten,
            use_triton=use_triton,
            **optimizer_kwargs,
        )
