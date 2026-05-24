import importlib
import json
from pathlib import Path

import torch

# Maps policy type (from config.json) to (module_path, class_name)
_POLICY_REGISTRY: dict[str, tuple[str, str]] = {
    "act": ("lerobot.policies.act.modeling_act", "ACTPolicy"),
    "diffusion": ("lerobot.policies.diffusion.modeling_diffusion", "DiffusionPolicy"),
    "pi0": ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
    "pi0fast": ("lerobot.policies.pi0fast.modeling_pi0fast", "PI0FastPolicy"),
    "tdmpc": ("lerobot.policies.tdmpc.modeling_tdmpc", "TDMPCPolicy"),
    "vqbet": ("lerobot.policies.vqbet.modeling_vqbet", "VQBeTPolicy"),
}


def _flatten(t: torch.Tensor) -> torch.Tensor:
    """Reduce any tensor to [N, D] via global average pooling."""
    if t.dim() <= 2:
        return t
    if t.dim() == 3:
        return t.mean(dim=1)  # [B, seq, D] -> [B, D]
    return t.mean(dim=list(range(2, t.dim())))  # [B, C, H, W] -> [B, C]


class PolicyEmbeddingExtractor:
    """Extract per-camera embeddings from a LeRobot policy via a forward hook.

    Args:
        policy_dir: Path to directory containing config.json + weights.
        hook_module: Dotted attribute path to the module to hook.
            ACT             -> "model.backbone"
            DiffusionPolicy -> "model.obs_encoder"
            Pi0             -> "model.paligemma_with_expert.paligemma.vision_tower"
        device: "auto", "cpu", "cuda", or "cuda:N".
    """

    def __init__(self, policy_dir: str, hook_module: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.policy = self._load_policy(Path(policy_dir))
        self.policy.eval().to(self.device)
        self._hook_outputs: list[torch.Tensor] = []
        self._hook_handle = self._register_hook(hook_module)

    def _load_policy(self, policy_dir: Path):
        config = json.loads((policy_dir / "config.json").read_text())
        raw_type = config.get("type") or config.get("model_type") or ""
        policy_type = raw_type.lower().replace("config", "").strip("_-")
        if policy_type not in _POLICY_REGISTRY:
            raise ValueError(f"Unsupported policy type '{policy_type}'. Supported: {list(_POLICY_REGISTRY)}")
        module_path, class_name = _POLICY_REGISTRY[policy_type]
        return getattr(importlib.import_module(module_path), class_name).from_pretrained(str(policy_dir))

    def _register_hook(self, module_path: str):
        module = self.policy
        for attr in module_path.split("."):
            module = getattr(module, attr)

        def _hook(mod, inp, output):
            if isinstance(output, torch.Tensor):
                self._hook_outputs.append(output.detach().cpu())
            elif isinstance(output, dict):
                # IntermediateLayerGetter returns OrderedDict {layer_name: tensor}
                vals = [v for v in output.values() if isinstance(v, torch.Tensor)]
                if vals:
                    self._hook_outputs.append(vals[-1].detach().cpu())
            elif isinstance(output, (tuple, list)) and output:
                if isinstance(output[0], torch.Tensor):
                    self._hook_outputs.append(output[0].detach().cpu())

        return module.register_forward_hook(_hook)

    def extract_per_camera(
        self, observation: dict[str, torch.Tensor], camera_keys: list[str]
    ) -> dict[str, torch.Tensor]:
        """Run one forward pass and return {camera_key: embedding [D]}."""
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in observation.items() if isinstance(v, torch.Tensor)}

        self._hook_outputs = []
        with torch.no_grad():
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            self.policy.select_action(batch)

        outputs = [_flatten(o) for o in self._hook_outputs]
        n_cams = len(camera_keys)

        if len(outputs) == n_cams:
            return {k: outputs[i].squeeze(0) for i, k in enumerate(camera_keys)}

        if len(outputs) == 1:
            flat = outputs[0]
            if flat.shape[0] == n_cams:
                return {k: flat[i] for i, k in enumerate(camera_keys)}
            emb = flat.squeeze(0)
            return {k: emb for k in camera_keys}

        raise RuntimeError(
            f"Hook fired {len(outputs)} time(s) but expected {n_cams} camera(s). "
            "Try a different hook_module path."
        )

    def __del__(self):
        if hasattr(self, "_hook_handle") and self._hook_handle:
            self._hook_handle.remove()
