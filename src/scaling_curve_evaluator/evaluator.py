import torch
from tqdm import tqdm

from .dataset import LeRobotDatasetLoader
from .embeddings import PolicyEmbeddingExtractor
from .similarity import policy_embedding_similarity


class PolicyEmbeddingSimilarityEvaluator:
    """Compute policy embedding similarity c̄_π between a train and eval dataset.

    Usage::

        evaluator = PolicyEmbeddingSimilarityEvaluator(
            policy_dir="policy/my_act_policy",
            train_data_dir="data/train",
            eval_data_dir="data/eval",
            hook_module="model.backbone",  # ACT example
        )
        score = evaluator.evaluate()
        print(f"c̄_π = {score:.4f}")

    Common hook_module values:
        ACT              -> "model.backbone"
        DiffusionPolicy  -> "model.obs_encoder"
        Pi0              -> "model.paligemma_with_expert.paligemma.vision_tower"
    """

    def __init__(
        self,
        policy_dir: str,
        train_data_dir: str,
        eval_data_dir: str,
        hook_module: str,
        device: str = "auto",
    ):
        self.extractor = PolicyEmbeddingExtractor(policy_dir, hook_module, device)
        self.train_loader = LeRobotDatasetLoader(train_data_dir)
        self.eval_loader = LeRobotDatasetLoader(eval_data_dir)

        self.camera_keys = sorted(
            set(self.train_loader.camera_keys) & set(self.eval_loader.camera_keys)
        )
        if not self.camera_keys:
            raise ValueError(
                "No common camera keys between train and eval datasets.\n"
                f"  Train cameras: {self.train_loader.camera_keys}\n"
                f"  Eval  cameras: {self.eval_loader.camera_keys}"
            )

    def _extract_embeddings(
        self, observations: list[dict], desc: str
    ) -> dict[str, torch.Tensor]:
        """Extract embeddings for all observations.

        Returns {camera_key: [N, D]}.
        """
        per_camera: dict[str, list[torch.Tensor]] = {k: [] for k in self.camera_keys}

        for obs in tqdm(observations, desc=desc, unit="ep"):
            embs = self.extractor.extract_per_camera(obs, self.camera_keys)
            for k in self.camera_keys:
                per_camera[k].append(embs[k])

        return {k: torch.stack(vs, dim=0) for k, vs in per_camera.items()}

    def evaluate(self) -> float:
        """Run the full evaluation and return c̄_π ∈ [0, 1]."""
        train_obs = self.train_loader.get_initial_observations()
        eval_obs = self.eval_loader.get_initial_observations()

        print(f"Train episodes: {len(train_obs)}, Eval episodes: {len(eval_obs)}")
        print(f"Cameras: {self.camera_keys}")

        train_embs = self._extract_embeddings(train_obs, "Train embeddings")
        eval_embs = self._extract_embeddings(eval_obs, "Eval embeddings")

        score = policy_embedding_similarity(train_embs, eval_embs)
        print(f"c̄_π = {score:.4f}")
        return score
