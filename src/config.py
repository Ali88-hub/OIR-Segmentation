from dataclasses import dataclass


@dataclass
class Config:
    # Paths
    data_root: str = "data"
    manifest_path: str = "data/manifest_enriched.csv"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Data
    image_size: tuple[int, int] = (768, 768)
    mask_names: tuple[str, ...] = ("nv", "vo", "retina")
    binarize_threshold: float = 0.5

    # Model
    encoder_name: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"
    num_classes: int = 3
    decoder_attention: str = "scse"

    # Training
    batch_size: int = 2
    num_workers: int = 2
    epochs: int = 150
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 45
    use_amp: bool = True
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 4  # effective batch = batch_size * grad_accum_steps
    warmup_epochs: int = 5
    save_every: int = 10  # periodic checkpoint every N epochs

    # Loss weights per mask (order: nv, vo, retina)
    loss_weights: tuple[float, ...] = (3.0, 2.0, 1.0)
    bce_weight: float = 0.35
    focal_tversky_weight: float = 0.65
    # Per-class Tversky alpha/beta (order: nv, vo, retina)
    tversky_alpha: tuple[float, ...] = (
        0.35,
        0.3,
        0.5,
    )  # FP penalty per class (NV raised from 0.2 to reduce vessel FPs)
    tversky_beta: tuple[float, ...] = (0.65, 0.7, 0.5)  # FN penalty per class
    focal_gamma: float = 1.5  # focal exponent on Tversky (raised from 1.33)

    # Augmentation
    aug_prob: float = 0.5
    copy_paste_prob: float = 0.5  # probability of NV copy-paste per training sample

    # NV Post-processing
    nv_outside_px: int = 520  # VO boundary zone: how far outside VO NV can extend (px at 768x768)
    nv_inside_px: int = 260  # VO boundary zone: how far inside VO NV can extend (px at 768x768)
    nv_vessel_suppression: bool = True  # suppress NV predictions overlapping known vessel masks
    nv_min_component_area: int = 0  # disabled — even 50px removes genuine small NV foci
    nv_max_eccentricity: float = (
        1.0  # disabled — NV is biologically elongated, eccentricity filter kills recall
    )

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        assert len(self.mask_names) == self.num_classes, (
            f"mask_names length {len(self.mask_names)} != num_classes {self.num_classes}"
        )
        assert len(self.loss_weights) == self.num_classes, (
            f"loss_weights length {len(self.loss_weights)} != num_classes {self.num_classes}"
        )
        assert len(self.tversky_alpha) == self.num_classes, (
            f"tversky_alpha length {len(self.tversky_alpha)} != num_classes {self.num_classes}"
        )
        assert len(self.tversky_beta) == self.num_classes, (
            f"tversky_beta length {len(self.tversky_beta)} != num_classes {self.num_classes}"
        )
