# configuration_fetal_plane.py
from transformers import PretrainedConfig

class FetalPlaneConfig(PretrainedConfig):
    model_type = "fetal_plane_cnn"

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        image_resolution: int = 128,
        dropout_conv: float = 0.2,
        dropout_fc: float = 0.5,
        hidden_dim: int = 256,
        id2label: dict = None,
        label2id: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_resolution = image_resolution
        self.dropout_conv = dropout_conv
        self.dropout_fc = dropout_fc
        self.hidden_dim = hidden_dim

        # Default label mappings if not provided
        if id2label is None:
            id2label = {
                0: "Fetal abdomen",
                1: "Fetal brain",
                2: "Fetal femur",
                3: "Fetal thorax",
                4: "Maternal cervix",
                5: "Other",
            }
        if label2id is None:
            label2id = {v: int(k) for k, v in id2label.items()}

        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label2id = label2id