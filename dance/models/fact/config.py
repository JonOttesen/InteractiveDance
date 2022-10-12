from dataclasses import dataclass
from typing import Union, List, Tuple

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1

@dataclass
class transformer:
    num_attention_heads: int = 10
    hidden_size: int = 800
    num_hidden_layers: int = 2
    intermediate_size: int = 3072

@dataclass
class fact_model:
    feature_name: str
    transformer: transformer
    feature_dim: int
    sequence_length: int


audio_config = fact_model(feature_name="audio", transformer=transformer(), feature_dim=35, sequence_length=240)
motion_config = fact_model(feature_name="motion", transformer=transformer(), feature_dim=225, sequence_length=120)
multi_model_config = fact_model(feature_name="multi_modal", transformer=transformer(num_attention_heads=10, hidden_size=800, num_hidden_layers=12), feature_dim=225, sequence_length=120)