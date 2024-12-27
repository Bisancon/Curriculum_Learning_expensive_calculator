from dataclasses import dataclass, field
import yaml
import marshmallow.validate
from marshmallow_dataclass import class_schema

@dataclass()
class TrainParams:
    batch_size: int = field(default=1, metadata={"validate": marshmallow.validate.Range(min=1)})
    shuffle: bool = field(default=False, metadata={"validate": marshmallow.validate.OneOf(choices=[True, False])})
    learning_rate: float = field(default=0.001, metadata={"validate": marshmallow.validate.Range(min=0, min_inclusive=False)})
    num_epochs: int = field(default=1, metadata={"validate": marshmallow.validate.Range(min=1)})
    device: str = field(default='cpu')

@dataclass()
class ModelParams:
    num_tokens: int = field(default=1, metadata={"validate": marshmallow.validate.Range(min=1)})
    embedding_dim: int = field(default=1, metadata={"validate": marshmallow.validate.Range(min=1)})
    num_layers: int = field(default=1, metadata={"validate": marshmallow.validate.Range(min=1)})
    num_heads: int = field(default=1, metadata={"validate": marshmallow.validate.Range(min=1)})
    dropout: float = field(default=0.1, metadata={"validate": marshmallow.validate.Range(min=0)})

@dataclass()
class DataParams:
    train_split: float = field(default=0.1, metadata={"validate": marshmallow.validate.Range(min=0, min_inclusive=False, max=1, max_inclusive=False)})
    val_split: float = field(default=0.1, metadata={"validate": marshmallow.validate.Range(min=0, min_inclusive=False, max=1, max_inclusive=False)})
    random_seed: int
    dataset_path: str

@dataclass()
class ConfigParams:
    training: TrainParams
    model: ModelParams
    data: DataParams
    logging_dir: str

ConfigParamsSchema = class_schema(ConfigParams)

def read_config_params(path: str) -> ConfigParams:
    with open(path, "r") as input_stream:
        schema = ConfigParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
