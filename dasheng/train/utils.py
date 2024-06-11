from __future__ import annotations
from pathlib import Path
import uuid
from typing import Dict, List, Optional, Tuple, Any, Type

import yaml
import datetime
import torch
from dataclasses import dataclass, field, asdict


@dataclass
class MAEConfig:
    train_data: List[str]
    cv_data: List[str]
    config_file: str = ''  # Will be overwritten during parsing
    logfile: str = 'train.log'
    outputpath: str = 'experiments'
    #Train args
    mask_ratio: float = 0.75
    use_scheduler: bool = True
    warmup_iters: Optional[int] = None
    warmup_epochs: Optional[int] = None
    model: str = 'dasheng_base'
    model_args: Dict[str, Any] = field(default_factory=lambda: dict())
    decay_frac: float = 0.01  # Decay fraction  of learning rate

    optimizer: str = 'AdamW8bit'
    optimizer_args: Dict[str,
                         Any] = field(default_factory=lambda: dict(lr=0.0003, weight_decay=0.01))
    epochs: int = 100
    epoch_length: int = 15000
    # Dataloader args
    batch_size: int = 32
    n_saved: int = 4  # Num models saved
    num_workers: int = 4
    resampled: bool = True
    crop_shuffle: int = 512
    chunk_length: float = 10.0  # Sample length during training/testing
    sample_rate: int = 16000  # Sampling rate of audio
    valid_every: int = 1  # When to run validation
    score_function: Tuple[str, float] = ('Loss', -1.0) # Save best loss on CV
    average_final_model: bool = True 
    outputdir: Path = field(init=False)

    def __post_init__(self):
        self.outputdir = Path(self.outputpath) / Path(
            self.config_file
        ).stem / self.model / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{uuid.uuid1().hex}"

    def to_dict(self):
        return asdict(self)

    def state_dict(self):
        return self.to_dict()

    @classmethod
    def load_state_dict(cls, state):
        return cls(**state)

    @classmethod
    def from_config_file(cls: Type[MAEConfig], config_file: str,
                         **kwargs) -> MAEConfig:
        """parse_config_or_kwargs

        :param config_file: Config file that has parameters, yaml format
        :param **kwargs: Other alternative parameters or overwrites for config
        """
        with open(config_file) as con_read:
            yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
        # values from config file are all possible params
        return cls(**dict(yaml_config, config_file=config_file, **kwargs))


def average_models(models: List[str]):
    model_res_state_dict = {}
    state_dict = {}
    has_new_structure = False
    for m in models:
        cur_state = torch.load(m, map_location='cpu')
        if 'model' in cur_state:
            has_new_structure = True
            model_params = cur_state.pop('model')
            # Append non "model" items, encoder, optimizer etc ...
            for k in cur_state:
                state_dict[k] = cur_state[k]
            # Accumulate statistics
            for k in model_params:
                if k in model_res_state_dict:
                    model_res_state_dict[k] += model_params[k]
                else:
                    model_res_state_dict[k] = model_params[k]
        else:
            for k in cur_state:
                if k in model_res_state_dict:
                    model_res_state_dict[k] += cur_state[k]
                else:
                    model_res_state_dict[k] = cur_state[k]

    # Average
    for k in model_res_state_dict:
        # If there are any parameters
        if model_res_state_dict[k].ndim > 0:
            model_res_state_dict[k] /= float(len(models))
    if has_new_structure:
        state_dict['model'] = model_res_state_dict
    else:
        state_dict = model_res_state_dict
    return state_dict
