from typing import Optional
from vllm.config.vllm import VllmConfig
from vllm.v1.attention.dsa.config import DSAConfig, SelectStrategyType

"""
DSA config functions
"""

_global_dsa_config: Optional[DSAConfig] = None

def set_dsa_config(dsa_config: DSAConfig):
    global _global_dsa_config
    _global_dsa_config = dsa_config


def get_dsa_config() -> DSAConfig:
    global _global_dsa_config
    if _global_dsa_config is None:
        _global_dsa_config = DSAConfig()
    return _global_dsa_config


def reset_dsa_config():
    global _global_dsa_config
    _global_dsa_config = None


def create_dsa_config_from_args(args: VllmConfig) -> DSAConfig:
    # TODO: parsed args
    return DSAConfig(
        dsa_enabled=True,
        block_size=args.cache_config.block_size,
        strategy=SelectStrategyType.QUEST,
    )