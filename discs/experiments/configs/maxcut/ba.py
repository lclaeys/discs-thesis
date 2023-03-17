"""Config for maxcut ba dataset."""

from discs.common import configs
from ml_collections import config_dict
from sco.experiments import default_configs


def get_config():
  """Get config."""
  exp_config = dict(
      num_models=1024,
      batch_size=32,
      t_schedule='exp_decay',
      chain_length=50000,
      log_every_steps=100,
      init_temperature=1,
      decay_rate=0.1,
      final_temperature=0.000001,
  )
  return config_dict.ConfigDict(exp_config)
