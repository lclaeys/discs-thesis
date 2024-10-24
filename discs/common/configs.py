"""Main Config Structure."""

from ml_collections import config_dict


def get_config():
  """Get common config sketch."""
  general_config = dict(
      model=dict(
          name='',
          data_root='sco',
      ),
      sampler=dict(
          name='',
      ),
      experiment=dict(
          name='Sampling_Experiment',
          evaluator='ess_eval',
          num_models=1,
          batch_size=100,
          chain_length=1000,
          ess_ratio=0.5,
          run_parallel=True,
          get_additional_metrics=False,
          shuffle_buffer_size=0,
          log_every_steps=1,
          plot_every_steps=1,
          save_root='./discs/results',
          fig_folder='',
          save_every_steps=20,
          save_samples=True,
          get_estimation_error=True,
          use_tqdm=False,
          co_opt_prob=False,
          window_size=10,
          window_stride=10
      ),
  )
  return config_dict.ConfigDict(general_config)
