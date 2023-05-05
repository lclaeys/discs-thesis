"""Config for rb job."""

from ml_collections import config_dict

def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='bernoulli',
          sampler='path_auxiliary',
          sweep=[
              {
                  'model_config.init_sigma': [0.5, 1.5],
                  'sampler_config.name': [
                      'randomwalk',
                  ],
              },
              {
                  'model_config.init_sigma': [0.5, 1.5],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'dlmc',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
          ],
      )
  )
  return config
