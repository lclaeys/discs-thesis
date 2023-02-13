from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(30, 30),
      lambdaa=1,
      init_sigma=1.5,
      num_categories=4,
      external_field_type=1,
      mu=0.5,
      save_dir_name='potts_4'
      name='potts',
  )
  return config_dict.ConfigDict(model_config)
