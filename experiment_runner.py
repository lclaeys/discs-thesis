"""Main script for sampling based experiments."""
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from absl import app
from absl import flags
from discs.common import configs as common_configs
from discs.common import utils
import discs.common.experiment_saver as saver_mod
from ml_collections import config_flags
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Handle any runtime initialization errors
# EXPERIMENTS 
batch_size = 1
temps = np.logspace(-1,0,10)
#temps = [0.22]
chain_length= 600000
final_temps = [0.05]
experiments = [
  #  {'experiment_name': f'replica exchange (20 replicas)',
  #                'experiment': {'t_schedule': 'constant',
  #                                'name': 'RE_CO_Experiment',
  #                                'init_temperature': 1,
  #                                'decay_rate': .1,
  #                                'minimum_temperature': .05,
  #                                'maximum_temperature': 1,
  #                                'num_replicas': 20,
  #                                'chain_length': chain_length, 
  #                                'batch_size': batch_size,
  #                                'save_replica_data': True,
  #                                'save_every_steps': chain_length // 1000,
  #                                'log_every_steps': chain_length // 1000,
  #                                'adaptive_temps':True
  #                               },
  #                 'sampler': {'adaptive': True}}
                  ] + [
                     {'experiment_name': f'exp decaying temperature (T={final_temp})',
                'experiment': {'t_schedule': 'exp_decay',
                               'name': 'CO_Experiment',
                               'init_temperature': 1,
                               'chain_length': chain_length,
                               'batch_size': batch_size,
                               'decay_rate': final_temp,
                               'save_every_steps': chain_length // 1000,
                               'log_every_steps': chain_length // 1000
                               },
                'sampler': {'adaptive': True}} for final_temp in final_temps
                ] + [
                #    {
                #   'experiment_name': f'constant temperature (T={temp:.2f})',
                # 'experiment': {'t_schedule': 'constant',
                #                'name': 'CO_Experiment',
                #                'init_temperature': temp,
                #                'chain_length': chain_length,
                #                'batch_size': batch_size,
                #                'save_every_steps': chain_length// 10000,
                #                'log_every_steps': chain_length // 10000
                #                },
                # 'sampler': {'adaptive': True}} for temp in temps
                ]

# CONFIG
model_name = 'mis'
sampler_name = 'path_auxiliary'
graph_type = 'ertest'
experiment_type = 'full_re_10k10'
experiment_folder = f'{sampler_name}_{graph_type}_{experiment_type}' 

experiment_config = importlib.import_module(
        f'discs.experiment.configs.{model_name}.{graph_type}'
    )
experiment_config = experiment_config.get_config()

model_config = importlib.import_module(
        f'discs.models.configs.{model_name}_config'
    )

model_config = model_config.get_config()

sampler_config = importlib.import_module(
        f'discs.samplers.configs.{sampler_name}_config'
    )

sampler_config = sampler_config.get_config()

def update_save_dir(config):
  experiment_name = config.get('experiment_name','unnamed_experiment')
  save_folder = config.model.get('save_dir_name', config.model.name)
  save_root = './discs/results/' + save_folder +  f'/{experiment_folder}/{experiment_name}'
  config.experiment.save_root = save_root


def get_main_config():
  """Merge experiment, model and sampler config."""
  config = common_configs.get_config()
  if (
      'graph_type' not in model_config
      and 'bert_model' not in model_config
  ):
    config.update(experiment_config)
  config.sampler.update(sampler_config)
  config.model.update(model_config)
  if config.model.get('graph_type', None):
    graph_config = importlib.import_module(
        'discs.models.configs.%s.%s'
        % (config.model['name'], config.model['graph_type'])
    )
    config.model.update(graph_config.get_model_config(config.model['cfg_str']))
    co_exp_default_config = importlib.import_module(
        'discs.experiment.configs.co_experiment'
    )
    config.experiment.update(co_exp_default_config.get_co_default_config())
    config.update(experiment_config)
    config.experiment.num_models = config.model.num_models

  if config.model.get('bert_model', None):
    config.update(experiment_config)

  return config


def main(_):
  for experiment in experiments:
    config = get_main_config()
    config.update(experiment)
    update_save_dir(config)
    utils.setup_logging(config)

    # model
    model_mod = importlib.import_module('discs.models.%s' % config.model.name)
    model = model_mod.build_model(config)

    # sampler
    sampler_mod = importlib.import_module(
        'discs.samplers.%s' % config.sampler.name
    )
    sampler = sampler_mod.build_sampler(config)

    # experiment
    experiment_mod = getattr(
        importlib.import_module('discs.experiment.sampling'),
        f'{config.experiment.name}',
    )
    experiment = experiment_mod(config)

    # evaluator
    evaluator_mod = importlib.import_module(
        'discs.evaluators.%s' % config.experiment.evaluator
    )
    evaluator = evaluator_mod.build_evaluator(config)

    # saver
    saver = saver_mod.build_saver(config)

    # chain generation
    experiment.get_results(model, sampler, evaluator, saver)

    print(f'Finished experiment {config.get('experiment_name', '')}')


if __name__ == '__main__':
  app.run(main)