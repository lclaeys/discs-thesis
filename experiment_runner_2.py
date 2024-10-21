"""Main script for sampling based experiments."""
import importlib
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from absl import app
from absl import flags
from discs.common import configs as common_configs
from discs.common import utils
import discs.common.experiment_saver as saver_mod
from ml_collections import config_flags

# EXPERIMENTS 

shape = (4,4)
chain_length = 1_000_000
init_sigma = 0.6

temps = np.logspace(-np.log10(2),1,10)
temp_mults = np.logspace(np.log10(2),1,10)
setups = [(t,mult) for t in temps for mult in temp_mults]

experiments = [
                 {'experiment_name': f'sampling_{t:.2f}_re10_{mult:.2f}x',
                  'sampler': {'adaptive': True},
                  'model': {'shape': shape,
                            'lambdaa': 1,
                            'mu': 0,
                            'init_sigma':0},
                  'experiment': {'name': 'RE_Sampling_Experiment',
                                 'chain_length': chain_length,
                                 'num_saved_samples': 10,
                                 'batch_size': 20,
                                 'save_every_steps':1000,
                                 'save_samples': False,
                                 'num_replicas': 10,
                                 'minimum_temperature':t,
                                 'maximum_temperature':t*mult,
                                 'save_replica_data': False}
                 }
              for t, mult in setups] + [
                 {'experiment_name': f'sampling_{t:.2f}',
                  'sampler': {'adaptive': True},
                  'model': {'shape': shape,
                            'lambdaa': 1,
                            'mu': 0,
                            'init_sigma':0,
                            'temperature':t},
                  'experiment': {'name': 'Sampling_Experiment',
                                 'chain_length': chain_length,
                                 'num_saved_samples': 1,
                                 'batch_size': 20,
                                 'save_every_steps':1000,
                                 'save_samples': False
                                 }
                 }
                for t in temps]

experiments = [{'experiment_name': f'sampling_re',
                  'sampler': {'adaptive': True},
                  'model': {'shape': shape,
                            'lambdaa': 1,
                            'mu': 0,
                            'init_sigma':init_sigma},
                  'experiment': {'name': 'RE_Sampling_Experiment',
                                 'chain_length': chain_length,
                                 'num_saved_samples': 10,
                                 'batch_size': 20,
                                 'save_every_steps':100,
                                 'save_samples': False,
                                 'num_replicas': 3,
                                 'minimum_temperature':1,
                                 'maximum_temperature':10,
                                 'save_replica_data': True,
                                 'adaptive_temps': True}
                 },
                 {'experiment_name': f'sampling',
                  'sampler': {'adaptive': True},
                  'model': {'shape': shape,
                            'lambdaa': 1,
                            'mu': 0,
                            'init_sigma':init_sigma,
                            'temperature':1},
                  'experiment': {'name': 'Sampling_Experiment',
                                 'chain_length': chain_length,
                                 'num_saved_samples': 1,
                                 'batch_size': 20,
                                 'save_every_steps':1000,
                                 'save_samples': False
                                 }
                 }]
# CONFIG
model_name = 'ising'
sampler_name = 'dlmc'
experiment_type = 'test'

experiment_folder = f'{sampler_name}_{experiment_type}' 

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
    pass
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
    config.experiment.num_models = config.model.num_models

  if config.model.get('bert_model', None):
    pass
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