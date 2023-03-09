"""Experiment class that runs sampler on the model to generate chains."""
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import tqdm
import pdb
import flax
import time
import functools
import optax
from discs.common import utils


class Experiment:
  """Experiment class that generates chains of samples."""

  def __init__(self, config):
    self.config = config.experiment
    self.config_model = config.model
    if jax.local_device_count() == 1:
      self.run_parallel = False
    else:
      self.run_parallel = True

  def _build_temperature_schedule(self, config):
    """Temperature schedule."""

    if config.t_schedule == 'constant':
      schedule = lambda step: step * 0 + config.init_temperature
    elif config.t_schedule == 'linear':
      schedule = optax.linear_schedule(config.init_temperature,
                                     config.final_temperature,
                                     config.chain_length)
    elif config.t_schedule == 'exp_decay':
      schedule = optax.exponential_decay(
          config.init_temperature, config.chain_length, config.decay_rate,
          end_value=config.final_temperature)
    else:
      raise ValueError('Unknown schedule %s' % config.t_schedule)
    return schedule

  def _initialize_model_and_sampler(self, rnd, model, sampler_init_fn, datagen=None):
    num_samples = self.config.batch_size
    rng_param, rng_x0, rng_x0_ess, rng_state = jax.random.split(rnd, num=4)
    if datagen:
      data_list = next(datagen)
      _, params, _ = zip(*data_list)
      params = utils.tree_stack(params)
      num_samples *= self.config.num_models
    elif not self.config_model.get('data_path', None):
      params = model.make_init_params(rng_param)
    else:
      params = flax.core.frozen_dict.freeze(self.config_model.params)
    x0 = model.get_init_samples(rng_x0, num_samples)
    x0_ess = model.get_init_samples(rng_x0_ess, 1)
    state = sampler_init_fn(jax.random.split(rng_state, self.config.num_models))
    
    return params, x0, state, x0_ess

  def _split(self, arr, n_devices):
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

  def _prepare_state(self, state, n_devices):
    for key in state:
      state[key] = jnp.hstack([state[key]] * n_devices)
    return state

  def _prepare_data(self, params, x, state, n_devices, co_model=False):
    if co_model:
      bshape = (n_devices, self.config.num_models // n_devices)
      fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
      params = jax.tree_map(fn_breshape, params)
      state = jax.tree_map(fn_breshape, state)
      x = jnp.reshape(x, bshape + (self.config.batch_size,) + x.shape[1:])
    elif self.run_parallel:
      params = jnp.stack([params] * n_devices)
      state = self._prepare_state(state, n_devices)
      x = self._split(x, n_devices)
    return params, x, state

  def _compile_sampler_step(self, step_fn):
    if not self.run_parallel:
      compiled_step = jax.jit(step_fn)
    else:
      compiled_step = jax.pmap(step_fn)
    return compiled_step

  def _compile_evaluator(self, obj_fn_step, obj_fn_chain):
    if not self.run_parallel:
      compiled_eval_step = jax.jit(obj_fn_step, static_argnums=1)
      compiled_eval_chain = jax.jit(obj_fn_chain)

    else:
      compiled_eval_step = jax.pmap(obj_fn_step)
      compiled_eval_chain = jax.pmap(
          obj_fn_chain, static_broadcasted_argnums=[1]
      )
    return compiled_eval_step, compiled_eval_chain

  def _setup_num_devices(self):
    if not self.run_parallel:
      n_rand_split = 2
    else:
      n_rand_split = jax.local_device_count()
    return n_rand_split

  def get_results(self, model, sampler, evaluator, datagen=None):
    num_ll_calls, acc_ratios, hops, evals, running_time, _ = (
        self._get_chains_and_evaluations(model, sampler, evaluator, datagen)
    )
    metrcis = evaluator.get_eval_metrics(evals[-1], running_time, num_ll_calls)
    return metrcis, running_time, acc_ratios, hops

  def _get_vmapped_functions(self, sampler, model, evaluator):
    sampler_init_fn = jax.vmap(sampler.make_init_state)
    step_fn = jax.vmap(functools.partial(sampler.step, model=model))
    obj_fn_step = jax.vmap(functools.partial(evaluator.evaluate_step, model=model) )
    obj_fn_chain = jax.vmap(evaluator.evaluate_chain)
    return sampler_init_fn, step_fn, obj_fn_step, obj_fn_chain

  def _get_chains_and_evaluations(
      self, model, sampler, evaluator, datagen=None
  ):
    """Sets up the model and the samlping alg and gets the chain of samples."""
    sampler_init_fn, step_fn, obj_fn_step, obj_fn_chain = (
        self._get_vmapped_functions(sampler, model, evaluator)
    )
    rnd = jax.random.PRNGKey(0)
    params, x, state, x0_ess = self._initialize_model_and_sampler(
        rnd, model, sampler_init_fn, datagen
    )
    model_params = params
    n_rand_split = self._setup_num_devices()
    params, x, state = self._prepare_data(params, x, state, n_rand_split, datagen)
    compiled_step = self._compile_sampler_step(step_fn)
    compiled_eval_step , compiled_eval_chain = self._compile_evaluator(
       obj_fn_step, obj_fn_chain
    )
    t_schedule = self._build_temperature_schedule(self.config)
    state, acc_ratios, hops, evals, running_time = self._compute_chain(
        model,
        compiled_step,
        compiled_eval_step,
        compiled_eval_chain,
        state,
        params,
        rnd,
        x,
        n_rand_split,
        x0_ess,
        t_schedule,
    )
    if self.run_parallel:
      num_ll_calls = state['num_ll_calls'][0]
    else:
      num_ll_calls = state['num_ll_calls']
    return num_ll_calls, acc_ratios, hops, evals, running_time, model_params

  def _compute_chain(
      self,
      model,
      step_fn,
      eval_step_fn,
      eval_chain_fn,
      state,
      params,
      rng,
      x,
      n_rand_split,
      x0_ess,
      t_schedule
  ):
    """Generates the chain of samples."""
    chain = []
    acc_ratios = []
    hops = []
    evaluations = []
    running_time = 0
    bshape = (2, self.config.num_models//2)
    init_temperature = jnp.ones(bshape, dtype=jnp.float32)
    for step in tqdm.tqdm(range(self.config.chain_length)):
#      if self.run_parallel:
#        rng_sampler_step_p = jax.random.split(
#            rng_sampler_step, num=n_rand_split
#        )
#      else:
#        rng_sampler_step_p = rng_sampler_step
      cur_temp = t_schedule(step)
      params['temperature'] = init_temperature * cur_temp
      rng = jax.random.fold_in(rng, step)
      fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
      rng_sampler_step_p = fn_breshape(jax.random.split(rng, self.config.num_models))
      start = time.time()
      new_x, state, acc = step_fn(
          rng=rng_sampler_step_p, x=x, model_param=params, state=state, x_mask=params['mask']
      )
      running_time += time.time() - start
      eval_val = eval_step_fn(samples=new_x, params=params)
      if eval_val != None:
        eval_val = jnp.mean(eval_val)
        evaluations.append(eval_val)
      acc_ratios.append(acc)
      hops.append(self._get_hop(x, new_x))
      chain.append(self._get_mapped_samples(new_x, x0_ess))
      x = new_x
    chain = chain[int(self.config.chain_length * self.config.ess_ratio) :]
    chain = jnp.array(chain)
    eval_val = eval_chain_fn(chain, rng_sampler_step)
    if eval_val:
      evaluations.append(eval_val)
    return (
        state,
        jnp.array(acc_ratios),
        jnp.array(hops),
        jnp.array(evaluations),
        running_time,
    )

  def _get_mapped_samples(self, samples, x0_ess):
    samples = samples.reshape((-1,) + self.config_model.shape)
    x0_ess = x0_ess.reshape(x0_ess.shape[0], -1)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)

  def _get_hop(self, x, new_x):
    return jnp.sum(abs(x - new_x)) / self.config.batch_size


def build_experiment(config: config_dict):
  return Experiment(config)
