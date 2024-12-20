"""Main class that runs sampler on the model to generate chains."""
import functools
import time
from discs.common import math_util as math
from discs.common import utils
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import types


class Experiment:
  """Experiment class that generates chains of samples."""

  def __init__(self, config):
    self.config = config.experiment
    self.config_model = config.model
    self.parallel = False
    self.sample_idx = None
    self.num_saved_samples = config.experiment.get('num_saved_samples', 4)
    if jax.local_device_count() != 1 and self.config.run_parallel:
      self.parallel = True
    self.replica_exchange = False
    self.temperature = config.model.get('temperature',1.0)
    self.verbose = False
    self.batch_size = config.experiment.get('batch_size',1)

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    """Initializes model params, sampler state and gets the initial samples."""

    if self.config.evaluator == 'co_eval':
      sampler_init_state_fn = jax.vmap(sampler.make_init_state)
    else:
      sampler_init_state_fn = sampler.make_init_state

    # include a temperature param in every model
    if self.replica_exchange or self.temperature != 1.0:
      print('Updating forward method')
      def temperature_forward(self, params, x):
        # Call the original forward method
        original_fwd = self._original_forward(params, x)
        # Multiply the result by the temperature
        return original_fwd / params['temperature']

      model._original_forward = model.forward
      model.forward = types.MethodType(temperature_forward, model)

    model_init_params_fn = model.make_init_params
    rng_param, rng_x0, rng_state = jax.random.split(rnd, num=3)
    # params of the model
    params = model_init_params_fn(
        jax.random.split(rng_param,self.config.num_models) if self.config.num_models > 1 else rng_param
    )
    # initial samples
    num_samples = self.config.batch_size * self.config.num_models
    x0 = model.get_init_samples(rng_x0, num_samples)
    # initial state of sampler
    state = sampler_init_state_fn(
        jax.random.split(rng_state,self.config.num_models) if self.config.num_models > 1 else rng_state
    )
    return params, x0, state

  def _prepare_data(self, params, x, state):
    use_put_replicated = False
    reshape_all = True
    if self.config.evaluator != 'co_eval':
      if self.parallel:
        assert self.config.batch_size % jax.local_device_count() == 0
        mini_batch = self.config.batch_size // jax.local_device_count()
        bshape = (jax.local_device_count(),)
        x_shape = bshape + (mini_batch,) + self.config_model.shape
        use_put_replicated = True
        if self.sample_idx:
          self.sample_idx = jnp.array(
              [self.sample_idx]
              * (jax.local_device_count() // self.config.num_models)
          )
      else:
        reshape_all = False
        bshape = ()
        x_shape = (self.config.batch_size,) + self.config_model.shape
    else:
      if self.parallel:
        if self.config.num_models >= jax.local_device_count():
          assert self.config.num_models % jax.local_device_count() == 0
          num_models_per_device = (
              self.config.num_models // jax.local_device_count()
          )
          bshape = (jax.local_device_count(), num_models_per_device)
          x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
        else:
          assert self.config.batch_size % jax.local_device_count() == 0
          batch_size_per_device = (
              self.config.batch_size // jax.local_device_count()
          )
          use_put_replicated = True
          bshape = (jax.local_device_count(), self.config.num_models)
          x_shape = bshape + (batch_size_per_device,) + self.config_model.shape
          if self.sample_idx:
            self.sample_idx = jnp.array(
                [self.sample_idx]
                * (jax.local_device_count() // self.config.num_models)
            )
      else:
        bshape = (self.config.num_models,)
        x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
    fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    if reshape_all:
      if not use_put_replicated:
        state = jax.tree_map(fn_breshape, state)
        params = jax.tree_map(fn_breshape, params)
      else:
        params = jax.device_put_replicated(params, jax.local_devices())
        state = jax.device_put_replicated(state, jax.local_devices())
    x = jnp.reshape(x, x_shape)

    print('x shape: ', x.shape)
    print('state shape: ', state['steps'].shape)
    return params, x, state, bshape

  def _compile_sampler_step(self, step_fn):
    if not self.parallel:
      compiled_step = jax.jit(step_fn)
    else:
      compiled_step = jax.pmap(step_fn)
    return compiled_step

  def _compile_frwrd_step(self, frwrd_fn):
    if not self.parallel:
      compiled_frwrd = jax.jit(frwrd_fn)
    else:
      compiled_frwrd = jax.pmap(frwrd_fn)
    return compiled_frwrd

  def _compile_evaluator(self, obj_fn):
    if not self.parallel:
      compiled_obj_fn = jax.jit(obj_fn)
    else:
      compiled_obj_fn = jax.pmap(obj_fn)
    return compiled_obj_fn

  def _compile_fns(self, sampler, model, evaluator):
    if self.config.evaluator == 'co_eval':
      step_fn = jax.vmap(functools.partial(sampler.step, model=model))
      obj_fn = self._vmap_evaluator(evaluator, model)
      frwrd_fn = jax.vmap(model.forward)
    else:
      step_fn = functools.partial(sampler.step, model=model)
      obj_fn = evaluator.evaluate
      frwrd_fn = model.forward

    get_hop = jax.jit(self._get_hop)
    compiled_step = self._compile_sampler_step(step_fn)
    compiled_step_burnin = compiled_step
    compiled_step_mixing = compiled_step
    compiled_obj_fn = self._compile_evaluator(obj_fn)
    compiled_frwrd = self._compile_frwrd_step(frwrd_fn)
    return (
        compiled_step_burnin,
        compiled_step_mixing,
        get_hop,
        compiled_obj_fn,
        compiled_frwrd,
    )
  
  def get_init_replica_data(self, init_state, params, bshape, x0):
    repeated_x = jnp.expand_dims(x0, axis=0)
    full_x = jnp.repeat(repeated_x, repeats=self.num_replicas,axis=0)

    init_temperature = jnp.ones(bshape, dtype=jnp.float32)
    repeated_init_temperature = jnp.expand_dims(init_temperature, axis=-1)
    init_temperature = jnp.repeat(repeated_init_temperature, repeats=self.batch_size,axis=-1)

    params['temperature'] = init_temperature
    state_keys = list(init_state.keys())
    state_array = jnp.array([init_state[key] for key in state_keys])
    repeated_state_array = jnp.expand_dims(state_array, axis=0)
    full_state_array = jnp.repeat(repeated_state_array, repeats=self.num_replicas,axis=0)

    init_replica_temps = jnp.expand_dims(self.replica_temps,axis=1+jnp.arange(len(init_temperature.shape)))*jnp.expand_dims(init_temperature,axis=0)

    return full_x, params, full_state_array, init_replica_temps


  def _compile_exchange_fns(self, compiled_sampler_step, compiled_frwrd, init_state, init_params, fn_reshape, bshape):
    
    state_keys = list(init_state.keys())

    def step_and_energy(x, state_array, temperature, rng, params):
      params_copy = params.copy()
      params_copy['temperature'] = temperature

      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))

      state = {state_keys[j]: state_array[j] for j in range(len(state_keys))}

      new_partial_x, new_state, _ = compiled_sampler_step(
          rng=step_rng,
          x=x,
          model_param=params_copy,
          state=state
      )

      energy = -compiled_frwrd(params,new_partial_x)
      new_state_array = jnp.array([new_state[key] for key in state_keys])
      
      return new_partial_x, energy, new_state_array

    def exchange_step(energies,swap_rng,temps,offset=0):
      
      arr = jnp.arange(self.num_replicas)
      arr = arr.reshape((self.num_replicas,) + (1,) * len(energies[0].shape))
      particle_labels = jnp.broadcast_to(arr, (self.num_replicas,) + energies[0].shape)

      swap_probs = jnp.zeros((energies.shape[0]-1,) + energies.shape[1:])
      
      swap_indices = range(offset,self.num_replicas-1,2)

      for i in swap_indices:
        energy_diff = energies[i]-energies[i+1]
        swap_prob = jnp.exp(jnp.clip((1/temps[i]-1/temps[i+1])*energy_diff,max=0))
        swap_probs = swap_probs.at[i].set(swap_prob)

        swap = jax.random.bernoulli(swap_rng[i],swap_prob)     

        swapped_first = jnp.where(swap,particle_labels[i+1],particle_labels[i])
        swapped_second = jnp.where(swap,particle_labels[i],particle_labels[i+1])
        particle_labels = particle_labels.at[i].set(swapped_first)
        particle_labels = particle_labels.at[i+1].set(swapped_second)
      
      return particle_labels, swap_probs
    
    if not self.config.evaluator == 'co_eval' or not self.parallel:
      full_step_and_energy = jax.vmap(jax.jit(functools.partial(step_and_energy, params=init_params)))
      exchange_step_even = jax.jit(functools.partial(exchange_step,offset=0))
      exchange_step_odd = jax.jit(functools.partial(exchange_step,offset=1))
    else:
      full_step_and_energy = jax.vmap(functools.partial(step_and_energy, params=init_params))
      exchange_step_even = functools.partial(exchange_step,offset=0)
      exchange_step_odd = functools.partial(exchange_step,offset=1)
      

    def offset_exchange_step(energies,swap_rng,temps,offset):
      if offset == 0:
        return exchange_step_even(energies,swap_rng,temps)
      elif offset == 1:
        return exchange_step_odd(energies,swap_rng,temps)
      else:
        raise ValueError

    def full_exchange_step(new_full_x, energies,swap_rng,temps,offset):
      dims_to_add = len(new_full_x.shape) - len(energies.shape)

      new_particle_labels, swap_prob = offset_exchange_step(energies, swap_rng, temps, offset)
      #new_particle_labels = jnp.expand_dims(new_particle_labels, axis = jnp.arange(len(new_full_x.shape))[2:])
      new_particle_labels = new_particle_labels.reshape(new_particle_labels.shape + (1,)*dims_to_add)
      new_full_x = jnp.take_along_axis(new_full_x,new_particle_labels,axis=0)
      return new_full_x, swap_prob

    def adaptive_temp_step(n, running_mean, running_ssd, prob, target_acc, replica_temps, alpha):
      new_running_mean = (n-1)/n*running_mean + 1/n*prob
      new_running_ssd = running_ssd + (prob - running_mean)*(prob - new_running_mean)
      std = jnp.sqrt(new_running_ssd/n)

      outside_bound = (jnp.abs(new_running_mean-target_acc) > 0.3*std) & (n > 100)

      # print(f'means: {new_running_mean[:,0,0]}')
      # print(f'stds: {std[:,0,0]}')
      # print(f'ns: {n[:,0,0]}')
      # print(f'temps: {replica_temps[:,0,0]}')
      log_lowest_temp = jnp.log(replica_temps[0])
      log_temp_diffs = jnp.diff(jnp.log(replica_temps),axis=0)
      ratio_log_acc = jnp.clip(jnp.log(target_acc)/jnp.log(new_running_mean),a_min=0.5,a_max=2)
      new_log_temp_diffs = ratio_log_acc*log_temp_diffs
      
      log_temp_diffs = jnp.where(outside_bound, jnp.clip(log_temp_diffs*(1-alpha) + new_log_temp_diffs*alpha,a_min=-5,a_max=5), log_temp_diffs)

      replica_temps = replica_temps.at[1:].set(jnp.clip(jnp.exp(jnp.cumsum(log_temp_diffs,axis=0)+log_lowest_temp)))
      new_running_mean = jnp.where(outside_bound, jnp.zeros_like(new_running_mean),new_running_mean)
      new_running_ssd = jnp.where(outside_bound, jnp.zeros_like(new_running_ssd),new_running_ssd)
      
      n = jnp.where(outside_bound, jnp.ones_like(n),jnp.minimum(n+1,2000))

      return n, new_running_mean, new_running_ssd, replica_temps
    
    return full_step_and_energy, full_exchange_step, jax.jit(adaptive_temp_step)



  def _get_hop(self, x, new_x):
    return (
        jnp.sum(abs(x - new_x))
        / self.config.batch_size
        / self.config.num_models
    )

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    raise NotImplementedError

  def vmap_evaluator(self, evaluator, model):
    raise NotImplementedError

  def preprocess(self, model, sampler, evaluator, saver, rnd_key=0):
    rnd = jax.random.PRNGKey(rnd_key)
    params, x, state = self._initialize_model_and_sampler(rnd, model, sampler)
    if params is None:
      print('Params is NONE')
      return False
    params, x, state, breshape = self._prepare_data(params, x, state)
    compiled_fns = self._compile_fns(sampler, model, evaluator)
    return [
        compiled_fns,
        state,
        params,
        rnd,
        x,
        saver,
        evaluator,
        breshape,
        model,
    ]

  def _get_chains_and_evaluations(
      self, model, sampler, evaluator, saver, rnd_key=0
  ):
    """Sets up the model and the samlping alg and gets the chain of samples."""
    preprocessed_info = self.preprocess(
        model, sampler, evaluator, saver, rnd_key=0
    )
    if not preprocessed_info:
      return False
    self._compute_chain(*preprocessed_info)
    return True

  def get_results(self, model, sampler, evaluator, saver):
    self._get_chains_and_evaluations(model, sampler, evaluator, saver)

  


class Sampling_Experiment(Experiment):
  """Class used to run classical graphical models and computes ESS."""

  def _vmap_evaluator(self, evaluator, model):
    obj_fn = evaluator.evaluate
    return obj_fn

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    """Generates the chain of samples."""
    assert self.config.num_models == 1
    (chain, acc_ratios, hops, running_time, samples) = (
        self._initialize_chain_vars()
    )
    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    # sample used to map for ess computation
    rng_x0_ess, rng = jax.random.split(rng)
    x0_ess = model.get_init_samples(rng_x0_ess, 1)
    stp_burnin, stp_mixing, get_hop, obj_fn, model_frwrd = compiled_fns
    get_mapped_samples, eval_metric = self._compile_additional_fns(evaluator)
    rng = jax.random.PRNGKey(10)
    selected_chains = jax.random.choice(
        rng,
        jnp.arange(self.config.batch_size),
        shape=(self.num_saved_samples,),
        replace=False,
    )
    
    init_temperature = jnp.ones(bshape, dtype=jnp.float32)*self.temperature
    params['temperature'] = init_temperature
    energies_array = []
    magnetizations = []
    magnetization_estimate = jnp.zeros(x.shape[0])

    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    for step in tqdm.tqdm(range(1, burn_in_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      new_x, state, acc = stp_burnin(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      
      energies = -model_frwrd(params,new_x)
      if self.verbose:
        print(new_x[0])
        print(energies[0])
      energies_array.append(jnp.array([energies]))

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = stp_mixing(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      running_time += time.time() - start

      energies = -model_frwrd(params,new_x)
      if self.verbose:
        print(new_x[0])
        print(energies[0])
      energies_array.append(jnp.array([energies]))
      
      n = step - burn_in_length + 1
      magnetization_estimate = (n-1)/n*magnetization_estimate + 1/n*(jnp.array([(2*new_x-1).sum(axis=[-1,-2])]))

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])
        magnetizations.append(jnp.array([magnetization_estimate]))

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      mapped_sample = get_mapped_samples(new_x, x0_ess)
      mapped_sample = jax.device_put(mapped_sample, jax.devices('cpu')[0])
      chain.append(mapped_sample)
      x = new_x

    chain = jnp.array(chain)
    if self.parallel:
      chain = jnp.array([chain])
      rng = jnp.array([rng])
      num_ll_calls = int(state['num_ll_calls'][0])
    else:
      num_ll_calls = int(state['num_ll_calls'])
    ess = obj_fn(samples=chain, rnd=rng)
    metrics = eval_metric(ess, running_time, num_ll_calls)
    saver.save_results(acc_ratios, hops, metrics, running_time)
    if self.config.save_samples or self.config.get_estimation_error:
      if self.config.save_samples and self.config_model.name in [
          'rbm',
          'resnet',
      ]:
        saver.dump_samples(samples, visualize=False)
      elif (
          self.config.get_estimation_error
      ):
        if self.config.save_samples:
          saver.dump_samples(samples, visualize=False)
        # samples= np.array(samples)
        # params = params['params'][0].reshape(self.config_model.shape)
        saver.dump_params(params)
        # saver.plot_estimation_error(model, params, samples)
    
    print(f'running time: {running_time}')
    #saver.dump_array(jnp.concatenate(energies_array),'energies')
    saver.dump_array(jnp.concatenate(magnetizations),'magnetizations')

  def _initialize_chain_vars(self):
    chain = []
    acc_ratios = []
    hops = []
    samples = []
    running_time = 0

    return (
        chain,
        acc_ratios,
        hops,
        running_time,
        samples,
    )

  def _compile_additional_fns(self, evaluator):
    get_mapped_samples = jax.jit(self._get_mapped_samples)
    eval_metric = jax.jit(evaluator.get_eval_metrics)
    return get_mapped_samples, eval_metric

  def _get_mapped_samples(self, samples, x0_ess):
    samples = samples.reshape((-1,) + self.config_model.shape)
    samples = samples.reshape(samples.shape[0], -1)
    x0_ess = x0_ess.reshape((-1,) + self.config_model.shape)
    x0_ess = x0_ess.reshape(x0_ess.shape[0], -1)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)

class RE_Sampling_Experiment(Experiment):
  """Class used to run classical graphical models and computes ESS."""

  def __init__(self, config):
    super().__init__(config)
    self.num_replicas = config.experiment.get('num_replicas',2)
    self.min_temp= config.model.temperature
    self.max_temp = config.model.temperature*config.experiment.max_temp_mult
    self.replica_temps = jnp.logspace(jnp.log10(self.min_temp),jnp.log10(self.max_temp),self.num_replicas)
    self.save_replica_data = config.experiment.get('save_replica_data',False)
    self.adaptive_temps = config.experiment.get('adaptive_temps', False)
    self.target_swap_prob = 0.2
    self.update_temps_every = 5000
    self.replica_exchange = True

  def _vmap_evaluator(self, evaluator, model):
    obj_fn = evaluator.evaluate
    return obj_fn

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    """Generates the chain of samples."""
    assert self.config.num_models == 1
    (chain, acc_ratios, hops, running_time, samples) = (
        self._initialize_chain_vars()
    )

    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    # sample used to map for ess computation
    rng_x0_ess, rng = jax.random.split(rng)
    x0_ess = model.get_init_samples(rng_x0_ess, 1)
    get_mapped_samples, eval_metric = self._compile_additional_fns(evaluator)
    rng = jax.random.PRNGKey(10)
    selected_chains = jax.random.choice(
        rng,
        jnp.arange(self.config.batch_size),
        shape=(self.num_saved_samples,),
        replace=False,
    )
    print(self.min_temp)
    print(self.max_temp)
    stp_burnin, stp_mixing, get_hop, obj_fn, model_frwrd = compiled_fns

    full_x, params, full_state_array, init_replica_temps = self.get_init_replica_data(state,params,bshape,x)
    
    self.replica_temps = init_replica_temps
    
    magnetizations = []

    if self.save_replica_data:
      #track the temperature of a particle
      swap_probs = []
      energies_array = []

      if self.adaptive_temps:
        temp_array = []

    if self.config.num_models == 1:
      swap_prob_shape = [self.num_replicas-1] + [x.shape[0]]
    else:
      swap_prob_shape = (self.num_replicas-1,) + bshape + (self.config.batch_size,) 

    swap_prob = jnp.zeros(swap_prob_shape)
    magnetization_estimate = jnp.zeros(x.shape[0])

    if self.adaptive_temps:
      n = jnp.ones(swap_prob_shape)
      running_mean = jnp.zeros(swap_prob_shape)
      running_ssd = jnp.zeros(swap_prob_shape)

    full_step_and_energy, full_exchange_step, adaptive_temp_step = self._compile_exchange_fns(stp_burnin, model_frwrd, state, params, fn_reshape, bshape)

    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1

    for step in tqdm.tqdm(range(1, burn_in_length)):

      rng = jax.random.fold_in(rng, step)
      step_rng = jax.random.split(rng,self.num_replicas)
      
      new_full_x, energies, full_state_array = full_step_and_energy(full_x,full_state_array,self.replica_temps,step_rng)
      
      swap_rng = jax.random.split(step_rng[0],self.num_replicas-1)

      new_full_x, new_swap_prob = full_exchange_step(new_full_x, energies, swap_rng, self.replica_temps, step % 2)
      swap_prob += new_swap_prob

      new_x = new_full_x[0]

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])

      if self.save_replica_data and step % self.config.save_every_steps == 0:
        swap_probs.append(jnp.array([swap_prob]))
        energies_array.append(jnp.array([energies]))
        if self.adaptive_temps:
          temp_array.append(jnp.array([self.replica_temps]))
      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))

      if self.adaptive_temps and step % 2 == 0:
        alpha = (1 - step/burn_in_length)
        n, running_mean, running_ssd, replica_temps = adaptive_temp_step(n, running_mean, running_ssd, swap_prob, self.target_swap_prob, self.replica_temps, alpha)
        self.replica_temps = replica_temps
        swap_prob = jnp.zeros(swap_prob_shape)


      full_x = new_full_x

    forward_time = 0
    swap_time = 0

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      start = time.time()
      rng = jax.random.fold_in(rng, step)
      step_rng = jax.random.split(rng,self.num_replicas)
      
      new_full_x, energies, full_state_array = full_step_and_energy(full_x,full_state_array,self.replica_temps,step_rng)
      
      swap_rng = jax.random.split(step_rng[0],self.num_replicas-1)

      new_full_x, new_swap_prob = full_exchange_step(new_full_x, energies, swap_rng, self.replica_temps, step % 2)
      swap_prob += new_swap_prob

      new_x = new_full_x[0]

      n = step - burn_in_length + 1
      magnetization_estimate = (n-1)/n*magnetization_estimate + 1/n*(jnp.array([(2*new_x-1).sum(axis=[-1,-2])]))

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])

        if self.save_replica_data:
          swap_probs.append(jnp.array([swap_prob]))
          energies_array.append(jnp.array([energies]))

        magnetizations.append(jnp.array([magnetization_estimate]))


      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      
      #print(new_x)
      mapped_sample = get_mapped_samples(new_x, x0_ess)
      mapped_sample = jax.device_put(mapped_sample, jax.devices('cpu')[0])
      chain.append(mapped_sample)
      full_x = new_full_x
      running_time += time.time() - start

    state_keys = list(state.keys())
    state = {state_keys[j]: full_state_array[0,j] for j in range(len(state_keys))}

    chain = jnp.array(chain)
    if self.parallel:
      chain = jnp.array([chain])
      rng = jnp.array([rng])
      num_ll_calls = int(state['num_ll_calls'][0])
    else:
      num_ll_calls = int(state['num_ll_calls'])
    ess = obj_fn(samples=chain, rnd=rng)
    metrics = eval_metric(ess, running_time, num_ll_calls)
    saver.save_results(acc_ratios, hops, metrics, running_time)
    if self.config.save_samples or self.config.get_estimation_error:
      if self.config.save_samples and self.config_model.name in [
          'rbm',
          'resnet',
      ]:
        saver.dump_samples(samples, visualize=False)
      elif (
          self.config.get_estimation_error
      ):
        if self.config.save_samples:
          saver.dump_samples(samples, visualize=False)
        # samples= np.array(samples)
        # params = params['params'][0].reshape(self.config_model.shape)
        saver.dump_params(params)
        # saver.plot_estimation_error(model, params, samples)
    
    if self.save_replica_data:
        #saver.dump_array(jnp.concatenate(swap_probs),'swap_probs')
        #saver.dump_array(jnp.concatenate(energies_array),'energies')
        if self.adaptive_temps:
          saver.dump_array(jnp.concatenate(temp_array),'temps')

    saver.dump_array(jnp.concatenate(magnetizations), 'magnetizations')

  def _initialize_chain_vars(self):
    chain = []
    acc_ratios = []
    hops = []
    samples = []
    running_time = 0
  
    return (
        chain,
        acc_ratios,
        hops,
        running_time,
        samples,
    )

  def _compile_additional_fns(self, evaluator):
    get_mapped_samples = jax.jit(self._get_mapped_samples)
    eval_metric = jax.jit(evaluator.get_eval_metrics)
    return get_mapped_samples, eval_metric

  def _get_mapped_samples(self, samples, x0_ess):
    samples = samples.reshape((-1,) + self.config_model.shape)
    samples = samples.reshape(samples.shape[0], -1)
    x0_ess = x0_ess.reshape((-1,) + self.config_model.shape)
    x0_ess = x0_ess.reshape(x0_ess.shape[0], -1)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)


class Text_Infilling_Experiment(Sampling_Experiment):
  """Class used to sample sentences for text infilling."""

  def get_results(self, model, sampler, evaluator, saver):
    obj_fn = jax.jit(evaluator.evaluate)
    infill_sents = []
    infill_sents_topk = []
    rnd_key = 0
    while True:
      contin, sents, sents_topk = self._get_chains_and_evaluations(
          model, sampler, evaluator, saver, rnd_key=rnd_key
      )
      rnd_key += 1
      if not contin:
        break
      infill_sents.extend(sents)
      if self.config.use_topk:
        infill_sents_topk.extend(sents_topk)
    res = obj_fn(infill_sents, self.config_model.data_root)
    if self.config.use_topk:
      res_topk = evaluator.evaluate(
          infill_sents_topk, self.config_model.data_root
      )
    else:
      res_topk = []
    saver.save_lm_results(res, res_topk)

  def _get_chains_and_evaluations(
      self, model, sampler, evaluator, saver, rnd_key=0
  ):
    preprocessed_info = self.preprocess(
        model, sampler, evaluator, saver, rnd_key=0
    )
    if not preprocessed_info:
      return False, None, None
    sentences = []
    loglikes = []
    topk_sentences = []

    obj_fn = preprocessed_info[0][-1]
    for i in range(self.config.num_same_resample):
      sent, rng, loglike = self._compute_chain(*preprocessed_info)
      if self.config.use_topk:
        sent = str(i) + ' ' + sent
        loglikes.append(loglike)
      sentences.append(sent)
      preprocessed_info[3] = rng

    if self.config.use_topk:
      sent_to_loglike = dict(zip(sentences, loglikes))
      sorted_sent = {
          k: v
          for k, v in sorted(sent_to_loglike.items(), key=lambda item: item[1])
      }
      topk_sentences = list(sorted_sent.keys())[-self.config.topk_num :]
      for i, sent in enumerate(topk_sentences):
        topk_sentences[i] = sent[2:]
      for i, sent in enumerate(sentences):
        sentences[i] = sent[2:]

    return True, sentences, topk_sentences

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    """Generates the chain of samples."""
    assert self.config.num_models == 1
    (_, acc_ratios, hops, running_time, _) = self._initialize_chain_vars()

    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    stp_burnin, stp_mixing, get_hop, _, model_frwrd = compiled_fns

    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    for step in tqdm.tqdm(range(1, burn_in_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      new_x, state, acc = stp_burnin(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = stp_mixing(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      running_time += time.time() - start
      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    loglike = 0
    if self.config.use_topk:
      x = x.astype(jnp.float32)
      loglike = model_frwrd(params, x)[0]

    sampled_sentence = model.decode(x, params)
    print('Sampled Sentence: ', sampled_sentence, 'Likelihood: ', loglike)
    return sampled_sentence, rng, loglike


class CO_Experiment(Experiment):
  """Class used to run annealing schedule for CO problems."""

  def get_results(self, model, sampler, evaluator, saver):
    while True:
      if not self._get_chains_and_evaluations(model, sampler, evaluator, saver):
        break

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    data_list, x0, state = super()._initialize_model_and_sampler(
        rnd, model, sampler
    )
    if data_list is None:
      return None, x0, state
    sample_idx, params, reference_obj = zip(*data_list)
    params = flax.core.frozen_dict.unfreeze(utils.tree_stack(params))
    self.ref_obj = jnp.array(reference_obj)
    if self.config_model.name == 'mis':
      self.ref_obj = jnp.ones_like(self.ref_obj)
    self.sample_idx = jnp.array(sample_idx)
    return params, x0, state

  def _vmap_evaluator(self, evaluator, model):
    obj_fn = jax.vmap(functools.partial(evaluator.evaluate, model=model))
    return obj_fn

  def _build_temperature_schedule(self, config):
    """Temperature schedule."""

    if config.t_schedule == 'constant':
      schedule = lambda step: step * 0 + config.init_temperature
    elif config.t_schedule == 'linear':
      schedule = optax.linear_schedule(
          config.init_temperature, config.final_temperature, config.chain_length
      )
    elif config.t_schedule == 'exp_decay':
      schedule = optax.exponential_decay(
          config.init_temperature,
          config.chain_length,
          config.decay_rate,
          end_value=config.final_temperature,
      )
    else:
      raise ValueError('Unknown schedule %s' % config.t_schedule)
    return schedule

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    """Generates the chain of samples."""

    (
        chain,
        acc_ratios,
        hops,
        running_time,
        best_ratio,
        init_temperature,
        t_schedule,
        sample_mask,
        best_samples,
    ) = self._initialize_chain_vars(bshape)

    stp_burnin, stp_mixing, get_hop, obj_fn, model_frwrd = compiled_fns
    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    energies_array = []
    
    for step in tqdm.tqdm(range(1, burn_in_length)):
      cur_temp = t_schedule(step)
      params['temperature'] = init_temperature * cur_temp
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      
      new_x, state, acc = stp_burnin(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
          x_mask=params['mask'],
      )

      if step % self.config.log_every_steps == 0:
        params['temperature'] = jnp.ones_like(params['temperature'])
        energies = -model_frwrd(params,new_x)
        if self.verbose:
          print(new_x[0])
          print(energies[0])
        energies_array.append(jnp.array([energies]))
        
        eval_val = obj_fn(samples=new_x, params=params)
        eval_val = eval_val.reshape(self.config.num_models, -1)
        ratio = jnp.max(eval_val, axis=-1).reshape(-1) / self.ref_obj
        is_better = ratio > best_ratio
        best_ratio = jnp.maximum(ratio, best_ratio)
        sample_mask = sample_mask.reshape(best_ratio.shape)

        br = np.array(best_ratio[sample_mask])
        br = jax.device_put(br, jax.devices('cpu')[0])
        chain.append(br)
        #jax.debug.print(f'Temp after {step} steps: {params['temperature']}')

        if self.config.save_samples or self.config_model.name == 'normcut':
          step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
          rnew_x = jnp.reshape(
              new_x,
              (self.config.num_models, self.config.batch_size)
              + self.config_model.shape,
          )
          chosen_samples = jnp.take_along_axis(
              rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
          )
          chosen_samples = jnp.squeeze(chosen_samples, -2)
          best_samples = jnp.where(
              jnp.expand_dims(is_better, -1), chosen_samples, best_samples
          )

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      cur_temp = t_schedule(step)
      params['temperature'] = init_temperature * cur_temp
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = stp_mixing(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
          x_mask=params['mask'],
      )
      running_time += time.time() - start
      if step % self.config.log_every_steps == 0:
        params['temperature'] = jnp.ones_like(params['temperature'])
        energies = -model_frwrd(params,new_x)
        if self.verbose:
          print(new_x[0])
          print(energies[0])
        energies_array.append(jnp.array([energies]))

        eval_val = obj_fn(samples=new_x, params=params)
        eval_val = eval_val.reshape(self.config.num_models, -1)
        ratio = jnp.max(eval_val, axis=-1).reshape(-1) / self.ref_obj
        is_better = ratio > best_ratio
        best_ratio = jnp.maximum(ratio, best_ratio)
        sample_mask = sample_mask.reshape(best_ratio.shape)

        br = np.array(best_ratio[sample_mask])
        br = jax.device_put(br, jax.devices('cpu')[0])
        chain.append(br)

        if self.config.save_samples or self.config_model.name == 'normcut':
          step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
          rnew_x = jnp.reshape(
              new_x,
              (self.config.num_models, self.config.batch_size)
              + self.config_model.shape,
          )
          chosen_samples = jnp.take_along_axis(
              rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
          )
          chosen_samples = jnp.squeeze(chosen_samples, -2)
          best_samples = jnp.where(
              jnp.expand_dims(is_better, -1), chosen_samples, best_samples
          )

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    if not (self.config.save_samples or self.config_model.name == 'normcut'):
      best_samples = []
    saver.save_co_resuts(
        chain, best_ratio[sample_mask], running_time, best_samples
    )
    saver.save_results(acc_ratios, hops, None, running_time)
    saver.dump_array(jnp.concatenate(energies_array),'energies')

  def _initialize_chain_vars(self, bshape):
    t_schedule = self._build_temperature_schedule(self.config)
    sample_mask = self.sample_idx >= 0
    chain = []
    acc_ratios = []
    hops = []
    running_time = 0
    best_ratio = jnp.ones(self.config.num_models, dtype=jnp.float32) * -1e9
    init_temperature = jnp.ones(bshape, dtype=jnp.float32)
    dim = math.prod(self.config_model.shape)
    best_samples = jnp.zeros([self.config.num_models, dim])
    return (
        chain,
        acc_ratios,
        hops,
        running_time,
        best_ratio,
        init_temperature,
        t_schedule,
        sample_mask,
        best_samples,
    )

class RE_CO_Experiment(CO_Experiment):
  
  def __init__(self, config):
    super().__init__(config)
    self.num_replicas = config.experiment.get('num_replicas',2)
    self.min_temp= config.experiment.minimum_temperature
    self.max_temp = config.experiment.maximum_temperature
    self.replica_temps = jnp.logspace(jnp.log10(self.min_temp),jnp.log10(self.max_temp),self.num_replicas)
    self.save_replica_data = config.experiment.get('save_replica_data',False)
    self.batch_size = config.experiment.get('batch_size',1)
    self.adaptive_temps = config.experiment.get('adaptive_temps', False)
    self.target_swap_prob = 0.2
    self.update_temps_every = 250
    
  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    """Generates the chain of samples."""

    (
        chain,
        acc_ratios,
        hops,
        running_time,
        best_ratio,
        init_temperature,
        t_schedule,
        sample_mask,
        best_samples,
    ) = self._initialize_chain_vars(bshape)

    stp_burnin, stp_mixing, get_hop, obj_fn, model_frwrd = compiled_fns
    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    # burn in
    full_x, params, full_state_array, init_replica_temps = self.get_init_replica_data(state,params,bshape,x)
    
    self.replica_temps = init_replica_temps
    
    magnetizations = []

    if self.save_replica_data:
      #track the temperature of a particle
      swap_probs = []
      energies_array = []

      if self.adaptive_temps:
        temp_array = []

    if self.config.num_models == 1:
      swap_prob_shape = [self.num_replicas-1] + [x.shape[0]]
    else:
      swap_prob_shape = (self.num_replicas-1,) + bshape + (self.config.batch_size,) 
    swap_prob = jnp.zeros(swap_prob_shape)

    if self.adaptive_temps:
      n = jnp.ones(swap_prob_shape)
      running_mean = jnp.zeros(swap_prob_shape)
      running_ssd = jnp.zeros(swap_prob_shape)

    full_step_and_energy, full_exchange_step, adaptive_temp_step = self._compile_exchange_fns(stp_burnin, model_frwrd, state, params, fn_reshape, bshape)

    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1

    for step in tqdm.tqdm(range(1, burn_in_length)):
      
      cur_temp = t_schedule(step)
      if step == 1 or not self.adaptive_temps:
        self.replica_temps = init_replica_temps * cur_temp
      
      else:
        self.replica_temps = self.replica_temps.at[0].set(init_replica_temps[0]*cur_temp)


      rng = jax.random.fold_in(rng, step)
      step_rng = jax.random.split(rng,self.num_replicas)

      new_full_x, energies, full_state_array = full_step_and_energy(full_x,full_state_array,self.replica_temps,step_rng)
      
      swap_rng = jax.random.split(step_rng[0],self.num_replicas-1)

      new_full_x, new_swap_prob = full_exchange_step(new_full_x, energies, swap_rng, self.replica_temps, step % 2)
      swap_prob += new_swap_prob

      new_x = new_full_x[0]

      #new_x, energy = step_and_energy_burnin(0)

      if self.adaptive_temps and step % 2 == 0:
        alpha = (1 - step/self.config.chain_length)
        n, running_mean, running_ssd, replica_temps = adaptive_temp_step(n, running_mean, running_ssd, swap_prob, self.target_swap_prob, self.replica_temps, alpha)
        self.replica_temps = replica_temps
        swap_prob = jnp.zeros(swap_prob_shape)

      if step % self.config.log_every_steps == 0:
        eval_val = obj_fn(samples=new_x, params=params)
        eval_val = eval_val.reshape(self.config.num_models, -1)
        ratio = jnp.max(eval_val, axis=-1).reshape(-1) / self.ref_obj
        is_better = ratio > best_ratio
        best_ratio = jnp.maximum(ratio, best_ratio)
        sample_mask = sample_mask.reshape(best_ratio.shape)

        br = np.array(best_ratio[sample_mask])
        br = jax.device_put(br, jax.devices('cpu')[0])
        chain.append(br)

        if self.save_replica_data:
          swap_probs.append(jnp.array([swap_prob]))
          energies_array.append(jnp.array([energies]))
          if self.adaptive_temps:
            temp_array.append(jnp.array([self.replica_temps]))

        if self.config.save_samples or self.config_model.name == 'normcut':
          step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
          rnew_x = jnp.reshape(
              new_x,
              (self.config.num_models, self.config.batch_size)
              + self.config_model.shape,
          )
          chosen_samples = jnp.take_along_axis(
              rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
          )
          chosen_samples = jnp.squeeze(chosen_samples, -2)
          best_samples = jnp.where(
              jnp.expand_dims(is_better, -1), chosen_samples, best_samples
          )
      

      if self.config.get_additional_metrics:
        # avg over all models
        swap_acc = jnp.mean(swap_acc)
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))

      full_x = new_full_x

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      start = time.time()
      cur_temp = t_schedule(step)
      if step == 1 or not self.adaptive_temps:
        self.replica_temps = init_replica_temps * cur_temp
      
      else:
        self.replica_temps = self.replica_temps.at[0].set(init_replica_temps[0]*cur_temp)

      rng = jax.random.fold_in(rng, step)
      step_rng = jax.random.split(rng,self.num_replicas)

      new_full_x, energies, full_state_array = full_step_and_energy(full_x,full_state_array,self.replica_temps,step_rng)
      
      swap_rng = jax.random.split(step_rng[0],self.num_replicas-1)

      new_full_x, new_swap_prob = full_exchange_step(new_full_x, energies, swap_rng, self.replica_temps, step % 2)
      swap_prob += new_swap_prob

      if self.adaptive_temps and step % 2 == 0:
        alpha = (1 - step/self.config.chain_length)
        n, running_mean, running_ssd, replica_temps = adaptive_temp_step(n, running_mean, running_ssd, swap_prob, self.target_swap_prob, self.replica_temps, alpha)
        self.replica_temps = replica_temps
        swap_prob = jnp.zeros(swap_prob_shape)
      
      new_x = new_full_x[0]
      running_time += time.time() - start

      if step % self.config.log_every_steps == 0:
        eval_val = obj_fn(samples=new_x, params=params)
        eval_val = eval_val.reshape(self.config.num_models, -1)
        ratio = jnp.max(eval_val, axis=-1).reshape(-1) / self.ref_obj
        is_better = ratio > best_ratio
        best_ratio = jnp.maximum(ratio, best_ratio)
        sample_mask = sample_mask.reshape(best_ratio.shape)

        br = np.array(best_ratio[sample_mask])
        br = jax.device_put(br, jax.devices('cpu')[0])
        chain.append(br)
        
        if self.save_replica_data:
          swap_probs.append(jnp.array([swap_prob]))
          energies_array.append(jnp.array([energies]))
          if self.adaptive_temps:
            temp_array.append(jnp.array([self.replica_temps]))

        if self.config.save_samples or self.config_model.name == 'normcut':
          step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
          rnew_x = jnp.reshape(
              new_x,
              (self.config.num_models, self.config.batch_size)
              + self.config_model.shape,
          )
          chosen_samples = jnp.take_along_axis(
              rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
          )
          chosen_samples = jnp.squeeze(chosen_samples, -2)
          best_samples = jnp.where(
              jnp.expand_dims(is_better, -1), chosen_samples, best_samples
          )

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      
      full_x = new_full_x
      

    if not (self.config.save_samples or self.config_model.name == 'normcut'):
      best_samples = []
    saver.save_co_resuts(
        chain, best_ratio[sample_mask], running_time, best_samples
    )
    saver.save_results(acc_ratios, hops, None, running_time)

    if self.save_replica_data:
      saver.dump_array(jnp.concatenate(swap_probs),'swap_probs')
      saver.dump_array(jnp.concatenate(energies_array),'energies')
      if self.adaptive_temps:
        saver.dump_array(jnp.concatenate(temp_array),'temps')


class EBM_Experiment(Experiment):

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    """Initializes model params, sampler state and gets the initial samples."""
    rng_param, rng_x, rng_state = jax.random.split(rnd, num=3)
    del rnd
    params = model.make_init_params(rng_param)
    params['temperature'] = 0
    x = model.get_init_samples(rng_x, self.config.batch_size)
    state = sampler.make_init_state(rng_state)
    return params, x, state

  def _compile_fns(self, sampler, model):
    if not self.parallel:
      score_fn = jax.jit(model.forward)
      step_fn = jax.jit(functools.partial(sampler.step, model=model))
    else:
      score_fn = jax.pmap(model.forward)
      step_fn = jax.pmap(functools.partial(sampler.step, model=model))
    return (
        score_fn,
        step_fn,
    )

  def preprocess(self, model, sampler, evaluator, saver, rnd_key=0):
    rnd = jax.random.PRNGKey(rnd_key)
    params, x, state = self._initialize_model_and_sampler(rnd, model, sampler)
    if self.parallel:
      params = jax.device_put_replicated(params, jax.local_devices())
      state = jax.device_put_replicated(state, jax.local_devices())
      assert self.config.batch_size % jax.local_device_count() == 0
      nn = self.config.batch_size // jax.local_device_count()
      x = x.reshape((jax.local_device_count(), nn) + self.config_model.shape)
    compiled_fns = self._compile_fns(sampler, model)
    return [
        compiled_fns,
        state,
        params,
        rnd,
        x,
        saver,
        model,
    ]

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      model,
  ):
    """Generates the chain of samples."""

    score_fn, stp_fn = compiled_fns

    logz_finals = []
    log_w = jnp.zeros(self.config.batch_size)
    if self.parallel:
      log_w = log_w.reshape(x.shape[0], -1)

    for step in tqdm.tqdm(range(1, 1 + self.config.chain_length)):
      rng = jax.random.fold_in(rng, step)

      old_val = score_fn(params, x)
      if not self.parallel:
        params['temperature'] = step * 1.0 / self.config.chain_length
      else:
        params['temperature'] = jnp.repeat(
            step * 1.0 / self.config.chain_length, x.shape[0]
        )

      log_w = log_w + score_fn(params, x) - old_val
      if not self.parallel:
        rng_step = rng
      else:
        rng_step = jax.random.split(rng, x.shape[0])
      new_x, state, _ = stp_fn(
          rng=rng_step,
          x=x,
          model_param=params,
          state=state,
      )
      log_w_re = log_w.reshape(-1)
      logz_final = jax.scipy.special.logsumexp(log_w_re, axis=0) - np.log(
          self.config.batch_size
      )
      logz_finals.append(logz_final)
      x = new_x

    saver.save_logz(logz_finals)

class OG_Experiment:
  """Experiment class that generates chains of samples."""

  def __init__(self, config):
    self.config = config.experiment
    self.config_model = config.model
    self.parallel = False
    self.sample_idx = None
    self.num_saved_samples = config.get('nun_saved_samples', 4)
    if jax.local_device_count() != 1 and self.config.run_parallel:
      self.parallel = True

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    """Initializes model params, sampler state and gets the initial samples."""

    if self.config.evaluator == 'co_eval':
      sampler_init_state_fn = jax.vmap(sampler.make_init_state)
    else:
      sampler_init_state_fn = sampler.make_init_state
    model_init_params_fn = model.make_init_params
    rng_param, rng_x0, rng_state = jax.random.split(rnd, num=3)
    # params of the model
    params = model_init_params_fn(
        jax.random.split(rng_param,self.config.num_models) if self.config.num_models > 1 else rng_param
    )
    # initial samples
    num_samples = self.config.batch_size * self.config.num_models
    x0 = model.get_init_samples(rng_x0, num_samples)
    # initial state of sampler
    state = sampler_init_state_fn(
        jax.random.split(rng_state,self.config.num_models) if self.config.num_models > 1 else rng_state
    )
    return params, x0, state

  def _prepare_data(self, params, x, state):
    use_put_replicated = False
    reshape_all = True
    if self.config.evaluator != 'co_eval':
      if self.parallel:
        assert self.config.batch_size % jax.local_device_count() == 0
        mini_batch = self.config.batch_size // jax.local_device_count()
        bshape = (jax.local_device_count(),)
        x_shape = bshape + (mini_batch,) + self.config_model.shape
        use_put_replicated = True
        if self.sample_idx:
          self.sample_idx = jnp.array(
              [self.sample_idx]
              * (jax.local_device_count() // self.config.num_models)
          )
      else:
        reshape_all = False
        bshape = ()
        x_shape = (self.config.batch_size,) + self.config_model.shape
    else:
      if self.parallel:
        if self.config.num_models >= jax.local_device_count():
          assert self.config.num_models % jax.local_device_count() == 0
          num_models_per_device = (
              self.config.num_models // jax.local_device_count()
          )
          bshape = (jax.local_device_count(), num_models_per_device)
          x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
        else:
          assert self.config.batch_size % jax.local_device_count() == 0
          batch_size_per_device = (
              self.config.batch_size // jax.local_device_count()
          )
          use_put_replicated = True
          bshape = (jax.local_device_count(), self.config.num_models)
          x_shape = bshape + (batch_size_per_device,) + self.config_model.shape
          if self.sample_idx:
            self.sample_idx = jnp.array(
                [self.sample_idx]
                * (jax.local_device_count() // self.config.num_models)
            )
      else:
        bshape = (self.config.num_models,)
        x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
    fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    if reshape_all:
      if not use_put_replicated:
        state = jax.tree_map(fn_breshape, state)
        params = jax.tree_map(fn_breshape, params)
      else:
        params = jax.device_put_replicated(params, jax.local_devices())
        state = jax.device_put_replicated(state, jax.local_devices())
    x = jnp.reshape(x, x_shape)

    print('x shape: ', x.shape)
    print('state shape: ', state['steps'].shape)
    return params, x, state, bshape

  def _compile_sampler_step(self, step_fn):
    if not self.parallel:
      compiled_step = jax.jit(step_fn)
    else:
      compiled_step = jax.pmap(step_fn)
    return compiled_step

  def _compile_evaluator(self, obj_fn):
    if not self.parallel:
      compiled_obj_fn = jax.jit(obj_fn)
    else:
      compiled_obj_fn = jax.pmap(obj_fn)
    return compiled_obj_fn

  def _compile_fns(self, sampler, model, evaluator):
    if self.config.evaluator == 'co_eval':
      step_fn = jax.vmap(functools.partial(sampler.step, model=model))
      obj_fn = self._vmap_evaluator(evaluator, model)
    else:
      step_fn = functools.partial(sampler.step, model=model)
      obj_fn = evaluator.evaluate

    get_hop = jax.jit(self._get_hop)
    compiled_step = self._compile_sampler_step(step_fn)
    compiled_step_burnin = compiled_step
    compiled_step_mixing = compiled_step
    compiled_obj_fn = self._compile_evaluator(obj_fn)
    model_frwrd = jax.jit(model.forward)
    return (
        compiled_step_burnin,
        compiled_step_mixing,
        get_hop,
        compiled_obj_fn,
        model_frwrd,
    )

  def _get_hop(self, x, new_x):
    return (
        jnp.sum(abs(x - new_x))
        / self.config.batch_size
        / self.config.num_models
    )

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    raise NotImplementedError

  def vmap_evaluator(self, evaluator, model):
    raise NotImplementedError

  def preprocess(self, model, sampler, evaluator, saver, rnd_key=0):
    rnd = jax.random.PRNGKey(rnd_key)
    params, x, state = self._initialize_model_and_sampler(rnd, model, sampler)
    if params is None:
      print('Params is NONE')
      return False
    params, x, state, breshape = self._prepare_data(params, x, state)
    compiled_fns = self._compile_fns(sampler, model, evaluator)
    return [
        compiled_fns,
        state,
        params,
        rnd,
        x,
        saver,
        evaluator,
        breshape,
        model,
    ]

  def _get_chains_and_evaluations(
      self, model, sampler, evaluator, saver, rnd_key=0
  ):
    """Sets up the model and the samlping alg and gets the chain of samples."""
    preprocessed_info = self.preprocess(
        model, sampler, evaluator, saver, rnd_key=0
    )
    if not preprocessed_info:
      return False
    self._compute_chain(*preprocessed_info)
    return True

  def get_results(self, model, sampler, evaluator, saver):
    self._get_chains_and_evaluations(model, sampler, evaluator, saver)

class OG_Sampling_Experiment(OG_Experiment):
  """Class used to run classical graphical models and computes ESS."""

  def _vmap_evaluator(self, evaluator, model):
    obj_fn = evaluator.evaluate
    return obj_fn

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      bshape,
      model,
  ):
    """Generates the chain of samples."""
    assert self.config.num_models == 1
    (chain, acc_ratios, hops, running_time, samples) = (
        self._initialize_chain_vars()
    )
    fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    # sample used to map for ess computation
    rng_x0_ess, rng = jax.random.split(rng)
    x0_ess = model.get_init_samples(rng_x0_ess, 1)
    stp_burnin, stp_mixing, get_hop, obj_fn, model_frwrd = compiled_fns
    get_mapped_samples, eval_metric = self._compile_additional_fns(evaluator)
    rng = jax.random.PRNGKey(10)
    selected_chains = jax.random.choice(
        rng,
        jnp.arange(self.config.batch_size),
        shape=(self.num_saved_samples,),
        replace=False,
    )

    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    for step in tqdm.tqdm(range(1, burn_in_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      new_x, state, acc = stp_burnin(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      print(x[0])
      print(model_frwrd(params,x)[0])

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = stp_mixing(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      running_time += time.time() - start

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      mapped_sample = get_mapped_samples(new_x, x0_ess)
      mapped_sample = jax.device_put(mapped_sample, jax.devices('cpu')[0])
      chain.append(mapped_sample)
      x = new_x

    chain = jnp.array(chain)
    if self.parallel:
      chain = jnp.array([chain])
      rng = jnp.array([rng])
      num_ll_calls = int(state['num_ll_calls'][0])
    else:
      num_ll_calls = int(state['num_ll_calls'])
    ess = obj_fn(samples=chain, rnd=rng)
    metrics = eval_metric(ess, running_time, num_ll_calls)
    saver.save_results(acc_ratios, hops, metrics, running_time)
    if self.config.save_samples or self.config.get_estimation_error:
      if self.config.save_samples and self.config_model.name in [
          'rbm',
          'resnet',
      ]:
        saver.dump_samples(samples, visualize=False)
      elif (
          self.config.get_estimation_error
          and self.config_model.name == 'bernoulli'
      ):
        saver.dump_samples(samples, visualize=False)
        # samples= np.array(samples)
        params = params['params'][0].reshape(self.config_model.shape)
        saver.dump_params(params)
        # saver.plot_estimation_error(model, params, samples)
    saver.dump_params(params)

  def _initialize_chain_vars(self):
    chain = []
    acc_ratios = []
    hops = []
    samples = []
    running_time = 0

    return (
        chain,
        acc_ratios,
        hops,
        running_time,
        samples,
    )

  def _compile_additional_fns(self, evaluator):
    get_mapped_samples = jax.jit(self._get_mapped_samples)
    eval_metric = jax.jit(evaluator.get_eval_metrics)
    return get_mapped_samples, eval_metric

  def _get_mapped_samples(self, samples, x0_ess):
    samples = samples.reshape((-1,) + self.config_model.shape)
    samples = samples.reshape(samples.shape[0], -1)
    x0_ess = x0_ess.reshape((-1,) + self.config_model.shape)
    x0_ess = x0_ess.reshape(x0_ess.shape[0], -1)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)