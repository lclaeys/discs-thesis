"""Tests for Path Auxiliary Sampler."""

import copy
from absl.testing import absltest
from absl.testing import parameterized
import discs.models.bernoulli as bernouli_model
import discs.samplers.locallybalanced as lb_sampler
import discs.samplers.dlmc as dlmc
import jax
from ml_collections import config_dict


class DLMCTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.num_categories = 2
    self.sample_shape = (5, 4)
    self.config = config_dict.ConfigDict(dict(
        model=dict(
            shape=self.sample_shape, num_categories=self.num_categories,
            init_sigma=0.5, name='bernoulli',
        ),
        sampler=dict(
            name='dlmc', init_log_tau=0.0, solver='interpolate',
            adaptive=False, target_acceptance_rate=0.574,
            balancing_fn_type=lb_sampler.LBWeightFn.SQRT,
        )))
    self.model = bernouli_model.build_model(self.config)
    self.num_samples = 3

  def run_single_step(self, config, sampler):  # pylint: disable=g-unreachable-test-method
    num_samples = self.num_samples
    rng, state_rng, init_rng = jax.random.split(self.rng, 3)
    state = sampler.make_init_state(state_rng)
    param_rng, sample_rng = jax.random.split(init_rng)
    params = self.model.make_init_params(param_rng)
    x = self.model.get_init_samples(sample_rng, num_samples)
    x, state = sampler.step(self.model, rng, x, params, state)
    self.assertEqual(x.shape, tuple([num_samples] + list(config.model.shape)))

  def test_adaptive_binary(self):
    config = copy.deepcopy(self.config)
    config.sampler.adaptive = True
    sampler = dlmc.build_sampler(config)
    self.run_single_step(config, sampler)
