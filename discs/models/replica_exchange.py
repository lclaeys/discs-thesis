from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections
import importlib

class ReplicaExchange(abstractmodel.AbstractModel):

    def __init__(self, config: ml_collections.ConfigDict):
        self.model_name = config['model_name']
        self.model_config = config.model_config
        self.max_temp = config.max_temp
        self.num_replicas = config.num_replicas
        
        model_mod = importlib.import_module('discs.models.%s' % self.model_name)
        self.base_model = model_mod.build_model(self.model_config)
        self.temp_multipliers = jnp.logspace(0,jnp.log10(self.max_temp),self.num_replicas)
   
    def make_init_params(self, rng):
        base_model_params = self.base_model.make_init_params(rng)
        return base_model_params

    def get_init_samples(self, rng, num_samples):
        samples = self.base_model.get_init_samples(rng, num_samples*self.num_replicas)

    def forward(self, params, x):
        N = x.shape[0]
        single_shape = (x.shape[1]//self.num_replicas)

        if len(x.shape) > 2:
            single_shape = single_shape + x.shape[2:]
        
        x_long = jnp.rehape(x,(N*self.num_replicas)+single_shape)
        
        forward_long = self.base_model.forward(params,x_long)
        forward_reshaped = jnp.reshape(forward_long, (N,self.num_replicas))

        ll = jnp.sum(forward_reshaped/jnp.reshape(self.temp_multipliers,(1,self.num_replicas)),axis=1)

        return ll

    def get_value_and_grad(self, params, x):
        x = x.astype(jnp.float32)  # int tensor is not differentiable

        def fun(z):
            loglikelihood = self.forward(params, z)
            return jnp.sum(loglikelihood), loglikelihood

        (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
        return loglikelihood, grad

    def 
    




    


