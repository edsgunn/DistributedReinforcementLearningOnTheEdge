import jax.numpy as jnp
import jax
from functools import partial
class Policy:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def act(self, state, parameters):
        pass

class DiscretePolicy(Policy):

    def __init__(self, input_size, output_size):
        super().__init__(input_size,output_size)

    @partial(jax.jit, static_argnames=['self', 'parameter_shapes'])
    def act(self, state, parameters, parameter_shapes):
        x = state.reshape(-1,1)
        loc = 0
        for (W_shape,b_shape) in parameter_shapes:
            W_size = W_shape[0]*W_shape[1]
            b_size = b_shape[0]*b_shape[1]
            W = jax.lax.dynamic_slice(parameters, (loc,1), (W_size,1)).reshape(W_shape)
            b = jax.lax.dynamic_slice(parameters, (loc+W_size,1), (b_size,1)).reshape(b_shape)
            loc += W_size + b_size
            x = jnp.tanh(W@x + b)
        return jnp.argmax(x)
    

class ContinousPolicy(Policy):
    def __init__(self, input_size, output_size):
        super().__init__(input_size,output_size)

    @partial(jax.jit, static_argnames=['self', "parameter_shapes"])
    def act(self, state, parameters, parameter_shapes):
        x = state.reshape(-1,1)
        loc = 0
        for (W_shape,b_shape) in parameter_shapes:
            W_size = W_shape[0]*W_shape[1]
            b_size = b_shape[0]*b_shape[1]
            W = jax.lax.dynamic_slice(parameters, (loc,1), (W_size,1)).reshape(W_shape)
            b = jax.lax.dynamic_slice(parameters, (loc+W_size,1), (b_size,1)).reshape(b_shape)
            loc += W_size + b_size
            x = jnp.tanh(W@x + b)
        return x.flatten()
    
