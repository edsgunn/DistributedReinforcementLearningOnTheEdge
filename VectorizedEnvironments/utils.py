import numpy as np

@np.vectorize
def parameters_to_vector(parameters):
        vecs = []
        for (W,b) in parameters:
            vecs.append(W.flatten())
            vecs.append(b.flatten())
        return np.concatenate(vecs)

def vector_to_parameters(vector, parameter_shapes):
    parameters = []
    loc = 0
    for (W_shape, W_size, b_shape, b_size) in parameter_shapes:
        parameters.append((vector[loc:loc+W_size].reshape(W_shape),vector[loc+W_size:loc+W_size+b_size].reshape(b_shape)))
        loc += W_size + b_size
    return parameters

