import numpy as np

class UpdateScheme:
    def __init__(self):
        pass

    def update(self, rewards, perturbations):
        pass

    def grad(self,rewards,perturbations):
        return 1/(len(rewards)*self.sigma) * np.sum([reward*perturbation for reward,perturbation in zip(rewards,perturbations)], axis=0)

class ESMomentum(UpdateScheme):
    def __init__(self, learning_rate, sigma, num_parameters, momentum=0.9, l2=0.05):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2 = l2
        self.sigma = sigma
        self.velocity = np.zeros(num_parameters)

    def update(self, rewards, perturbations, parameters):
        if len(rewards) > 0:
            grad = self.grad(rewards,perturbations) - self.l2 * parameters
            self.velocity = self.momentum * self.velocity + (1-self.momentum) * grad
            update = self.learning_rate * self.velocity

            return parameters + update
        else:
            return parameters
        
class ESAdam(UpdateScheme):
    def __init__(self, learning_rate, sigma, num_parameters, beta1=0.9, beta2=0.999, epsilon=1e-08, l2=0.05):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.l2 = l2
        self.sigma = sigma
        self.t = 1
        self.m = np.zeros((num_parameters,1))
        self.v = np.zeros((num_parameters,1))

    def update(self, rewards, perturbations, parameters):
        if len(rewards) > 0:
            grad = self.grad(rewards,perturbations) - self.l2 * parameters

            a = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
            update = a * self.m / (np.sqrt(self.v) + self.epsilon)
            self.t +=1
            return parameters + update
        else:
            return parameters
        