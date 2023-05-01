import numpy as np
from scipy.stats import norm, binom

class CommunicationScheme:
    def doCommunicate(epsilon, reward):
        pass


class ESCommunicationScheme(CommunicationScheme):

    def doCommunicate(self, epsilon,reward):
        return True

class ESPCCommunicationScheme(CommunicationScheme):
    def __init__(self, n, m, k):
        self. n = n
        self.m = m
        self.k = k
        self.past_magnitudes = []

    def doCommunicate(self, epsilon, reward):
        mag = reward*np.linalg.norm(epsilon)
        self.past_magnitudes.append(mag)
        mean = np.mean(self.past_magnitudes[-self.k:])
        sig = np.std(self.past_magnitudes[-self.k:])
        if sig == 0:
            sig = 1
        pNorm = norm.cdf((mag-mean)/sig)
        p = 1-binom.cdf(self.n-self.m, self.n-1, pNorm)
        return np.random.random() < p