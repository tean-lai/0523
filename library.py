import numpy as np


class Discrete:
    @staticmethod
    def u(n):
        return np.where(np.array(n) >= 0, 1.0, 0.0)

    @staticmethod
    def d(n):
        return np.where(np.array(n) == 0, 1.0, 0.0)
    


class Continuous:
    @staticmethod
    def u(t):
        return np.where(np.array(t) >= 0., 1.0, 0.0)
    
    @staticmethod
    def integrate(x, a, b, precision=0.01):
        num_segments = np.ceil((a + b) / precision)
        times = np.linspace(a, b, (a + b) / precision)

        # np.cumsum could be useful


        pass

    @staticmethod
    def derivative(x, t):
        """wip, just wanted to write some potentially useful stuff here for a
        possible implementation in the future"""
        dxdt = np.gradient(x, t)
    

    