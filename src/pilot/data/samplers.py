from scipy.stats import qmc
import numpy as np

# -------------------------------------------------------------------------------
# Samplers 
# -------------------------------------------------------------------------------


class LatinHypercubeSampler:
    """
    This class is generates initial conditions 
    using Latin Hypercube Sampling (LHS).

    Args:
        dimensions (integer):
            Number of samples to generate.
        lows (array-type):
            Defines the lower bounds of each subsequent sample.
        highs (array-type):
            Defines the higher bound of each subsequent sample
    """

    def __init__(self, dimensions, lows=None, highs=None):
        """
        Initializes LatinHypercube Class. 
        This method also ensures that samples range
        between [0, 1] in the case where no lower or upper
        bounds are defined in the initialization. 

        Args:
            dimensions (integer):
                Number of samples to generate.
            lows (array-type):
                Defines the lower bounds of each subsequent sample.
                Default is None.
            highs (array-type):
                Defines the higher bound of each subsequent sample
                Default is None.
        """

        self.dimension = dimensions
        self.low       = np.zeros(dimensions) if lows is None else np.array(lows)
        self.high      = np.ones(dimensions) if highs is None else np.array(highs)
    
    def __call__(self, num_samples):
        """
        Calls the LHS sampler and draws the specified
        number of samples in the arguments. It also scales
        the samples based on the lower and upper bounds specified
        during initialization. 

        Args:
            num_samples (integer): 
                Number of Samples 

        Returns:
            array_like (num_samples, dimensions)
                Scaled samples
        """

        sampler = qmc.LatinHypercube(d=self.dimension)
        samples = sampler.random(num_samples)

        return qmc.scale(samples, self.low, self.high)


class DirichletSampler:
    """
    Initial Condition Sampler based on Dirichlet Distribution. 
    
    Often denoted Dir(alpha). Samples x_i in Dir(alpha) satisfy 
    the following properties: 

    x_i > 0 for all i and x_0 + x_1 + ... + x_n = 1.

    In other words all samples are positive and sum up to one. 

    Args: 
        alpha: sequence of floats. Concentration parameter. 
        It must contain as many elements as values needed per sample. 
        Fill array with ones for uniform distribution. 
    """

    def __init__(self, alpha=[1, 1, 1]):
        """
        Initializes Dirichlet Sampler.

        Args:
            alpha (sequence of floats): 
                Concentration parameter. 
                It must contain as many elements as values needed per sample. 
                Fill array with ones for uniform distribution. 
        """
        
        self.alpha = alpha

    def __call__(self, num_samples):
        """
        Calls the Dirichlet sampler. 

        Args:
            num_samples (integer): 
                Number of samples
        """

        return np.random.dirichlet(alpha=self.alpha, size=num_samples) 



class ConstrainedLHCSampler:
    """
    Samples 3-species system with constraints:

    x1 in [0.98, 1]
    x2 = 0
    x3 = 1 - x1

    Uses 1D Latin Hypercube sampling internally.
    """

    def __init__(self, low=0.98, high=1.0):
        self.low = low
        self.high = high
        self.sampler = qmc.LatinHypercube(d=1)

    def __call__(self, num_samples):
        """
        Generate samples.

        Args:
            num_samples (int): number of samples
        """

        # LHS in [0,1]
        u = self.sampler.random(n=num_samples).flatten()

        # Map to [low, high]
        x1 = self.low + (self.high - self.low) * u

        # Apply constraints
        x2 = np.zeros_like(x1)
        x3 = 1.0 - x1

        return np.stack([x1, x2, x3], axis=1)