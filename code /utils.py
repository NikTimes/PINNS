import numpy as np 
import matplotlib.pyplot as plt 

from scipy.integrate import solve_ivp
from scipy.stats import qmc


# -------------------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------------------

class harm_osc():
    """
    Represents a one-dimensional damped harmonic oscillator governed by
    a second-order ordinary differential equation.

    The system is defined by:
        x''(t) + c x'(t) + k x(t) = 0

    which is rewritten as a first-order system for numerical integration.

    Attributes:
        k (float): Spring (restoring) constant.
        c (float): Damping coefficient.
    """

    def __init__(self, args):
        """
        Initialize Harmonic Oscillator

        Args:
            args (array-like of length 2):
                args[0] (float): Spring constant k.
                args[1] (float): Damping coefficient c.
        """

        self.k = args[0]
        self.c = args[1]


    def derivative(self, t, y):
        """
        Compute the time derivative of the state vector.

        This method defines the system of first-order ODEs in the format
        required by ``scipy.integrate.solve_ivp``.

        State vector:
            y[0] = x(t)   : position
            y[1] = v(t)   : velocity

        The system is:
            x'(t) = v(t)
            v'(t) = -c v(t) - k x(t)

        Args:
            t (float):
                Current time (required by solve_ivp, not used explicitly
                since the system is autonomous).
            y (array-like of shape (2,)):
                Current state vector [position, velocity].

        Returns:
            numpy.ndarray of shape (2,):
                Time derivatives [dx/dt, dv/dt].
        """

        x, v = y
        dxdt = v
        dvdt = - self.k*x - self.c*v

        return np.array([dxdt, dvdt])
    
class Robertson():
    """
    Represents the Robertson model involving three chemical species
    A, B, C. Each of which have concentrations denoted by y1, y2, y3 
    respectively. Sum of these concentrations should always add up to one. 

    y1 + y2 + y3 = 1

    The reactions between these species are: 

    A --> B
    B + C --> A + C
    B + B --> C

    The rate equations governing the evolution of the concentrations can 
    be written by:

    y1' = -k1y1 + k2y2y3
    y2' =  k1y1 - k2y2y3 - k3(y2**2)
    y3' =  k3(y2**2)

    Where the k's are the reaction rates. The Class Stores these as attributes

    Attributes:
        k1 (float): reaction rate of y1
        k2 (float): reaction rate of y2
        k3 (float): reaction rate of y3
    
    """
    def __init__(self, args):
        """
        Initialize Robertson Model

        Args:
            args (array-like of length 2):
                args[0] (float): reaction rate of y1
                args[1] (float): reaction rate of y2
                args[2] (float): reaction rate of y3
        """
        
        self.k1 = args[0]
        self.k2 = args[1]
        self.k3 = args[2]

    def derivative(self, t, y):
        """
        Compute the time derivative of the state vector.

        This method defines the system of first-order ODEs in the format
        required by ``scipy.integrate.solve_ivp``.

        State vector:
            y[0] = y1(t)   : concentration of A
            y[1] = y2(t)   : concentration of B
            y[2] = y3(t)   : concentration of C

        The system is:
            y1' = -k1y1 + k2y2y3
            y2' =  k1y1 - k2y2y3 - k3(y2**2)
            y3' =  k3(y2**2)

        Args:
            t (float):
                Current time (required by solve_ivp, not used explicitly
                since the system is autonomous).
            y (array-like of shape (3,)):
                Current state vector [concentration of A, concentration of B, concentration of C].
                Note the sum of all of its components must be equal to 1!

        Returns:
            numpy.ndarray of shape (2,):
                Time derivatives [dy1/dt, dy2/dt, dy3/dt].
        """

        y1, y2, y3 = y

        dy1 = -self.k1*y1 + self.k2*y2*y3
        dy2 =  self.k1*y1 - self.k2*y2*y3 - self.k3*y2**2
        dy3 =  self.k3*y2**2

        return np.array([dy1, dy2, dy3])


class ODEsolver():
    """
    This is a generic ODE wrapper object for solving systems of ordinary differential equations
    using ``scipy.integrate.solve_ivp``
    
    This class stores a dynamical system object that provides a ``derivative(t, y)``
    method in the format required by ``scipy.integrate.solve_ivp``

    Attributes: 
        system (object)        : Dynamical system providing a derivative(t, y) method.
        derivatives (collable) : Reference to ``system.derivative``

    """
    
    def __init__(self, system):
        """
        Initialize the ODE solver.

        Args:
            system (object): 
                Dynamical system providing a derivative(t, y) with a ``derivative(t, y)``
                method. 
        """
        self.system      = system
        self.derivatives = system.derivative
    
    def solve(self, t_span, y0, method="RK45", t_eval= None, dense_output=False, vectorized=False):
        """
        This method takes most arguments from solve_ivp to solve 
        the system of ODE's attribute of the class

        Args:
            t_span (float tuple)         : time interval of integration (t0, tf)
            y0 (array-like)              : Initial state of the system
            method (str, optional)       : Integration method. Defaults to "RK45".
            t_eval (array, optional)     : Times at whoch to store computed solution. Defaults to None.
            dense_output (bool, optional): Whether to compute continuous solution. Defaults to False.
            vectorized (bool, optional)  : Whether the derivative function is vectorized. Defaults to False.

        Returns:
           object : solve_ivp solution object
        """
        sol = solve_ivp(
                self.derivatives, 
                t_span=t_span,
                y0 = y0,
                method=method,
                t_eval=t_eval,
                dense_output=dense_output, 
                vectorized=vectorized,
                rtol=1e-6,
                atol=1e-9,
            )

        return sol

# -------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------

def initial_sampler(num_samples, dimension):
    """
    Generate initial conditions using Latin Hypercube Sampling (LHS).

    This function draws samples from the unit hypercube [0, 1]^dimension,
    which can be rescaled externally to match the desired domain for
    ODE initial conditions.

    Args:
        num_samples (integer):
            Number of samples to generate.
        dimension (integer):
            Dimensionality of each sample (i.e. number of state variables).

    Returns:
        numpy.ndarray of shape (num_samples, dimension):
            Array of sampled initial conditions in the unit hypercube.
    """
    
    sampler = qmc.LatinHypercube(d=dimension)
    samples = sampler.random(num_samples)

    return samples