
import numpy as np 


# -------------------------------------------------------------------------------
# Physics System Objects
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
        
        self.k1      = args[0]
        self.k2      = args[1]
        self.k3      = args[2]

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

        dy1 = -self.k1*y1 + self.k3*y2*y3
        dy2 =  self.k1*y1 - self.k3*y2*y3 - self.k2*y2**2
        dy3 =  self.k2*y2**2

        return np.array([dy1, dy2, dy3])