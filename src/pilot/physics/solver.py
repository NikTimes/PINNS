from scipy.integrate import solve_ivp

# -------------------------------------------------------------------------------
# ODE Solver
# -------------------------------------------------------------------------------


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
