"""
This module implements a generic Runge-Kutta solver.
"""
class Vector:
    """
    A simple vector class.

    Attributes
    ----------
    _components: list
        Stores the components of the vector.
    _len: int
        Stores the length of the vector.
    """
    def __init__(self, components):
        """
        Initialize a vector by specifying its components.
        
        Parameters
        ----------
        components: list
            The components of the vector.
        """
        self._components = components
        self._len = len(components)        

    def __add__(self, other):
        """
        Add two vectors.

        Parameters
        ----------
        other: Vector
            The vector to add to `self`.

        Returns
        -------
        Vector
            The sum of `self` and `other`.

        Examples
        --------
        >>> v1 = Vector([1,2,3])
        >>> v2 = Vector([4,5,6])
        >>> v3 = v1 + v2
        >>> v3._components == [5, 7, 9]
        True
        """
        assert(self._len == other._len)
        return Vector([x+y for x,y in zip(self._components, other._components)])

    def __call__(self):
        """
        Get the underlying list o components.

        Return
        ------
        The list of components.

        Examples
        --------
        >>> v = Vector([1, 2, 3])
        >>> v() == [1, 2, 3]
        True
        """
        return self._components

    def __mul__(self, other):
        """
        Multiply the vector by other.

        Examples
        --------
        >>> v1 = Vector([1, 2, 3])
        >>> v2 = v1 * 2
        >>> v2._components == [2, 4, 6]
        True
        """
        if isinstance(other, (float, int)):
            return self._multiply_scalar(other)
        else:
            raise TypeError("Good unknown type in multiplication.")


    def _multiply_scalar(self, other):
        """
        Multiply the vector by a scalar `other`.

        Return
        ------
        Vector
            A vector whose components correspond to those of `self` multiplied by `other`.
        """
        return Vector([x*other for x in self._components])


def runge_kutta(f, x0, t_start, t_end, delta_t):
    """
    The Runge-Kutta method for `n` dimensional (non-linear) systems.

    Parameters
    ----------
    f: function
        A list containing the entries of the vector-valued function `f`.
    x0: list
        Contains the initial values of the system.
    t_start: float
        The starting time.
    t_end: float
        The ending time.
    delta_t: float
        The time step used when solving the system.

    Return
    ------
    tuple
        The first entry is a list containing all the employed time steps.
        The second entry is a list, containing the corresponding at each time step.
    """
    t = []
    x = []
    t_last = t_start
    x_last = x0
    while t_start < t_end:
    
        k1 = f(x_last) * delta_t
        k2 = f(x_last + k1 * 0.5) * delta_t
        k3 = f(x_last + k2 * 0.5) * delta_t
        k4 = f(x_last + k3) * delta_t

        x_last = x_last + 1. / 6. * (k1 + k2 * 2. + k3 * 2 + k4)
        x.append(x_last())
        t_last = t_last + delta_t
        t.apped(delta_t)

    return t_last, x_last


if __name__ == "__main__":
    import doctest
    doctest.testmod()
