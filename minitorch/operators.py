"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers"""
    return float(x * y)


def id(x: float) -> float:
    """Returns the input unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers"""
    return x + y


def neg(x: float) -> float:
    """Negates a number"""
    return float(-x)


def lt(x: float, y: float) -> float:  #
    """Checks if one number is less than another"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:  #
    """Checks if two numbers are equal"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value"""
    # return float(abs(x - y) < 1e-2)
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# TODO: Implement for Task 0.3.

# Implement the following core functions
# - map
# - zipWith
# - reduce


# copy from answer
def map(func: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map
    Args :
    ----
        func: A function from value to value.
    Returns :
    -------
        A function that takes a list, applies `fn` to each element, and returns a new list.
    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(func(x))
        return ret

    return _map


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    return map(neg)(lst)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith (or map2).
    Args :
    ----
        fn : combine two values.
    Returns :
    -------
        Function that takes two equally sized lists `ls1` and `ls2`, produces a new list applying fn(x, y) to each pair of elements.
    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` together using zipWith"""
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.
    Args :
    ----
        fn : combine two values.
        start : start value $x_0$
    Returns :
    -------
        Function that takes a list `ls` of elements
        $[x_1, \ldots, x_n]$ and computes the reduction: math:`fn(x_3,fn(x_2,fn(x_1, x_0)))`
    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`"""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul"""
    return reduce(mul, 1.0)(ls)
