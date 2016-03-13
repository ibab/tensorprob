import tensorflow as tf
import numpy as np
from functools import wraps
import sys

def PyFunc(*args):
    output_spec = args

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if sys.version_info.major >= 3:
                kwnames = func.__code__.co_varnames[len(args):]
            else:
                kwnames = func.func_code.varnames[len(args):]

            extra = []
            for w in kwnames:
                if w in kwargs:
                    extra.append(kwargs[w])
            op =  tf.py_func(func, list(args) + extra, output_spec, name=func.__name__)
            if len(output_spec) == 1:
                # Remove list if there's only one output
                return op[0]
            else:
                return op
        return wrapper
    return decorator

@PyFunc(tf.float64)
def normal_rvs(mu, sigma, size=()):
    return np.random.normal(mu, sigma, size=size)

s = tf.Session()
mu = tf.Variable(0.)
sigma = tf.Variable(1.)

X = normal_rvs(mu, sigma, size=100)
s.run(tf.initialize_all_variables())
print(X)
print(s.run(X))

