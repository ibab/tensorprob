from collections import defaultdict, Iterable
import itertools

import numpy as np
import tensorflow as tf
from six.moves import zip_longest


NAME_COUNTERS = defaultdict(lambda: 0)


def generate_name(obj):
    """Generate a unique name for the object in question

    Returns a name of the form "{calling_class_name}_{count}"
    """
    global NAME_COUNTERS

    calling_name = obj.__name__

    NAME_COUNTERS[calling_name] += 1
    return '{0}_{1}'.format(calling_name, NAME_COUNTERS[calling_name])


class classproperty(object):
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


def grouper(iterable, n=2, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def flatten(l):
    """Recursivly flattens a interable argument, ignoring strings and bytes.

    Taken from: http://stackoverflow.com/a/2158532
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def is_finite(obj):
    return isinstance(obj, tf.Tensor) or np.isfinite(obj)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
