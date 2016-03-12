import tensorflow as tf
import numpy as np

class Parameterization(object):
    def __init__(self, shape=(), session=None):
        self._shape = shape
        if session is None:
            self._session = tf.get_default_session()
        else:
            self._session = session
        self._transformed = self.transform(self._inner)

    def transform(self, inner):
        return NotImplementedError

    def jacobian_log_det(self):
        return NotImplementedError

    @property
    def value(self):
        return self._session.run(self._transformed)

    @value.setter
    def value(self, other):
        raise NotImplementedError

class Id(Parameterization):
    '''Identity parameterization

    Inner and outer representation are identical
    '''
    def __init__(self, shape=(), **kwargs):
        self._inner = tf.Variable(np.zeros(shape=shape, dtype=tf.float64.as_numpy_dtype()))
        self._feed = tf.placeholder(dtype=tf.float64)
        self._setter = self._inner.assign(self._feed)
        super(Id, self).__init__(shape, **kwargs)
    
    def transform(self, x):
        return x

    def jacobian_log_det(self):
        return tf.constant(0, dtype=tf.float64)
    
    @Parameterization.value.setter
    def value(self, other):
        if self._shape != other.shape:
            raise ValueError("Parameterization set with invalid shape: {} and {}".format(self._shape, other.shape))
        self._session.run(self._setter, feed_dict={self._feed: other})

def logit(x):
    return tf.log(np.float64(x) / (np.float64(1) - x))

def inverse_logit(x):
    return np.float64(1)/(np.float64(1) + tf.exp(-x))

def cumsum(xs):
    values = tf.unpack(xs)
    out = []
    prev = tf.zeros_like(values[0])
    for val in values:
        s = prev + val
        out.append(s)
        prev = s
    result = tf.pack(out)
    return result

def cumprod(xs):
    values = tf.unpack(xs)
    out = []
    prev = tf.ones_like(values[0])
    for val in values:
        s = prev * val
        out.append(s)
        prev = s
    result = tf.pack(out)
    return result

class Ordered(Parameterization):
    '''Ordered parameterization

    Values x_i for which x_i < x_(i+1)
    '''

    def __init__(self, shape=(), **kwargs):
        if len(shape) != 1:
            raise ValueError("Ordered must be a vector")
        if shape[0] == 0:
            raise ValueError("Ordered needs at least one element")
        self._length = shape[0]
        self._inner = tf.Variable(np.zeros(self._length, dtype=np.float64))
        self._feed = tf.placeholder(dtype=tf.float64)
        self._setter = self._inner.assign(self._feed)
        super(Ordered, self).__init__(shape, **kwargs)

    def transform(self, x):
        return x[0] + tf.concat(0, [[np.float64(0)], cumsum(tf.exp(x[1:]))])

    @Parameterization.value.setter
    def value(self, x):
        if np.any(np.diff(x) < 0):
            raise ValueError("Parameter to Ordered is not ordered: {}".format(x))
        y = np.concatenate([[x[0]], np.log(np.diff(x))])
        self._session.run(self._setter, feed_dict={self._feed: y})

    def jacobian_log_det(self):
        return tf.reduce_sum(self._inner[1:])

class Simplex(Parameterization):
    '''Simplex parameterization
    
    K values that always sum to 1, internally represented by K - 1 variables.
    '''
    def __init__(self, shape=(), **kwargs):
        if len(shape) != 1:
            raise ValueError("Simplex must be a vector")
        if shape[0] == 0:
            raise ValueError("Simplex needs at least one element")
        self._length = shape[0] - 1
        if self._length == 0:
            # As TensorFlow won't allow us to initialize a variable with length
            # zero and throws an error on run, we make the simplex of length 1
            # a special case
            self._inner = None
        else:
            self._inner = tf.Variable(np.zeros(self._length, dtype=np.float64))
        self._feed = tf.placeholder(dtype=tf.float64)
        self._setter = self._inner.assign(self._feed)
        super(Simplex, self).__init__(shape, **kwargs)

    def transform(self, x):
        if self._length == 0:
            return tf.constant([1], dtype=tf.float64)
        j = tf.cast(tf.range(0, self._length), tf.float64)
        fix = - tf.log(self._length - j)
        z = inverse_logit(self._inner + fix)
        yl = tf.concat(0, [z, [np.float64(1)]])
        yu = tf.concat(0, [[np.float64(1)], 1 - z])
        S = cumprod(yu)
        self._saved = S*yl
        return self._saved

    @Parameterization.value.setter
    def value(self, x):

        # This is the TensorFlow implementation for this,
        # in case we need it later
        #x = np.array(x).astype(np.float64)
        #x0 = x[:-1]
        #s = tf.reverse(cumsum(tf.reverse(x0, [True])), [True]) + x[-1]
        #z = x0 / s
        #Km1 = np.float64(self._length)
        #k = tf.cast(tf.range(0, Km1), tf.float64)
        #eq_share = - tf.log(Km1 - k)
        #y = logit(z) - eq_share
        #y_val = self._session.run(y)

        if np.array(x).shape != (self._length + 1,):
            raise ValueError("Simplex must be a vector of length {}".format(self._length + 1))

        if not np.isclose(np.sum(x), 1):
            raise ValueError("Simplex values don't sum to 1.")

        x0 = x[:-1]
        s = np.cumsum(x0[::-1])[::-1] + x[-1]
        z = x0 / s
        Km1 = self._length
        k = np.arange(Km1)
        eq_share = - np.log(Km1 - k)
        y = np.log(z / (1 - z)) - eq_share
        print('y', y)
        self._session.run(self._setter, feed_dict={self._feed: y})

    def jacobian_log_det(self):
        k = tf.cast(tf.range(0, self._length), tf.float64)
        eq_share = -T.log(self._length - k) 
        yl = y + eq_share
        z = inverse_logit(self._inner + fix)
        yu = tf.concat(0, [[np.float64(1)], 1 - z])
        S = cumprod(yu)
        return reduce_sum(tf.log(S[:-1]) - tf.log(1 + tf.exp(yl)) - tf.log(1 + tf.exp(-yl)))

# TODO(ibab) Add covariance matrix once tensorflow/tensorflow#1465 is merged

s = tf.Session()
X = np.random.uniform(0, 1, size=(10000, 1000))
v = tf.Variable(X)
s.run(tf.initialize_all_variables())
y = cumsum(v)
print('x')
print(s.run(y))
print(np.cumsum(X, axis=0))

#para = Ordered(shape=(4,), session=s)
#s.run(tf.initialize_all_variables())
#print(para.value)
#print(s.run(para._inner))
#para.value = [1,1.01,1.0,2]
#print(para.value)
#print(s.run(para._inner))
#para.value = [1, 1.00001, 1.000011, 1.2]
#print(para.value)
#print(s.run(para._inner))

#for i, (a, b) in enumerate(zip([0, 0.2, 0.5, 0.3, 0.1], [0, 0., 0.5, 0.3, 0.1])):
#    print('grad', s.run(tf.gradients(para._transformed, para._inner)))
#    print(np.sum([a, b, 1 - a - b]))
#    print('setting to ', a, b, 1 - a - b)
#    para.value = [a, b, 1 - a - b]
#    print('inner', s.run(para._inner))
#    print(i, para.value, np.sum(para.value))

