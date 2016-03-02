
import numpy as np
from tensorprob import Model, Normal, Parameter, fisher

def test_fisher():
    with Model() as model:

        mu = Parameter()
        sigma = Parameter(lower=0)
        X = Normal(mu, sigma)

    model.observed(X)
    model.initialize({
        mu: 2,
        sigma: 2,
    })
    np.random.seed(0)
    xs = np.random.normal(0, 1, 200)
    model.fit(xs)
    cov = fisher(model)
    # TODO(ibab) make this test more robust and useful
    # The result I got when running this
    assert np.isclose(cov[mu][mu], 0.00521652579559)

