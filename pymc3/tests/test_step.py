import unittest

from .checks import close_to
from .models import mv_simple, mv_simple_discrete, simple_2model
from pymc3.sampling import assign_step_methods, sample
from pymc3.model import Model
from pymc3.step_methods import (NUTS, BinaryGibbsMetropolis, Metropolis, Constant, Slice,
                                CompoundStep, MultivariateNormalProposal, HamiltonianMC)
from pymc3.distributions import Binomial, Normal, Bernoulli, Categorical
from numpy.testing import assert_almost_equal
import numpy as np


def check_stat(name, trace, var, stat, value, bound):
    s = stat(trace[var][2000:], axis=0)
    close_to(s, value, bound)


def test_step_continuous():
    start, model, (mu, C) = mv_simple()

    with model:
        slicer = Slice()
        hmc = HamiltonianMC(scaling=C, is_cov=True, blocked=False)
        nuts = NUTS(scaling=C, is_cov=True, blocked=False)

        mh_blocked = Metropolis(S=C,
                                proposal_dist=MultivariateNormalProposal,
                                blocked=True)
        slicer_blocked = Slice(blocked=True)
        hmc_blocked = HamiltonianMC(scaling=C, is_cov=True)
        nuts_blocked = NUTS(scaling=C, is_cov=True)

        compound = CompoundStep([hmc_blocked, mh_blocked])

    steps = [slicer, hmc, nuts, mh_blocked, hmc_blocked,
             slicer_blocked, nuts_blocked, compound]

    unc = np.diag(C) ** .5
    check = [('x', np.mean, mu, unc / 10.),
             ('x', np.std, unc, unc / 10.)]

    for st in steps:
        h = sample(8000, st, start, model=model, random_seed=1)
        for (var, stat, val, bound) in check:
            yield check_stat, repr(st), h, var, stat, val, bound


def test_non_blocked():
    """Test that samplers correctly create non-blocked compound steps.
    """

    start, model = simple_2model()

    with model:
        # Metropolis and Slice are non-blocked by default
        mh = Metropolis()
        assert isinstance(mh, CompoundStep)
        slicer = Slice()
        assert isinstance(slicer, CompoundStep)
        hmc = HamiltonianMC(blocked=False)
        assert isinstance(hmc, CompoundStep)
        nuts = NUTS(blocked=False)
        assert isinstance(nuts, CompoundStep)

        mh_blocked = Metropolis(blocked=True)
        assert isinstance(mh_blocked, Metropolis)
        slicer_blocked = Slice(blocked=True)
        assert isinstance(slicer_blocked, Slice)
        hmc_blocked = HamiltonianMC()
        assert isinstance(hmc_blocked, HamiltonianMC)
        nuts_blocked = NUTS()
        assert isinstance(nuts_blocked, NUTS)
        CompoundStep([hmc_blocked, mh_blocked])


def test_step_discrete():
    start, model, (mu, C) = mv_simple_discrete()

    with model:
        mh = Metropolis(S=C, proposal_dist=MultivariateNormalProposal)
        Slice()

    steps = [mh]
    unc = np.diag(C) ** .5
    check = [('x', np.mean, mu, unc / 10.),
             ('x', np.std, unc, unc / 10.)]
    for st in steps:
        h = sample(20000, st, start, model=model, random_seed=1)
        for (var, stat, val, bound) in check:
            yield check_stat, repr(st), h, var, stat, val, bound


def test_constant_step():
    with Model():
        x = Normal('x', 0, 1)
        start = {'x': -1}
        tr = sample(10, step=Constant([x]), start=start)
        assert_almost_equal(tr['x'], start['x'], decimal=10)


class TestAssignStepMethods(unittest.TestCase):
    def test_bernoulli(self):
        """Test bernoulli distribution is assigned binary gibbs metropolis method"""
        with Model() as model:
            Bernoulli('x', 0.5)
            steps = assign_step_methods(model, [])
        self.assertIsInstance(steps, BinaryGibbsMetropolis)

    def test_normal(self):
        """Test normal distribution is assigned NUTS method"""
        with Model() as model:
            Normal('x', 0, 1)
            steps = assign_step_methods(model, [])
        self.assertIsInstance(steps, NUTS)

    def test_categorical(self):
        """Test categorical distribution is assigned binary gibbs metropolis method"""
        with Model() as model:
            Categorical('x', np.array([0.25, 0.75]))
            steps = assign_step_methods(model, [])
        self.assertIsInstance(steps, BinaryGibbsMetropolis)

    # with Model() as model:
    #     x = Categorical('x', np.array([0.25, 0.70, 0.05]))
    #     steps = assign_step_methods(model, [])
    #
    #     assert isinstance(steps, ElemwiseCategoricalStep)

    def test_binomial(self):
        """Test binomial distribution is assigned metropolis method."""
        with Model() as model:
            Binomial('x', 10, 0.5)
            steps = assign_step_methods(model, [])
        self.assertIsInstance(steps, Metropolis)
