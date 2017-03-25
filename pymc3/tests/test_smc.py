import pymc3 as pm
import numpy as np
from pymc3.step_methods import smc
from tempfile import mkdtemp
import shutil
import theano.tensor as tt
import multiprocessing as mp

from .helpers import SeededTest


class TestSMC(SeededTest):

    def setUp(self):
        super(TestSMC, self).setUp()
        self.test_folder = mkdtemp(prefix='ATMIP_TEST')

    def test_sample_n_core(self, n_jobs=1):
        n_chains = 300
        n_steps = 100
        tune_interval = 25

        n = 4

        mu1 = np.ones(n) * (1. / 2)
        mu2 = -mu1

        stdev = 0.1
        sigma = np.power(stdev, 2) * np.eye(n)
        isigma = np.linalg.inv(sigma)
        dsigma = np.linalg.det(sigma)

        w1 = stdev
        w2 = (1 - stdev)

        def last_sample(x):
            return x[(n_steps - 1)::n_steps]

        def two_gaussians(x):
            log_like1 = - 0.5 * n * tt.log(2 * np.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            log_like2 = - 0.5 * n * tt.log(2 * np.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))

        with pm.Model() as ATMIP_test:
            X = pm.Uniform('X',
                           shape=n,
                           lower=-2. * np.ones_like(mu1),
                           upper=2. * np.ones_like(mu1),
                           testval=-1. * np.ones_like(mu1),
                           transform=None)
            like = pm.Deterministic('like', two_gaussians(X))
            llk = pm.Potential('like', like)

        with ATMIP_test:
            step = smc.SMC(
                n_chains=n_chains,
                tune_interval=tune_interval,
                likelihood_name=ATMIP_test.deterministics[0].name)

        mtrace = smc.ATMIP_sample(
            n_steps=n_steps,
            step=step,
            n_jobs=n_jobs,
            progressbar=True,
            stage='0',
            homepath=self.test_folder,
            model=ATMIP_test,
            rm_flag=True)

        d = mtrace.get_values('X', combine=True, squeeze=True)
        x = last_sample(d)
        mu1d = np.abs(x).mean(axis=0)

        np.testing.assert_allclose(mu1, mu1d, rtol=0., atol=0.03)

    def test_sample_multi_core(self):
        self.tearDown()
        cpus = mp.cpu_count()
        if cpus > 3:
            n_jobs = 4
        else:
            n_jobs = 2

        self.test_sample_n_core(n_jobs=n_jobs)

    def tearDown(self):
        shutil.rmtree(self.test_folder)
