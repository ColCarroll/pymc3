import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
plt.style.use('seaborn-darkgrid')
x = np.linspace(0, 8, 500)
nus = [0., 0., 4., 4.]
sigmas = [1., 2., 1., 2.]
for nu, sigma in  zip(nus, sigmas):
    pdf = st.rice.pdf(x, nu / sigma, scale=sigma)
    plt.plot(x, pdf, label=r'$\nu$ = {}, $\sigma$ = {}'.format(nu, sigma))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc=1)
plt.show()