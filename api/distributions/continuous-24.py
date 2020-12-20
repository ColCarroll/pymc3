import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
plt.style.use('seaborn-darkgrid')
x = np.linspace(-10, 10, 1000)
mus = [0.,  0., 0.]
sigmas = [3.,5.,7.]
a1 = [-3, -5, -5]
b1 = [7, 5, 4]
for mu, sigma, a, b in zip(mus, sigmas,a1,b1):
    an, bn = (a - mu) / sigma, (b - mu) / sigma
    pdf = st.truncnorm.pdf(x, an,bn, loc=mu, scale=sigma)
    plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, a={}, b={}'.format(mu, sigma, a, b))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc=1)
plt.show()