import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
plt.style.use('seaborn-darkgrid')
x = np.arange(1, 15)
N = 50
k = 10
for n in [20, 25]:
    pmf = st.hypergeom.pmf(x, N, k, n)
    plt.plot(x, pmf, '-o', label='n = {}'.format(n))
plt.plot(x, pmf, '-o', label='N = {}'.format(N))
plt.plot(x, pmf, '-o', label='k = {}'.format(k))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc=1)
plt.show()