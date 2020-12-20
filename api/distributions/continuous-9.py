import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
plt.style.use('seaborn-darkgrid')
x = np.linspace(0, 5, 200)
for sigma in [0.4, 1., 2.]:
    pdf = st.halfnorm.pdf(x, scale=sigma)
    plt.plot(x, pdf, label=r'$\sigma$ = {}'.format(sigma))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc=1)
plt.show()