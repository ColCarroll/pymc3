import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
plt.style.use('seaborn-darkgrid')
x = np.linspace(-2, 10, 500)
lowers = [0., -1, 2]
cs = [2., 0., 6.5]
uppers = [4., 1, 8]
for lower, c, upper in zip(lowers, cs, uppers):
    scale = upper - lower
    c_ = (c - lower) / scale
    pdf = st.triang.pdf(x, loc=lower, c=c_, scale=scale)
    plt.plot(x, pdf, label='lower = {}, c = {}, upper = {}'.format(lower,
                                                                   c,
                                                                   upper))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc=1)
plt.show()