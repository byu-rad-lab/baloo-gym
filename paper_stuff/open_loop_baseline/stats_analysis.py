import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

# Parameters for two binomial distributions
n1, p1 = 100, 0.98
n2, p2 = 100, 0.89

# Range of k values (possible number of successes)
k = np.arange(0, 101)

# PMF values for both distributions
pmf1 = binom.pmf(k, n1, p1)
pmf2 = binom.pmf(k, n2, p2)

print(np.sum(pmf1))
print(np.sum(pmf2))

# Set up the figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# PMF comparison using bar plot
axs[0].bar(k[60:],
           pmf1[60:],
           width=0.8,
           color='b',
           label=f'B({n1}, {p1})',
           alpha=0.6)
axs[0].bar(k[60:],
           pmf2[60:],
           width=0.8,
           color='r',
           label=f'B({n2}, {p2})',
           alpha=0.6)
axs[0].set_xlabel('Number of Successes')
axs[0].set_ylabel('Probability')
axs[0].set_title('Probability Mass Function (PMF) Comparison')
axs[0].legend()
axs[0].grid(True)

# CDF comparison using step plot (discrete)
axs[1].step(k,
            binom.cdf(k, n1, p1),
            where='post',
            color='b',
            label=f'B({n1}, {p1})')
axs[1].step(k,
            binom.cdf(k, n2, p2),
            where='post',
            color='r',
            label=f'B({n2}, {p2})')
axs[1].set_xlabel('Number of Successes')
axs[1].set_ylabel('Cumulative Probability')
axs[1].set_title('Cumulative Distribution Function (CDF) Comparison')
axs[1].legend()
axs[1].grid(True)

# Difference in PMFs (Bar plot)
axs[2].bar(k,
           pmf1 - pmf2,
           width=0.8,
           color='g',
           label='Difference (PMF1 - PMF2)',
           alpha=0.6)
axs[2].set_xlabel('Number of Successes')
axs[2].set_ylabel('Difference in Probability')
axs[2].set_title('Difference Between Binomial Distributions (PMFs)')
axs[2].legend()
axs[2].grid(True)

# Layout adjustments
plt.tight_layout()
plt.show()

import scipy.stats as stats

# Uncertainty in success probability using book equation
sigma_1 = np.sqrt(p1 * (1 - p1) / n1)
sigma_2 = np.sqrt(p2 * (1 - p2) / n2)

print(f"sigma_1: {sigma_1}")
print(f"sigma_2: {sigma_2}")

#test whether the two distribution means are significantly different
from statsmodels.stats.proportion import proportions_ztest

# Perform the Z-test
stat, p_value = proportions_ztest([p1*n1, p2*n2], [n1, n2])

print(f"Z-statistic: {stat}")
print(f"P-value: {p_value}")

# Decision based on p-value
if p_value < 0.05:
    print(
        "Reject the null hypothesis: The proportions are significantly different."
    )
else:
    print(
        "Fail to reject the null hypothesis: The proportions are not significantly different."
    )
