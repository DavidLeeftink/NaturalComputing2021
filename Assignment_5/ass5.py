# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

# %%

n_students = 31
maj = n_students//2 + 1
student_comp = 0.6

# 2b)
def binomial(n, k, p):
    return binom(n, k) * np.power(p, k) * np.power((1-p), n-k)


# probability of exactly s students to make the correct decision
students = [binomial(31, s, student_comp) for s in range(n_students+1)]

# Probability of decision by majority vote
maj_students = sum(students[maj:])

print(f'p of making the correct decision based on majority vote: {round(maj_students, 3)}')

# %%

# 2c)
various_sizes_of_jury = np.arange(1, 101, 1)
range_p = np.arange(0.5, 1.05, 0.05)
p_across_jury_size = []

p_vs_jury_size = np.zeros((len(various_sizes_of_jury), len(range_p)))
for size in various_sizes_of_jury:
    for i, p in enumerate(range_p):

        # Get the number of individuals that form the majority
        maj = size // 2 + 1

        # For each number of jury size, compute the probability that they are correct and sum;
        # Probability that the majority makes the correct decision
        maj_prob = sum([binomial(size, k, p) for k in range(maj, size+1)])

        # p_across_jury_size.append(maj_prob)
        p_vs_jury_size[size-1, i] = maj_prob

# Plot
fig, ax = plt.subplots(1,1, figsize=(14,7))
for i, p in enumerate(range_p):
    ax.plot(p_vs_jury_size[:, i], label=round(p, 3))
    ax.set_xlabel('Jury Size')
    ax.set_ylabel('P(correct decision)')
plt.legend()
# plt.savefig('2c.png')
plt.show()


# %%

# 2d)

radiologist = 0.85
three_docs = sum([binomial(3, k, 0.75) for k in range(2, 4)])
print(f'1 Radiologist: {radiologist}\n3 Docs: {round(three_docs, 2)}\n31 Students: {round(maj_students, 2)}')



# %%

# 3a)
# Here we need a poisson binomial distribution, I used a discrete Fourier transform implementation
ps = [0.6] * 10
ps.append(0.75)
n = len(ps)
maj = n//2+1

def poisson_binom(n, k, ps):
    C = np.exp(2j * np.pi / (n+1))
    return 1 / (n+1) * np.sum( [np.power(C, -l*k) * np.prod([1 + (np.power(C, l) - 1) * ps[m] for m in range(n)]) for l in range(n+1)])

p_ensemble_mix = sum([poisson_binom(n, k, ps).real for k in range(maj, n+1)])

print(f'The total of the classifiers is {round(p_ensemble_mix, 3)}, slightly better than the strong classifier by itself.')

# %%

# 3b)

def weighted_poisson_binom(n, k, ps, thr, w_weak, w_strong):
    norm = 1 / (n+1)
    C = np.exp(2j * np.pi / (n+1))
    summ = 0
    for l in range(n+1):
        prod = 1
        for m in range(n):
            
            w = w_weak if ps[m] < thr else w_strong # Here the weights are chosen and multiplied with the classifiers p
            prod *= 1 + (np.power(C, l) - 1) * ps[m] * w
        summ += np.power(C, -l*k) * prod
    return norm * summ

ws_strong = [w/n for w in range(1,50)]
w_weak = lambda w: (n - w) / n
all_poissons = [sum([weighted_poisson_binom(n, k, ps, ps[-1], w_weak(w_strong), w_strong).real for k in range(maj, n+1)]) for w_strong in ws_strong]

plt.plot(ws_strong, all_poissons)
plt.xlabel('Weights of the strong classifier')
plt.ylabel('P(correct decision)')
# plt.savefig('3b.png')
plt.show()

print(all_poissons.index(max(all_poissons)))
print(max(all_poissons))


# %%

# 3c)

errs = 1 - np.array(ps)
alpha = np.log((1 - errs) / errs)

alpha_poissons = [sum([weighted_poisson_binom(n, k, ps, ps[-1], 1, alpha[-1]).real for k in range(maj, n+1)])]
print(alpha_poissons)
print(alpha)


# %%

# 3d)
err = np.arange(0.1, 1.1, 0.1)
alpha = np.log((1-err) / err)
plt.plot(err, alpha)
plt.xlabel('error')
plt.ylabel('weights')
# plt.savefig('3d.png')
plt.show()

# %%
