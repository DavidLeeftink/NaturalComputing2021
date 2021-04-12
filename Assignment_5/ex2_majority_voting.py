import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

###----------------###
##    Exercise 2    ##
###----------------###
def p_majority(c, p):
    """
    Computes the probability that a ensemble of c voters with each p chance at being correct
        gets the correct majority vote.
    :param c (int)
    :param p (float) probability at being correct for each member. 0<= p <= 1
    return: probability of correct ensemble majority vote (sum of Binomials)
    """
    m = int(np.floor(c/2)+1)
    return np.sum([comb(c,i)*p**i * (1-p)**(c-i) for i in np.arange(m,c+1)])


## 3 doctors (to confirm analytical results)
c_doctors = 3
p_doctors = 0.75
pred_doctors = p_majority(c_doctors, p_doctors)

## 31 students
c_students = 31
p_students = 0.6
pred_students = p_majority(c_students, p_students)

print(f"Probability of {c_doctors} doctors majority vote: {round(pred_doctors,3)}")
print(f"Probability of {c_students} doctors majority vote: {round(pred_students,3)}")


## Calculate as a function of ensemble size
c_vals = np.arange(1,100)
c_preds = np.array([p_majority(c_i, p_students) for c_i in c_vals])
plt.plot(c_vals, c_preds, label='all values')
plt.plot(c_vals[c_vals%2==1], c_preds[c_vals%2==1], label='odd values')
plt.plot(c_vals[c_vals%2==0], c_preds[c_vals%2==0], label='even values')
plt.axhline(y=0.85, color='black', linestyle=':', label='Expert prediction')
plt.ylabel('Probability of correct majority vote')
plt.xlabel('Number of students')
plt.legend()
plt.savefig('figures/A5_2c.png')
plt.show()

###----------------###
##    Exercise 3    ##
###----------------###
# def p_majority_unequal_probs(c_weak, c_strong, p_weak, p_strong, w_weak=1, w_strong=1)
#     """
#     Computes the probability that a ensemble of c voters with each p chance at being correct
#         gets the correct majority vote.
#     :param c (int)
#     :param p (float) probability at being correct for each member. 0<= p <= 1
#     return: probability of correct ensemble majority vote (sum of Binomials)
#     """
#     c = c_weak+c_strong
#     m = int(np.floor(c/2)+1)
#     return np.sum([comb(c,i)*p_weak**i * (1-p)**(c-i) for i in np.arange(m,c+1)])
