from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# The english and anomalous class files
# files = ['english.test', 'tagalog.test']

# English and the other files
others = ['hiligaynon', 'middle-english', 'plautdietsch', 'xhosa']
files = ['english.test', f'lang/{others[0]}.txt']

# Get the english words and make labels
with open(files[0]) as eng_test:
    english = eng_test.read().split()
labels = np.zeros(len(english))

# Get the other language and make labels
with open(files[1]) as other_test:
    other = other_test.read().split()
labels = np.append(labels, np.ones(len(other)))


fprs = []
tprs = []
auc_scores = []

for r in range(1, 10):

    anomaly_scores = np.array([])
    for f in files:
        with open(f) as infile:

            # Run the negative selection algorithm and save the anomaly scores
            process = Popen(['java', '-jar', 'negsel2.jar', '-self', 'english.train', '-n', '10', '-r', f'{r}', '-c', '-l'], stdin=infile, stdout=PIPE)
            stdout, _ = process.communicate()
            stdout = [float(e) for e in stdout.split()]
        anomaly_scores = np.append(anomaly_scores, stdout)
        

    # Compute the false positive and true positive rates
    fpr, tpr, thresholds = metrics.roc_curve(labels, anomaly_scores)
    fprs.append(fpr)
    tprs.append(tpr)

    # Compute the AUC score
    auc_score = roc_auc_score(labels, anomaly_scores)
    auc_scores.append(auc_score)

    print(f'AUC score with r={r}: {round(auc_score, 3)}')

# Plot
fig, ax = plt.subplots(3,3, figsize=(10,10))
ax = ax.ravel()

for i in range(9):
    ax[i].plot(fprs[i], tprs[i])
    ax[i].set_title(f'AUC score {round(auc_scores[i], 3)} with r = {i + 1}')
    if i >= 6:
        ax[i].set_xlabel('False Positive Rate')  
    if i % 3 == 0:
        ax[i].set_ylabel('True Positive Rate')
    ax[i].plot(np.linspace(0, 1, 100),
        np.linspace(0, 1, 100),
        label='baseline',
        linestyle='--')
plt.legend()
plt.show()

