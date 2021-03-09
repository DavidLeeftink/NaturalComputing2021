from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics


def preprocess(train_file, out_file, k):
    """
    Takes each line from train_file and creates chunks of substrings
    of size k and writes them to out_file
    """
    k = 7
    with open(train_file, 'r') as data_train:
        lines = data_train.readlines()
        with open(out_file, 'w') as out:
            for line in lines:
                print(line, len(line))
                for i in range(len(line)-k): 
                    out.write(f'{line[i:i+k]}\n')



train_file = 'syscalls/snd-cert/snd-cert.train'
preprocessed_train = 'syscalls/snd-cert/snd-cert-preprocessed.train'
test_files = ['syscalls/snd-cert/snd-cert.1.test', 'syscalls/snd-cert/snd-cert.2.test', 'syscalls/snd-cert/snd-cert.3.test']
label_files = ['syscalls/snd-cert/snd-cert.1.labels', 'syscalls/snd-cert/snd-cert.2.labels', 'syscalls/snd-cert/snd-cert.3.labels']
k = 7
# preprocess(train_file, preprocessed_train, k)


process = Popen(['java', '-jar', 'negsel2.jar', '-alphabet', 'file://syscalls/snd-cert/snd-cert.alpha', '-self', f'{preprocessed_train}', '-n', f'{k}', '-r', '4', '-c', '-l'], stdin=PIPE, stdout=PIPE, stderr=PIPE)

# for i in range(len(test_files)):
i = 2
anomaly_scores = []
with open(test_files[i], 'r') as test:
    lines = test.readlines()

    # Cut to chunks
    for line in lines:
        substrings = [line[i:i+k] for i in range(len(line)-k)]
        
        # Compute average anomaly score per chunk
        anom_score = 0
        for sub in substrings:
            process.stdin.write(f'{sub}\n'.encode('utf-8'))
            process.stdin.flush()
            out = process.stdout.readline().decode("utf-8").strip()
            anom_score += float(out)
        anomaly_scores.append(anom_score/len(substrings))
        
        print(anomaly_scores[-1])

with open(label_files[i]) as labels_file:
    labels = labels_file.read().split()
    labels = np.array(labels, dtype=np.int)

print(len(labels), len(anomaly_scores))

# Compute the false positive and true positive rates
fpr, tpr, thresholds = metrics.roc_curve(labels, anomaly_scores)

# Plot
plt.plot(fpr, tpr)
plt.plot(np.linspace(0, 1, 100),
        np.linspace(0, 1, 100),
        label='baseline',
        linestyle='--')
plt.legend()
plt.show()

# Compute the AUC score
auc_score = roc_auc_score(labels, anomaly_scores)

print(f'AUC score: {round(auc_score, 3)}')

# Close the process    
process.stdin.close()
process.terminate()
process.wait(timeout=0.2)




        
            






    


# Find the shortest string
# n = len(min(strings, key=len))
# print()
# substrings = [string[i:i+k] for i in range(len(string)-(k-1))]


# 1. Open a process in which I can indefinitely insert calls
# 2. Open the test file
# 3. Use readlines() to read each line. 
# 4. Chunk each line and feed each substring to the program 
# to get an anomaly score for each substring
# 5. Average and put the result in a list

# with r = 4 and n = 7
# snd-cert test1 got AUC: 0.98
# snd-cert test2 got AUC: 0.945
# snd-cert test2 got AUC: 0.956