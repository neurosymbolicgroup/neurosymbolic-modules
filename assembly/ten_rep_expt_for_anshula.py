import numpy as np
from assembly_classifier import AssemblyClassifier

train_acts = np.load('data/new_representations/binary_digits_binary_pixels/train_activations.npy', allow_pickle=True).item()
test_acts = np.load('data/new_representations/binary_digits_binary_pixels/test_activations.npy', allow_pickle=True).item()
x_train, y_train = train_acts['fc2'], np.load("data/old_representations/binary_digits_all_pixels/y_train.npy")
x_test, y_test = test_acts['fc2'], np.load("data/old_representations/binary_digits_all_pixels/y_test.npy")

d, p, reps = x_train.shape[1], 0.1, 10
results = {'pre_train':[], 'post_train':[]}

for r in range(reps):
    AC = AssemblyClassifier(d, n_assembly=1000, n_cap=41, edge_prob=p, initial_projection=True)
    # AC = AssemblyClassifier(d, n_cap=27, initial_projection=False)
    correct = AC.accuracy(x_test, y_test)
    total = len(y_test)
    results['pre_train'].append(100.*correct/total)
    n_train = 1000
    AC.train(x_train[:n_train], y_train[:n_train])
    correct = AC.accuracy(x_test, y_test)
    total = len(y_test)
    results['post_train'].append(100.*correct/total)
    if (r+1) % 10 == 0:
        print(r+1)

import matplotlib.pyplot as plt
plt.hist(results['post_train']); plt.title('$p = %.2f$'%(p,)); plt.xlabel('accuracy'); plt.ylabel('number of experiments'); plt.show()
