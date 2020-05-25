import numpy as np
from assembly_classifier import AssemblyClassifier

x_train, y_train = np.load("data/old_representations/binary_digits_binary_pixels/x_train.npy"), np.load("data/old_representations/binary_digits_all_pixels/y_train.npy")
x_test, y_test = np.load("data/old_representations/binary_digits_binary_pixels/x_test.npy"), np.load("data/old_representations/binary_digits_all_pixels/y_test.npy")


d = x_train.shape[1]
# AC = AssemblyClassifier(d)
AC = AssemblyClassifier(d, n_cap=27, initial_projection=False)

correct = AC.accuracy(x_test, y_test)
total = len(y_test)
print("-----PRE-TESTING-----")
print('Accuracy of the network on the test set: %d / %d = %.2f %%' % (correct, total, 100.*correct/total))
pre_acts = AC.get_output_activities(x_test)

print("-----TRAINING-----")
n_train = 100
AC.train(x_train[:n_train], y_train[:n_train])

correct = AC.accuracy(x_test, y_test)
total = len(y_test)
print("-----PRE-TESTING-----")
print('Accuracy of the network on the test set: %d / %d = %.2f %%' % (correct, total, 100.*correct/total))
post_acts = AC.get_output_activities(x_test)

# Code Snippet to compute margins for correctly classified examples and plot them

pre_margins = np.abs(pre_acts[:,0] - pre_acts[:,1])
post_margins = np.abs(post_acts[:,0] - post_acts[:,1])
post_labels = np.argmax(post_acts, axis=1)
correct_examples = np.where(post_labels == y_test)[0]
import matplotlib.pyplot as plt
plt.hist(pre_margins[correct_examples]); plt.figure(); plt.hist(post_margins[correct_examples]); plt.show()

# Code snippet to run a certain number of reps of the classifier and see how often it succeeds

p, reps = 0.1, 50
results = {'pre_train':[], 'post_train':[]}

for r in range(reps):
    AC = AssemblyClassifier(d, n_cap=27, edge_prob=p, initial_projection=False)
    correct = AC.accuracy(x_test, y_test)
    total = len(y_test)
    results['pre_train'].append(100.*correct/total)
    n_train = 100
    AC.train(x_train[:n_train], y_train[:n_train])
    correct = AC.accuracy(x_test, y_test)
    total = len(y_test)
    results['post_train'].append(100.*correct/total)
    if (r+1) % 10 == 0:
        print(r+1)

plt.hist(results['post_train']); plt.title('$p = %.2f$'%(p,)); plt.xlabel('accuracy'); plt.ylabel('number of experiments'); plt.show()
