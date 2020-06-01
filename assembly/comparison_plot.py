import matplotlib.pyplot as plt
import numpy as np

# PLOT WITH SEMILOG ON X AXIS 

data = np.genfromtxt('data/comparison_n.csv', delimiter=',', skip_header=1)
# print(accuracies)

p_vals = []
median_accuracies = []
results_with_over_99_percent_accuracies = []
for row in data:
	p_val = row[0]
	accuracies = row[1:]
	median_accuracy = np.median(accuracies)
	results_with_over_99_percent_accuracy = np.count_nonzero(accuracies > 99)/accuracies.shape[0]

	p_vals.append(p_val)
	median_accuracies.append(median_accuracy)
	results_with_over_99_percent_accuracies.append(results_with_over_99_percent_accuracy)


	
plt.semilogx(p_vals,median_accuracies, marker='o')
plt.title("Accuracy v.s. assembly size")
plt.xlabel("Value of n")
# plt.title("Accuracy v.s. edge connection probability")
# plt.xlabel("Value of p")
# plt.title("Accuracy v.s. capped neurons")
# plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.show()

plt.semilogx(p_vals,results_with_over_99_percent_accuracies, marker='o')
plt.title("Fraction of high-accuracy results v.s. assembly size")
plt.xlabel("Value of n")
# plt.title("Fraction of high-accuracy results v.s. edge connection probability")
# plt.xlabel("Value of p")
# plt.title("Fraction of high-accuracy results v.s. capped neurons")
# plt.xlabel("Value of k")
plt.ylabel("Fraction of results with over 99% accuracy")
plt.show()