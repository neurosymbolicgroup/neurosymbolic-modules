import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('data/comparison_p.csv', delimiter=',', skip_header=1)
# print(accuracies)

x = []
y = []
for row in data:
	p_val = row[0]

	accuracies = row[1:]
	median_accuracy = np.median(accuracies)

	x.append(p_val)
	y.append(median_accuracy)
	
plt.plot(x,y, marker='o')
plt.title("Accuracy v.s. probability of edge connection")
plt.xlabel("Value of p")
plt.ylabel("Accuracy")
plt.show()