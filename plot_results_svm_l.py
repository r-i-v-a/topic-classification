#!/usr/bin/env python

import matplotlib.pyplot as pyplot

vec_size = [1000, 2000, 3000, 4000, 5000]

acc_mi = [88.32, 88.30, 88.38, 88.31, 88.45]
acc_x2 = [57.68, 56.63, 56.85, 57.64, 54.53]
acc_tf_idf = [86.50, 88.46, 88.59, 88.53, 88.47]

ax = pyplot.subplot()

ax.plot(vec_size, acc_mi, label='MI', marker='|', markersize=10)
ax.plot(vec_size, acc_x2, label='X2', marker='x', markersize=10)
ax.plot(vec_size, acc_tf_idf, label='TF-IDF', marker='o', markersize=10)

pyplot.title('SVM: accuracy of feature selection methods')
pyplot.xlabel('number of features used')
pyplot.ylabel('classification accuracy (percent)')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

pyplot.show()