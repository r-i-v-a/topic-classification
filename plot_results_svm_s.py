#!/usr/bin/env python

import matplotlib.pyplot as pyplot

vec_size = [200, 400, 600, 800, 1000]

acc_mi = [84.06, 87.86, 88.28, 88.32, 88.32]
acc_x2 = [57.62, 57.54, 57.77, 57.67, 57.68]
acc_tf_idf = [75.75, 79.70, 84.38, 85.66, 86.50]

ax = pyplot.subplot()

ax.plot(vec_size, acc_mi, label='MI', color='b', marker='|', markersize=10)
ax.plot(vec_size, acc_x2, label='X2', color='b', marker='x', markersize=10)
ax.plot(vec_size, acc_tf_idf, label='TF-IDF', color='b', marker='o', markersize=10)

pyplot.title('SVM: accuracy of feature selection methods')
pyplot.xlabel('number of features used')
pyplot.ylabel('classification accuracy (percent)')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

pyplot.show()