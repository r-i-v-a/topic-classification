#!/usr/bin/env python

import matplotlib.pyplot as pyplot

vec_size = [200, 400, 600, 800, 1000]

acc_mi = [84.79, 86.29, 84.73, 84.05, 83.64]
acc_x2 = [62.80, 60.96, 59.61, 56.96, 55.40]
acc_tf_idf = [72.03, 76.85, 80.79, 81.60, 82.96]

ax = pyplot.subplot()

ax.plot(vec_size, acc_mi, label='MI', marker='|', markersize=10)
ax.plot(vec_size, acc_x2, label='X2', marker='x', markersize=10)
ax.plot(vec_size, acc_tf_idf, label='TF-IDF', marker='o', markersize=10)

pyplot.title('NB: accuracy of feature selection methods')
pyplot.xlabel('number of features used')
pyplot.ylabel('classification accuracy (percent)')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

pyplot.show()