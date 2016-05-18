#!/usr/bin/env python

import matplotlib.pyplot as pyplot

vec_size = [1000, 2000, 3000, 4000, 5000]

acc_mi = [83.64, 81.67, 78.62, 75.83, 72.03]
acc_x2 = [55.40, 48.34, 44.67, 42.16, 39.17]
acc_tf_idf = [82.96, 84.52, 84.11, 83.10, 81.60]

ax = pyplot.subplot()

ax.plot(vec_size, acc_mi, label='MI', color='b', marker='|', markersize=10)
ax.plot(vec_size, acc_x2, label='X2', color='b', marker='x', markersize=10)
ax.plot(vec_size, acc_tf_idf, label='TF-IDF', color='b', marker='o', markersize=10)

pyplot.title('NB: accuracy of feature selection methods')
pyplot.xlabel('number of features used')
pyplot.ylabel('classification accuracy (percent)')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

pyplot.show()