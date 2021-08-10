import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import time
import os

df=pd.read_csv('/Users/avik/Documents/courses/course-python-ml/day-9/data/weight-height.csv', sep=',',header=0)
X = df['Height']
Y = df['Weight']
range_m = np.linspace(-8,8,100)
range_c = np.linspace(-400,400,100)

def error(m, c):
    y_predict = m * X + c
    error = Y - y_predict
    error_sq = error ** 2
    error_sum = error_sq.sum()
    error_avg = error_sq.sum()/(2*len(error_sq))
    return error_avg


M, C = np.meshgrid(range_m, range_c)

Z = []
for c in range_c:
    t = []
    for m in range_m:
        t.append(error(m,c))
    Z.append(t)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("Gradient(m)")
ax.set_ylabel("Intercept(c)")
ax.set_zlabel("MSE")
ax.contour3D(M, C, Z, 100, cmap=cm.cool)
plt.title(f"Error function for changing m and c")
plt.show()
