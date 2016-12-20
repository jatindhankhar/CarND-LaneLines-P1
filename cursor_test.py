import numpy as np
import matplotlib.pyplot as plt
import mpldatacursor


fig, ax = plt.subplots()
image= plt.imread('test_images/solidWhiteCurve.jpg')
ax.imshow(image, interpolation='none')
mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
plt.show()
