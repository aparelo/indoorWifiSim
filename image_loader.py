import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import cv2
from scipy.special import hankel1, hankel2, hankel1e, hankel2e


img = cv2.imread("plan.png", 0)

XX = np.linspace(0, img.shape[1], img.shape[1])
YY = np.linspace(0, img.shape[0], img.shape[0])

k = (2 * np.pi) / (12.5 * 10**-4)


dx = 1
dy = 1


X, Y = np.meshgrid(XX, YY)
            


source_size = 80
source_x = 200
source_y = 200

source = np.maximum(source_size**3-(X-source_x)**2-(Y-source_y)**2, 1)

field = np.ones(X.shape)

permissivity = np.full(X.shape, 100)

for i, row in enumerate(source):
    for j, elem in enumerate(row):
        if(img[i][j]) < 100:
            permissivity[i][j] = 1


plt.imshow(source, cmap='hot')
plt.show()

    




print(field.shape)
print(source.shape)


for i in range(1,source.shape[0]-1):
    for j in range(1,source.shape[1]-1):
        # print("i: {0}".format(i))
        # print(j)
            
        field[i][j] = source[i][j]/(((field[i+1][j]+field[i-1][j]-2*field[i][j])/(dx**2) +
                            (field[i][j+1]+field[i][j-1]-2*field[i][j])/(dy**2)) +
                            (k**2/permissivity[i][j]**2)* field[i][j])


print(field)
# field = np.absolute(field)

plt.imshow(field, cmap='hot')
plt.show()






