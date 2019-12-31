import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import cv2
 
 
img = cv2.imread("plan_sm.png", 0)
 
XX = np.linspace(0, img.shape[1], img.shape[1])
YY = np.linspace(0, img.shape[0], img.shape[0])
 
k = (2 * np.pi) / (12.5 * 10**-4)
 
X, Y = np.meshgrid(XX, YY)
 
 
dx = 1
dy = 1
 
 
 
             
 
 
source_size = 1
source_x = 20
source_y = 20
 
source = np.maximum(source_size**2-(X-source_x)**2-(Y-source_y)**2, 1)
 
# field = np.ones(X.shape)
 
permissivity = np.full(img.shape, 1)
 
for i, row in enumerate(permissivity):
    for j, elem in enumerate(row):
        if(img[i][j]) < 100:
            permissivity[i][j] = 1000
 
 
# plt.imshow(permissivity, cmap='hot')
# plt.show()
 
sim_mat = np.zeros((img.size, img.size))
 
print(permissivity.shape)
 
for i in range(1, len(source)-1):
     
    for j in range(1,len(source[i])-1):
        M = img.shape[1]
        k = M * (i-1)
        sim_mat[k+j][k+j] = -(2/dx**2)+((k**2) / (permissivity[i][j] **2)) - (2/dx**2)
        sim_mat[k+j][(k-M)+j] = source[i-1][j]
        sim_mat[k+j][(k+M)+j] = source[i+1][j]
        sim_mat[k+j][k+(j-i)] = source[i][j-1]
        sim_mat[k+j][k+(j+1)] = source[i][j+1]
         
 
print(sim_mat[30])
 
# plt.plot(sim_mat)
# plt.show()
 
# sim_mat = np.matrix(sim_mat)

print(sim_mat.shape)
source = np.ravel(source)
source.transpose
print(source.shape)
field, res, rank, s = np.linalg.lstsq(sim_mat,source)
 
field = np.reshape(field, (img.shape[0], img.shape[1]))
 
 
print(field.shape)
plt.imshow(field[1:-1][1:-1], cmap='hot')
plt.show()
#print(source.shape)
 
 
 
#print(field)
# field = np.absolute(field)