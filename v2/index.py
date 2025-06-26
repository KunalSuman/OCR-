import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


f = open('/home/kunal/ML/2/archives/train-images.idx3-ubyte', 'rb')

f.read(4)

images = f.read(4)
rows = f.read(4)
cols = f.read(4)

n_images = int.from_bytes(images, byteorder='big')
n_rows = int.from_bytes(rows, byteorder='big')
n_cols = int.from_bytes(cols, byteorder='big')


Test_images = f.read();
final_images = np.frombuffer(Test_images, dtype=np.uint8)
final_images = final_images.reshape(n_images, n_rows*n_cols)



print(final_images[0].reshape(n_rows, n_cols))
print(final_images.shape)

final_images = final_images / 255.0
print(final_images[0].reshape(n_rows, n_cols))

W1 = np.empty((128, 784))

for i in range(128) :
    for j in range(784):
        W1[i][j] = np.random.rand()
        print(W1[i][j])

WF2 = pd.DataFrame(W1)
WF2.to_csv('W1.csv', index=False)

B1 = np.empty(128)
for i in range(128) :
    B1[i] = np.random.rand()
    #print(B1[i])

Z1 = W1.dot(final_images) + B1
print(Z1)

W2 = np.empty((10, 128))
for i in range(10):
    for j in range(128):
        W2[i][j] = np.random.rand()
       # print(W2[i][j])

WF2 = pd.DataFrame(W2)

B2 = np.empty(10)
for i in range(10) :
    B2[i] = np.random.rand()
    #print(B2[i])

Z2 = W2.dot(Z1) + B2



print(Z1)
f.close()
