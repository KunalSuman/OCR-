import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


f = open('/home/kunal/ML/1/archives/train-images.idx3-ubyte', 'rb')
f2 = open('/home/kunal/ML/1/archives/train-labels.idx1-ubyte', 'rb')
f2.read(8)
f.read(4)

images = f.read(4)
rows = f.read(4)
cols = f.read(4)

n_images = int.from_bytes(images, byteorder='big')
n_rows = int.from_bytes(rows, byteorder='big')
n_cols = int.from_bytes(cols, byteorder='big')

Test_lable = f2.read();
Test_images = f.read();
final_images = np.frombuffer(Test_images, dtype=np.uint8)
final_images = final_images.reshape(n_images, n_rows*n_cols)

final_label = np.frombuffer(Test_lable, dtype=np.uint8)
print(final_label.shape)
print(final_label)

plt.imshow(final_images[0].reshape(n_rows, n_cols), cmap='gray')
plt.show()

# print(final_images[0].reshape(n_rows, n_cols))
# print(final_images.shape)

final_images = final_images / 255.0
# print(final_images[0].reshape(n_rows, n_cols))

W1 = np.empty((128, 784))

for i in range(128) :
    for j in range(784):
        W1[i][j] = np.random.rand()
        #print(W1[i][j])

WF2 = pd.DataFrame(W1)
WF2.to_csv('W1.csv', index=False)

B1 = np.empty(128)
for i in range(128) :
    B1[i] = np.random.rand()


Z1 = final_images.dot(W1.T) + B1
print("Z1 before RELU",Z1)

Z1 = np.maximum(0 , Z1)
print("Z1 after RELU",Z1)


print(Z1.shape)

W2 = np.empty((10, 128))
for i in range(10):
    for j in range(128):
        W2[i][j] = np.random.rand()


WF2 = pd.DataFrame(W2)


B2 = np.empty(10)
for i in range(10) :
    B2[i] = np.random.rand()


Z2 = Z1.dot(W2.T) + B2

print("Logits[0:5]:\n", Z2[:5])
print("Z2 min/max per class:\n", Z2.min(axis=0), Z2.max(axis=0))
print("Column means of logits:", Z2.mean(axis=0))


Z3 = np.argmax(Z2, axis=1)
print("Z3",Z3)



f.close()
