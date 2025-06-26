import numpy as np

Image_file = open('/home/kunal/ML/1/archives/train-images.idx3-ubyte', 'rb')
Label_file = open('/home/kunal/ML/1/archives/train-labels.idx1-ubyte', 'rb')

First_four_bytes = Image_file.read(4)
First_four_bytes = int.from_bytes(First_four_bytes, byteorder='big')
print(First_four_bytes)

First_four_bytes = Label_file.read(8)
First_four_bytes = int.from_bytes(First_four_bytes, byteorder='big')
print(First_four_bytes)

n_images = Image_file.read(4)
n_rows = Image_file.read(4)
n_cols = Image_file.read(4)

n_images = int.from_bytes(n_images, byteorder='big')
n_rows = int.from_bytes(n_rows, byteorder='big')
n_cols = int.from_bytes(n_cols, byteorder='big')
print(n_images, n_rows, n_cols)

images_data = Image_file.read()
images_data = np.frombuffer(images_data, dtype=np.uint8)
images_data = images_data.reshape(n_images, n_rows*n_cols)
images_data = images_data / 255.0
print(images_data.shape)

label_data = Label_file.read()
label_data = np.frombuffer(label_data, dtype=np.uint8)
print(label_data)

layer1 = np.empty((128, 784))
layer2 = np.empty((64, 128))
layer3 = np.empty((32, 64))
layer4 = np.empty((10, 32))

layer1 = np.random.randn(128, 784)
layer2 = np.random.randn(64, 128)
layer3 = np.random.randn(32, 64)
layer4 = np.random.randn(10, 32)

bias1 = np.zeros(128)
bias2 = np.zeros(64)
bias3 = np.zeros(32)
bias4 = np.zeros(10)

# Scale1  = np.sqrt(2/784)
# Scale2  = np.sqrt(2/128)
# Scale3  = np.sqrt(2/128)
# Scale4  = np.sqrt(2/32)

# layer1 = layer1 * Scale1
# layer2 = layer2 * Scale2
# layer3 = layer3 * Scale3
# layer4 = layer4 * Scale4

step_size = 0.01

for epoch in range(100):
    Z1 = images_data.dot(layer1.T) + bias1
    A1 = np.maximum(0, Z1)

    Z2 = A1.dot(layer2.T) + bias2
    A2 = np.maximum(0, Z2)

    Z3 = A2.dot(layer3.T) + bias3
    A3 = np.maximum(0, Z3)

    Z4 = A3.dot(layer4.T) + bias4

    Z4s   = Z4 - np.max(Z4, axis=1, keepdims=True)
    expZ4 = np.exp(Z4s)
    Softmax = expZ4 / np.sum(expZ4, axis=1, keepdims=True)

    Label_data_one_hot = np.zeros((n_images, 10))
    Label_data_one_hot[np.arange(n_images), label_data] = 1

    # back propogation starts here

    dZ4 = (Softmax - Label_data_one_hot)/n_images
    dW4 = dZ4.T.dot(A3)
    dB4 = np.sum(dZ4, axis=0)

    dA3 = dZ4.dot(layer4)

    dZ3 = dA3 * (Z3 > 0)
    dW3 = dZ3.T.dot(A2)
    dB3 = np.sum(dZ3, axis=0)

    dA2 = dZ3.dot(layer3)

    dZ2 = dA2 * (Z2 > 0)
    dW2 = dZ2.T.dot(A1)
    dB2 = np.sum(dZ2, axis=0)

    dA1 = dZ2.dot(layer2)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = dZ1.T.dot(images_data)
    dB1 = np.sum(dZ1, axis=0)

    layer1 = layer1 - step_size * dW1
    layer2 = layer2 - step_size * dW2
    layer3 = layer3 - step_size * dW3
    layer4 = layer4 - step_size * dW4

    bias1 = bias1 - step_size * dB1
    bias2 = bias2 - step_size * dB2
    bias3 = bias3 - step_size * dB3
    bias4 = bias4 - step_size * dB4


    loss_vals = -np.log(Softmax[np.arange(n_images), label_data])
    loss_mean = loss_vals.mean()
    acc       = (np.argmax(Softmax, axis=1)==label_data).mean()
    print(f"Epoch {epoch:2d}  loss={loss_mean:.4f}  acc={acc*100:5.2f}%")
