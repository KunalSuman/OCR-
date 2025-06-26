import numpy as np ;
import pandas as pd ;

Train_data_file = open('/home/kunal/ML/1/archives/train-images.idx3-ubyte', 'rb')
Train_label_file = open('/home/kunal/ML/1/archives/train-labels.idx1-ubyte', 'rb')

Train_data_file.read(4)
Train_label_file.read(8)

n_images = Train_data_file.read(4)
n_rows = Train_data_file.read(4)
n_cols = Train_data_file.read(4)

n_images = int.from_bytes(n_images, byteorder='big')
n_rows = int.from_bytes(n_rows, byteorder='big')
n_cols = int.from_bytes(n_cols, byteorder='big')

n_test_label = Train_label_file.read();
n_test_images = Train_data_file.read();

n_test_label = np.frombuffer(n_test_label, dtype=np.uint8)

n_test_images = np.frombuffer(n_test_images, dtype=np.uint8)
n_test_images = n_test_images.reshape(n_images, n_rows*n_cols)
n_test_images = n_test_images / 255.0

W1 = np.empty((128, 784))
W1 = np.random.rand(128, 784)

W2 = np.empty((10, 128))
W2 = np.random.rand(10, 128)

B1 = np.empty(128)
B1 = np.random.rand(128)

B2 = np.empty(10)
B2 = np.random.rand(10)

Scale1 = 1/np.sqrt(784)
Scale2 = 1/np.sqrt(128)

W1 = W1*Scale1
W2 = W2*Scale2
step_size = 0.02

for epoch in range(500):
    Z1 = n_test_images.dot(W1.T) + B1
    A1 = np.maximum(0, Z1)
    Z2 = A1.dot(W2.T) + B2
    Softmax = np.exp(Z2) / np.sum(np.exp(Z2),axis=1, keepdims=True)

    softmax_frame = pd.DataFrame(Softmax)
    confidance = np.argmax(Softmax, axis=1)

    Loss_value = Softmax[np.arange(n_test_images.shape[0]),n_test_label]
    Loss_value = -np.log(Loss_value)
    Loss_value_mean = np.mean(Loss_value)

    print("Epoch: ", epoch, " Loss: ", Loss_value)
    loss_mean = np.mean(Loss_value)
    accuracy = np.mean(confidance == n_test_label)

    print(f"Epoch {epoch:2d}   loss = {loss_mean:.4f}   acc = {accuracy*100:5.2f}%")

    hot_loss_val = np.zeros((n_images,10))
    hot_loss_val[np.arange(n_images), n_test_label] = 1

    dZ2 = (Softmax - hot_loss_val)/n_images
    dW2 = dZ2.T.dot(A1)
    dB2 = np.sum(dZ2, axis=0)

    dA1 = dZ2.dot(W2)

    dZ1 = dA1 * (Z1 > 0)
    dW1 = dZ1.T.dot(n_test_images)
    dB1 = np.sum(dZ1, axis=0)

    W1 = W1 - step_size * dW1
    W2 = W2 - step_size * dW2
    B1 = B1 - step_size * dB1
    B2 = B2 - step_size * dB2
