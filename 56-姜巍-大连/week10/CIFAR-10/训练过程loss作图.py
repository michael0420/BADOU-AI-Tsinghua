import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(2000, 12000, 6, dtype=np.int32)

epoch_01 = np.array([2.219, 1.978, 1.754, 1.600, 1.517, 1.469])
epoch_02 = np.array([1.405, 1.363, 1.356, 1.323, 1.293, 1.280])
epoch_03 = np.array([1.234, 1.228, 1.204, 1.186, 1.179, 1.183])
epoch_04 = np.array([1.111, 1.095, 1.135, 1.119, 1.119, 1.105])
epoch_05 = np.array([0.995, 1.043, 1.048, 1.060, 1.064, 1.042])

plt.plot(x, epoch_01, 'ro-.', x, epoch_02, 'bo-.', x, epoch_03, 'yo-.', x, epoch_04, 'ko-.', x, epoch_05, 'go-.')
plt.legend(['Epoch_1', 'Epoch_2', 'Epoch_3', 'Epoch_4', 'Epoch_5'])
plt.xlabel('number of mini-batches')
plt.ylabel('loss')
plt.title('Loss during CIFAR-10 training procedure in Convolution Neural Networks')
plt.show()
