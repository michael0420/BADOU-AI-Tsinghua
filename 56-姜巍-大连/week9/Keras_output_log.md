<font color=#CD5C5C>2022-06-05 11:44:34.004451: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll}</font>
<br>tran_labels: [5 0 4 ... 5 6 8]</br>
test_labels: [7 2 1 ... 4 5 6]
<font color=#CD5C5C><br>2022-06-05 11:44:41.370356: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library nvcuda.dll
<br>2022-06-05 11:44:42.626686: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties: 
<br>pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1660 Ti with Max-Q Design computeCapability: 7.5
<br>coreClock: 1.335GHz coreCount: 24 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 268.26GiB/s
<br>2022-06-05 11:44:42.627740: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
<br>2022-06-05 11:44:42.722063: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
<br>2022-06-05 11:44:42.722426: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll
<br>2022-06-05 11:44:42.756142: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cufft64_10.dll
<br>2022-06-05 11:44:42.766597: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library curand64_10.dll
<br>2022-06-05 11:44:42.791697: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusolver64_11.dll
<br>2022-06-05 11:44:42.823936: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cusparse64_11.dll
<br>2022-06-05 11:44:42.829024: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudnn64_8.dll
<br>2022-06-05 11:44:42.829905: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
<br>2022-06-05 11:44:42.834620: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
<br>To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
<br>2022-06-05 11:44:42.837086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties: 
<br>pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1660 Ti with Max-Q Design computeCapability: 7.5
<br>coreClock: 1.335GHz coreCount: 24 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 268.26GiB/s
<br>2022-06-05 11:44:42.837833: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
<br>2022-06-05 11:44:44.784431: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
<br>2022-06-05 11:44:44.784800: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      0 
<br>2022-06-05 11:44:44.785012: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1277] 0:   N 
<br>2022-06-05 11:44:44.789616: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3985 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1660 Ti with Max-Q Design, pci bus id: 0000:01:00.0, compute capability: 7.5)</font>

_________________________________________________________________
<br>Model: "sequential"</br>

|    Layer (type)     |       Output Shape        |     Param     |
|:-------------------:|:-------------------------:|:-------------:|
|    dense (Dense)    |        (None, 512)        |    401920     |
|   dense_1 (Dense)   |        (None, 10)         |     5130      | 

<br>Total params: 407,050
<br>Trainable params: 407,050</br>
Non-trainable params: 0
_________________________________________________________________
<br>before change: 7
<br>after change:  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
<br><font color=#CD5C5C>2022-06-05 11:44:45.356797: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)</font>
<br>Epoch 1/5
<font color=#CD5C5C><br>2022-06-05 11:44:45.960450: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublas64_11.dll
<br>2022-06-05 11:44:47.964959: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cublasLt64_11.dll</font>
<br>469/469 [==============================] - 4s 2ms/step - loss: 0.2546 - accuracy: 0.9269
<br>Epoch 2/5
<br>469/469 [==============================] - 1s 2ms/step - loss: 0.1030 - accuracy: 0.9693
<br>Epoch 3/5
<br>469/469 [==============================] - 1s 2ms/step - loss: 0.0673 - accuracy: 0.9801
<br>Epoch 4/5
<br>469/469 [==============================] - 1s 2ms/step - loss: 0.0486 - accuracy: 0.9855
<br>Epoch 5/5
<br>469/469 [==============================] - 1s 2ms/step - loss: 0.0368 - accuracy: 0.9894
<br>313/313 [==============================] - 1s 1ms/step - loss: 0.0745 - accuracy: 0.9788
<br>test_loss 0.07452710717916489
<br>test_acc 0.9787999987602234</br>
<br>the number for the picture is :  2
