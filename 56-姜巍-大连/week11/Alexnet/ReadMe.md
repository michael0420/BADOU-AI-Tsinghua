# 内容导引

分类网络是在Pytorch框架下，数据集使用catVSdog(猫狗大战)，在GPU上训练模型并使用模型预测的。
在下载并整理好数据之后，查看“代码使用方法.md”。

<ol>
  <li>数据目录.png  -- 数据集存储结构，方便制作dataset  (VGG16、ResNet50等网络公用一个dataset)</li>
  <li>index_for_images.py  --  为数据做分类标签，并存储成[数据路径, 数据标签]形式于.txt文件中，以完成制作dataset的准备</li>
  <li>Original_Alexnet PyTorch.py  -- 根据原论文编写接近于原始的Alexnet神经网络</li>
  <li>ModifiedAlexNet.py  --  根据作业、数据内容、GPU限制等客观条件而优化的Alexnet网络</li>
  <li>datasets_process.py  --  重写datasets类</li>
  <li>train_and_save_net.py  --  实现训练过程代码</li>
  <li>test_net.py  --  测试模型代码，输出正确率</li>
  <li>Modified_Alex_net.pth  --  保存的训练好的权重</li>
  <li>loss plot.py  --  画loss图代码</li>
  <li>loss plot.png  --  作图结果展示</li>
  <li>predict_single_img.py  --  单张图片预测代码</li>
  <li>单张图片的预测结果.png  --  单张图片预测结果图</li>
  <li>test_cat.jpg  --  用作单张图片预测的图片</li>
</ol>
