#### 摘要

本实验旨在通过实现一个条件生成对抗网络（Conditional GAN）来生成图像。实验中使用了深度学习框架PyTorch，并在cityscapes数据集和facades数据集上进行了训练和验证。

#### 方法

- **数据集**：使用cityscapes数据集和facades数据集，按照train_list和val_list训练，该数据集包含RGB图像和相应的语义分割图像。
- **模型**：实现了一个条件GAN，包括一个生成器和一个判别器。生成器负责生成图像，判别器负责区分真实图像和生成的图像。
- **训练**：使用Adam优化器和二元交叉熵损失函数（BCELoss）进行训练。训练过程中，生成器和判别器的参数被交替更新。
- **验证**：在验证集上评估模型性能，并保存生成的图像以进行视觉比较。

### 文件解释

1. **ConditionalGAN_network.py**
   - 定义了条件生成对抗网络的生成器（`Condition_GAN_Generate`）和判别器（`Condition_GAN_Discriminitor`）。
   - 生成器和判别器都接收图像通道（`img_channels`）和条件通道（`condition_channels`）作为输入。
   - 生成器通过一系列卷积层和转置卷积层（下采样和上采样）生成图像。
   - 判别器通过一系列卷积层来判断输入图像是真实的还是由生成器生成的。
2. **facades_dataset.py**
   - 定义了一个用于加载和处理图像数据集的`FacadesDataset`类。
   - 该类从文本文件中读取图像文件名，加载图像，并将其转换为PyTorch张量。
3. **GAN_network.py**
   - 定义了一个非条件的生成对抗网络的生成器（`GAN_Generate`）和判别器（`GAN_Discriminitor`）。
   - 这些类与条件GAN的结构类似，但不使用条件通道。
4. **train.py**
   - 包含了训练和验证Conditional GAN的代码。
   - 定义了图像张量转换为可由OpenCV处理的NumPy数组的函数。
   - 实现了保存训练和验证过程中生成的图像的函数。
   - 定义了训练一个epoch和验证模型的函数。
   - 计算模型参数数量的函数。
   - 主函数`main`中设置了训练和验证过程，包括数据加载、模型初始化、损失函数和优化器的设置，并执行了训练循环。

#### 结果

train_results

![result_1](.\train_results\cityscapes\epoch_1600\result_1.png)

val_results![result_2](.\val_results\cityscapes\epoch_1600\result_2.png)