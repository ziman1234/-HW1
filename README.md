# DATA620004
course DATA620004
本次作业构建两层神经网络分类器用于对MINIST数据集进行分类，该神经网络包含：输入层、一个隐藏层、输出层。数据从输入层经过前向传播到输出层，损失函数梯度经过反向传播用于更新每层的权重。搭建该神经网络时，将学习率learning rate、隐藏层大小hidden_dim、正则化强度reg_lambda作为模型待选择参数。
1、先对权重W、b进行初始化，其中对权重W作正态分布假定，对权重b作0假定。2、根据ReLU、softmax写出前向传播函数。3、根据梯度下降和带有L2正则项的交叉熵定义写出反向传播函数。4、	把训练数据shuffle，并根据batch_size对shuffle后的数据进行划分，逐次进行反向传播更新参数。对每个epoch重复上述操作，记录在每个epoch下，训练集的loss和accuracy。并且在循环了一定次数epoch时，对学习率lr进行衰减，每decay_steps下学习率lr衰减一个数量级。
导入将已经下载好的MINIST并使用struct包对其进行处理，将训练集按80%、20%划分。共获得训练集数据48000个（48000*784维）、训练集标签48000个；验证集数据12000个（12000*784维）、验证集标签12000个；测试集数据10000个（10000*784维）、测试集标签10000个。（注意训练集、验证集、测试集数据像素为255，对其进行归一化处理）。输入层大小为784，输出层大小为10，epoch数为100，batch_size为32，学习率衰减率decay_rate为0.1，衰减步长decay_steps为50。
分别对训练集进行训练，并使用训练后的模型对验证集进行预测，以验证集的accuracy作为评价指标，选择出最好的最好的待确定参数。选择出的最优待确定参数为，此时验证集accuracy为98.158%：Hidden layer size: 128, Learning rate: 0.02, Regularization: 0.01
Struct使用相同方法对训练集数据进行处理，隐藏层大小为128，学习率为0.02，正则化强度为0.01，为节省计算时间，将epoch总数设置为40，decay步长为5，batch_size为32。
使用该训练后的模型对测试集10000对数据进行预测，计算所得样本外Accuracy分类精度为0.9817。
