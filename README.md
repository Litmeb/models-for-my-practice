# models-for-my-practice
<p>to store my practice accomplishments</p>
<p>环境是pytorch2.8.0,python3.9,cuda12.9</p>

# 情绪识别
<p>支持识别50词及内的英文句子，输出文本背后的情绪</p>
<p>支持识别6种：sadness,anger,joy,surprise,love,fear</p>
<p>基于self-attention的模型</p>

# 井字棋
<p>可以下井字棋的AI</p>
<p>基于reinforcement learning</p>

# 文本续写
<p>给出一个英文句子的前半段，自动续写后半段</p>
<p>先实现了基于rnn的模型，然后稍微调整了下变成了基于gru的模型</p>
<p>效果很一般，只能输出一些勉强符合语法的文本</p>

# Mnist
<p>基于深度学习的手写数字识别</p>
<p>输入:np.array(x,28,28) x个样例的28x28像素图片</p>
<p>输出:np.array($n_1,n_2,...,n_x$) $n_i$表示第i个图片最接近的数字</p>
<p>神经网络的结构是：第0层将28x28的输入展平为728个神经元，第1层是128神经元的ReLU激活的全连接层，第2层是10个神经元的softmax激活的全连接层</p>
<p>训练数据来自mnist数据集</p>
<p>个人练手用的</p>

# cifar10
<p>一个cifar10，可以识别10种物体</p>

<p>用pytorch做的一个基于cnn的一个小模型，网络结构是抄的：</p>
<img src="https://p.sda1.dev/27/6b557608adcd54e85bd8793d79e66035/image.png">
<p>对验证集的正确率变化图：</p>
<img src="https://p.sda1.dev/27/1fa3c50467b911728cc55c114fb02515/image.png" >
<p>最终正确率接近70%，还有优化的空间</p>
<p>损失函数定义为交叉熵，对验证集测试的loss变化图：</p>
<img src="https://p.sda1.dev/27/51d916b0c678e812823c90e935ea2017/image.png">
<p>可以看出尚未饱和，还能进一步训练降低loss</p>
