# An example for CartPole

## 安装库
安装 gym 库： `pip install gym` 

安装 pygame 库： `pip install gym[classic_control]`

安装 moviepy 库： `pip install moviepy`

## 代码中的三个tricks
1. 对 state 向量 normalize，使各元素值接近，便于模型训练
2. 原始的游戏中 reward 恒为 1，需要修改，因此增加位置和角度惩罚项
3. MLP中激活函数选用 selu 函数，而非常用的 relu 或者 leaky relu
