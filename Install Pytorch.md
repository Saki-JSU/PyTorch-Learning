# Pytorch库安装

软件预装：[Anaconda](https://www.anaconda.com/download) + [Pycharm](https://www.jetbrains.com/pycharm/)

## 1. 新建项目并创建环境

在Pycharm中新建项目时，会出现以下选项：

1. 第一个location表示项目（Project）保存的位置和项目名称（可自定义，此处为项目名为Homework）
2. New environment using 选择 Conda
3. 第二个location表示新创建的环境（Environment）所在位置，需要修改 \Anaconda\envs\ 后面的环境名。环境名命名可以自定义，一般选择python对应版本作为名字，如 py38 或者 py39，此处为 py39
4. conda executable 地址为 Anaconda 软件安装对应的地址
5. Make avaliable to all projects 选√

![图1](https://github.com/Saki-JSU/MarkdownImage/blob/main/Fig1.png?raw=true)


## 2. 删除环境
在Anaconda中可以直接删除，同时在\Anaconda\envs\ 中删除相应文件夹

![图2](https://github.com/Saki-JSU/MarkdownImage/blob/main/Fig2.png?raw=true)



## 3. Anaconda Prompt常用环境命令
显示所有的环境：`conda env list`

激活pytorch：`conda activate pytorch`

退出pytorch：`conda deactivate pytorch`


## 4. 安装pytorch
进入Anaconda Prompt

激活pytorch环境：`conda activate pytorch`

设置清华镜像：

 `conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/`
 
 `conda config --set show_channel_urls yes`
 
安装：`conda install pytorch torchvision torchaudio cpuonly`

> 安装的命令不同版本参见官网   https://pytorch.org/

> 去掉 `-c pytorch` 表示从清华镜像上下载而非原文件下载

> 如下载失败（镜像不稳定，失败时多试几次） 需在\Anaconda\pkgs\  将pytorch-1.7.1-py3.8_cpu_0文件夹删除，然后重新下载

## 5. 打开pycharm，新建一个python 文件，测试代码：
```javascript
import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)
```

如输出正常输出版本号，则安装成功
![图3](https://github.com/Saki-JSU/MarkdownImage/blob/main/Fig3.png?raw=true)
