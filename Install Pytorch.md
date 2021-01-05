# Pytorch库安装

软件预装：Anaconda+Pycharm

## 1. 创建名为pytorch的环境


在Pycharm中新建项目时，第一行location表示项目保存的位置，第二行location表示新创建的环境所在位置，只需更改\Anaconda\envs\ 后面的环境名，如pytorch，第三行为conda执行文件地址如下

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
