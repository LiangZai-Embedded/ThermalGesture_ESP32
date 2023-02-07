<<<<<<< HEAD
=======

![_](/4.Img/preface.jpg)

# ThermalGesture_ESP32 - 在ESP32上实现热成像手语识别  
**演示视频链接** https://www.bilibili.com/video/BV1jK41117yV/

## 0. 项目概述 
本项目是一个基于`ESP32`的热成像手势识别系统，使用的传感器是`AMG8833`，这是一个I2C接口的，输出8x8温度数值的热成像传感器阵列，系统对传感器输出的温度数组进行双三次插值，得到24x24的温度数值送入预训练的模型进行推理，进行手势分类，其中，识别**数字手势**使用图像4分类（背景，手势1，手势2，手势3）的卷积神经网络,识别**调节手势**使用图像5分类（背景，张开，合拢远，合拢近，交叉）的卷积神经网络。   

网络模型的训练在PC端，是使用`ESP32`串口回传的温度数据进行训练的，训练框架使用`Keras`，量化工具使用`TensorFlowLite`,训练完成的网络模型在边缘端`ESP32`上进行部署，ESP32的开发平台为`VScode`+`PlatformIO IDE`+`Arduino`,推理框架使用`TensorFlowLite_ESP32`提供的API函数 

项目文件夹包含内容如下：  

* `1.NetworkModel` 包含存有数据集的文件夹`dataset`,模型训练脚本`gesture_train.ipynb`,已量化和未量化的tflite网络文件`model.tflite`,`model_quantized.tflite`
* `2.Firmware_ESP32` 包含三个ESP32工程文件夹，获取数据集`1.GetDataset`,识别数字手势`2.NumberPred`,模拟调节滚动条`3.ProgressBar`
* `3.HardwareInfo` 包含项目用到的硬件信息和datasheet
* `4.Img` 项目相关图片



## 1. 数据集获取
使用`SecureCRT`软件以日志(x.log)的形式保存串口回传的数据，**波特率设定为921600**,采样单个手势的650组温度值的时间大约为1min15s  

**ESP32启动后再连接SecureCRT，最后的log文件删除第一行和最后一行数据，避免因为SecureCRT的连接和断开导致单组数据不完整(一行不足576个温度值)**  

使用`numpy.genfromtxt`读入温度数据，每一组温度数据都需要进行预处理，具体方法是减去均值(单组数据温度值相加除以576),再除以10，防止数值过大

## 2. 网络模型设计
卷积层使用**5个3x3卷积核**，池化层为**2x2的Maxpooling池化**，接下来是两层**全连接（128，64）**，最后的**softmax层**对应网络模型的**分类数**,数5代表手势分5类


![_](/4.Img/network.png)
这里设置单个手势的模型训练样本数为600，测试样本数为50（即**数字手势**的总训练样本为2400，总测试样本为200）batchsize为10，在训练3个epoch后模型在测试集的精度就很高了(>99%)，loss也降得非常低(<0.0001)，因为数据集非常小，且网络学到的数据比较单一，后续会考虑增加数据集（如单个手势2000）并且进行数据增强，让模型的鲁棒性增加，泛化性能提高。

## 3. 网络模型部署
在Linux环境下使用`xxd -i model_quantized.tflite > model.cpp`命令将`tflite`文件转为16进制`unsigned char`数组，数组包含了模型的权重参数和框架(**TensorFlowLite_ESP32库会解析该模型数组**),将`model.cpp`导入ESP32工程文件的`src`文件夹中,更改数组名称为`alignas(8) const unsigned char g_model[]`,进行8字节对齐以及保存进Flash，更改长度名称为`const int g_model_len`


## 4.硬件连接
| ST7789 LCDTFT | ESP32 DevKitC |
|-------|--------|
| CS  | 15或接GND |
| SDA |  23 |
| SCL |  18 |
| RES |  4  |
| DC  |  2  | 

| AMG8833 | ESP32 DevKitC |
|------------| -------------|
| SDA |  21 |
| SCL |  22 |
>>>>>>> 6e7f11aa09eeb6457aa54d2f14da62f8ac5ae3e7
