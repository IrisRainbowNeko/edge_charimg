# edge_charimg
把图像的边缘特征转化成字符图

利用opencv提取图像边缘特征，再将边缘图网格化送进神经网络进行分类，实现边缘特征的字符图生成
<br>
分类网络使用LeNet5架构，字符分为以下5类
<br>
/ - \ = |
<br>
在此基础上可以增加类别或者制作更好的端到端的数据集来提高效果

# 转换效果
![image](https://github.com/7eu7d7/edge_charimg/tree/master/imgs/hinata.PNG)
原始图片

![image](https://github.com/7eu7d7/edge_charimg/tree/master/imgs/hinata_char.png)
转换的字符图

# 把一只猫的图片转换成字符图
![image](https://github.com/7eu7d7/edge_charimg/tree/master/imgs/fbk.jpg)
原始图片

![image](https://github.com/7eu7d7/edge_charimg/tree/master/imgs/fbk_char.PNG)
转换的字符图
