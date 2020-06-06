# 我的个人公众号
我的公众号:**Gambler_Evolution**
(分享个人在量化投资上的研究)

 ![image](https://github.com/wbbhcb/hefei_ecg/blob/master/Gambler_Evolution.jpg)

**个人知乎**: https://www.zhihu.com/people/e-zhe-shi-wo/activities

# OTTO Data Lab公众号
**公众号id**: OTTODataLab

分享数据挖掘等方面的知识。

 ![image](https://github.com/wbbhcb/hefei_ecg/blob/master/OTTO.jpg)
 
# 代码说明
**代码目录**

>train 存放了训练的代码

>classfier 存放了线上测试的代码

**运行顺序：**

```
# train
# ---先将config2中的参数拷贝到config1中
cp -r config1/config2.py train/config2.py
cp -r config1/config_lgb.py train/lgb/config_lgb.py

# -------------------模型训练
cd train
# 生成预训练的训练测试集
python data_process.py --kind first

# 生成微调阶段的训练测试集
python data_process.py --kind third

# 模型预训练
python main.py --model_name myecgnet --command transfer 

# 微调两种不同模型一种含有34个异常，另一种含有20个异常
python main.py --model_name myecgnet --command train --model_kind 1
python main.py --model_name myecgnet --command train --model_kind 2

# 提取概率特征和fc层特征
python main.py --model_name myecgnet --command get_feature --model_kind 1
python main.py --model_name myecgnet --command get_feature --model_kind 2

# 训练lgb模型
cd lgb
python get_mlfeature.py
python train.py
python train2.py
cd ..
cd ..

# ----------------线上生成结果
# predict
cd classifier

# 获取深度特征(概率特征和fc层特征)
python main.py --command get_feature --model_name myecgnet
python main2.py --command get_feature --model_name myecgnet

# 获取传统特征
cd lgb
python get_mlfeature.py

# 生成最终结果
python get_test.py

```