# 介绍
本项目提出了一个简单的多策略古汉语命名实体识别系统，在EvaHan2025比赛中评估结果中得到了F1值 (Test Total) 84.91 的成绩，大于baseline F1值 (Test Total) 80.61

# 说明
**[update 2025.03.27]** 整理了实验和消融实验部分，各自放在experiments和ablation_study，与论文对应。

**[update 2025.03.23]** 关于论文消融实验表格部分，缺少各类任务每类实体详细F1值的解释：对于单策略模型，我们采用把任务A、B、C的数据集混合到一起作为单策略的数据集。而任务B的标签是任务A的子集，导致无法单独提取出各自任务的F1。


- 模型设计解释：请见论文原文的模型、实验、消融实验部分
- 学习率设置：没有搜索学习率，学习率是5e-5 or 3e-5
- 由于版权问题，GitHub仓库暂不上传数据集及结果
- 可读性问题：因为用到了CRF这个库，其不允许非法标签-1存在
  - BERT的PAD部分使用0，并且使用Attention Mask在各阶段确保计算合法性
  - 为确保Attention Mask 正确应用，重写了部分Trainer类的方法

## 消融实验说明：
**对比实验：验证多策略框架的有效性**，准确率角度

1. 使用 单策略（GujiRoBERTa_jian_fan + CRF）完成任务 A、B 和 C。
2. 使用 单策略（GujiRoBERTa_jian_fan + Softmax）完成任务 A、B 和 C。
3. 使用 多策略（见论文）分别完成任务 A、B 和 C。


**对比实验：trade off**，准确率及训练用时双重因素角度，说明任务B模型，选择softmax即保证了相对准确率又控制了成本

1. 使用 GujiRoBERTa_jian_fan + CRF 完成任务B，并记录训练用时。
2. 使用 GujiRoBERTa_jian_fan + Softmax完成任务 B，并记录训练用时。
3. 可视化的对比这两种训练的模型在测试集上的准确率与用时，并分析了时间复杂度，即O(n^2)与O(n)


# 致谢
感谢我的指导教师、主办方的全体工作人员