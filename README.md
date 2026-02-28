# FZT量化选股模型

基于FZT选股公式和LightGBM机器学习的量化选股系统。

## 项目结构
- `data/` - 数据目录
- `src/` - 源代码
- `config/` - 配置文件
- `notebooks/` - 分析笔记本
- `results/` - 结果输出
- `docs/` - 文档

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 下载数据：`python src/data_prep.py --download`
3. 训练模型：`python src/model_train.py`
4. 运行回测：`python src/backtest.py`