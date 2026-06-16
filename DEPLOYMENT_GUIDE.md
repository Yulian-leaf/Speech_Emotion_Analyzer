# 🚀 Speech Emotion Analyzer Pro - 部署指南

## 📋 前置条件

- Python 3.8+
- CUDA 11.0+ (可选,用于GPU加速)
- 约2GB可用磁盘空间

## 📥 安装步骤

### 1. 克隆/下载项目
```bash
cd Speech-Emotion-Analyzer-master
```

### 2. 创建虚拟环境
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt

# 如果使用GPU (可选)
pip install tensorflow-gpu==2.10.0
```

### 4. 下载数据集

#### 方式一: 手动下载
```
RAVDESS: https://zenodo.org/record/1188976
SAVEE: http://kahlan.eps.surrey.ac.uk/savee/Download.html

解压后放到: RawData/ 目录
```

#### 方式二: 脚本下载 (即将推出)
```bash
python scripts/download_datasets.py
```

## 🎯 使用方式

### 方式1️⃣: Python脚本 (离线分析)

```python
from emotion_analyzer.preprocessing.feature_extractor import MultiFeatureExtractor
from emotion_analyzer.models.hybrid_models import create_model
import librosa

# 加载音频
y, sr = librosa.load('audio.wav')

# 提取特征
extractor = MultiFeatureExtractor()
features = extractor.extract_all_features(y)

# 加载模型并预测
model = create_model(model_type='hybrid_cnn_lstm')
prediction = model.predict(features)
```

### 方式2️⃣: Web 仪表板 (推荐)

```bash
# 进入web应用目录
cd web_app

# 启动Flask应用
python app.py

# 在浏览器打开
http://localhost:5000
```

特性:
- 📁 实时音频上传
- 📊 结果可视化
- 📜 历史记录管理
- 📦 批量处理
- 📈 统计分析

### 方式3️⃣: Jupyter Notebook

```bash
# 启动Jupyter
jupyter notebook

# 打开 COMPLETE_GUIDE.py 或创建新notebook
# 导入模块并使用
```

## 🔧 配置调整

编辑 `emotion_analyzer/config/config.yaml`:

```yaml
# 模型配置
model:
  type: "hybrid_cnn_lstm"        # cnn / lstm / hybrid_cnn_lstm
  cnn_filters: [32, 64, 128]
  lstm_units: 128

# 训练配置
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

# 特征配置
features:
  n_mfcc: 13
  n_mel: 128
  use_augmentation: true
```

## 📊 使用示例

### 1. 基础分析
```python
from COMPLETE_GUIDE import quick_start_example

config, feature_extractor, augmentation, model = quick_start_example()
```

### 2. 数据增强
```python
from COMPLETE_GUIDE import augmentation_demo

augmented_samples = augmentation_demo('audio.wav')
```

### 3. 情感轨迹分析
```python
from COMPLETE_GUIDE import trajectory_analysis_demo

trajectory = trajectory_analysis_demo('audio.wav', model, feature_extractor)
```

### 4. 模型可解释性
```python
from COMPLETE_GUIDE import explainability_demo

report = explainability_demo(X_sample, y_sample, model)
```

### 5. 实验跟踪
```python
from COMPLETE_GUIDE import experiment_tracking_demo

tracker = experiment_tracking_demo()
```

## 🏗️ 项目结构

```
Speech-Emotion-Analyzer-master/
├── emotion_analyzer/              # 核心模块
│   ├── config/
│   │   └── config.yaml           # 配置文件
│   ├── models/
│   │   └── hybrid_models.py      # 模型定义
│   ├── preprocessing/
│   │   ├── augmentation.py       # 数据增强
│   │   └── feature_extractor.py  # 特征提取
│   └── utils/
│       ├── trajectory_analyzer.py # 情感轨迹分析
│       ├── explainability.py     # 可解释性
│       └── trainer.py            # 训练框架
├── web_app/                       # Web应用
│   ├── app.py                    # Flask主程序
│   ├── templates/
│   │   └── index.html            # 前端界面
│   └── static/                   # 静态文件
├── RawData/                       # 数据集 (需要下载)
├── saved_models/                  # 保存的模型
├── logs/                          # 日志文件
├── COMPLETE_GUIDE.py             # 完整使用指南
├── requirements.txt              # 依赖列表
└── README.md                      # 项目说明
```

## 🐛 常见问题

### Q1: 导入模块时出错
**A:** 确保:
- Python路径包含项目根目录
- 已安装所有依赖 (pip install -r requirements.txt)
- 虚拟环境已激活

### Q2: 模型加载失败
**A:** 检查:
- saved_models/ 目录是否存在
- 模型文件是否完整 (.h5 文件)
- TensorFlow版本兼容性

### Q3: GPU不工作
**A:** 尝试:
```bash
pip install tensorflow-gpu==2.10.0
# 验证GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Q4: Web应用无法访问
**A:** 检查:
- 防火墙设置
- 端口5000是否被占用
- Flask是否正确运行

## 📈 性能优化

### 1. 使用GPU加速
```bash
pip install tensorflow-gpu
```

### 2. 批量处理
```python
# 一次处理多个文件而不是单个
predictions = model.predict(X_batch)
```

### 3. 模型量化
```bash
# 将模型转换为TensorFlow Lite
python scripts/convert_to_tflite.py
```

## 🔐 安全建议

- 不要在Web应用中上传超大文件 (>100MB)
- 定期清理上传的临时文件
- 在生产环境中使用反向代理 (Nginx)
- 启用HTTPS加密

## 📝 日志文件

- `logs/training_history_*.json` - 训练历史
- `logs/predictions.csv` - 预测记录
- `experiments/experiments_log.json` - 实验记录

## 🆘 技术支持

如遇问题:

1. 查看 README.md 中的完整文档
2. 检查 logs/ 目录中的错误信息
3. 查看console输出信息
4. 参考 COMPLETE_GUIDE.py 中的示例代码

## 📦 模型文件

- `saved_models/best_model.h5` - 最佳模型检查点
- `saved_models/model_weights.h5` - 模型权重
- `model.json` - 模型配置

## 🚀 下一步

1. 训练自己的模型
2. 微调参数以提高准确率
3. 在自己的应用中集成
4. 尝试迁移学习

祝使用愉快! 🎉
