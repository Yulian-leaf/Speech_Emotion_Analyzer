# 🎯 立即开始使用

## 📥 第一步：安装依赖

```bash
# 安装所有依赖包
pip install -r requirements.txt
```

## 📂 第二步：获取数据集 (可选但推荐)

```
下载地址:
1. RAVDESS: https://zenodo.org/record/1188976
2. SAVEE: http://kahlan.eps.surrey.ac.uk/savee/Download.html

解压到项目的 RawData/ 目录
```

## 🚀 第三步：选择启动方式

### 方式1️⃣: Web 仪表板 (推荐演示) ⭐

```bash
# 进入web应用目录
cd web_app

# 启动Flask应用
python app.py

# 打开浏览器
# http://localhost:5000
```

**功能:**
- 📁 拖拽上传音频
- 📊 实时情感分析
- 📈 结果可视化
- 📜 历史记录管理
- 📦 批量文件处理

### 方式2️⃣: Python 脚本 (开发调试)

```python
from emotion_analyzer.preprocessing.feature_extractor import MultiFeatureExtractor
from emotion_analyzer.models.hybrid_models import create_model
import librosa

# 加载音频
y, sr = librosa.load('audio.wav')

# 提取特征 (450维)
extractor = MultiFeatureExtractor()
features = extractor.extract_all_features(y)

# 创建和预测
model = create_model(model_type='hybrid_cnn_lstm')
prediction = model.predict(features)

# 获取结果
emotion = prediction.argmax()
confidence = prediction[emotion]
```

### 方式3️⃣: 完整指南 (学习所有功能)

```bash
# 运行完整演示 (包含所有功能介绍)
python COMPLETE_GUIDE.py
```

## 📚 阅读文档

按以下顺序阅读:

1. **PROJECT_README.md** - 了解项目和功能
2. **DEPLOYMENT_GUIDE.md** - 学习如何部署
3. **PROJECT_UPGRADE_SUMMARY.txt** - 查看升级内容
4. **COMPLETE_GUIDE.py** - 参考代码示例

## ⚡ 快速命令参考

```bash
# 启动Web应用
cd web_app && python app.py

# 查看所有示例
python COMPLETE_GUIDE.py

# 安装依赖
pip install -r requirements.txt

# 查看配置
cat emotion_analyzer/config/config.yaml
```

## 🎓 6大创新功能一览

| 功能 | 文件位置 | 用途 |
|-----|--------|------|
| 📈 情感轨迹分析 | `emotion_analyzer/utils/trajectory_analyzer.py` | 分析长音频中的情感变化 |
| 🌐 Web仪表板 | `web_app/app.py` | 实时分析和交互 |
| 🔄 数据增强 | `emotion_analyzer/preprocessing/augmentation.py` | 提升模型鲁棒性 |
| 🧠 Hybrid模型 | `emotion_analyzer/models/hybrid_models.py` | 高精度分类 (78%+) |
| 🔍 可解释性 | `emotion_analyzer/utils/explainability.py` | 理解预测原因 |
| ⚙️ 训练框架 | `emotion_analyzer/utils/trainer.py` | 管理实验 |

## 💡 示例代码片段

### 基础预测
```python
model = create_model('hybrid_cnn_lstm')
pred = model.predict(features)
```

### 数据增强
```python
pipeline = AugmentationPipeline()
augmented = pipeline.generate_augmented_samples(y, num=5)
```

### 情感轨迹
```python
analyzer = EmotionTrajectoryAnalyzer(model, extractor)
traj = analyzer.analyze_trajectory('audio.wav')
analyzer.plot_trajectory(traj)
```

### 模型解释
```python
explainer = ModelExplainability(model)
report = explainer.generate_report(X, y, labels)
```

## ❓ 常见问题

**Q: 没有数据集可以运行吗?**  
A: 可以。Web应用可以直接使用，但完整训练需要数据。

**Q: GPU加速怎么启用?**  
A: `pip install tensorflow-gpu` 后自动启用

**Q: 可以在哪些系统上运行?**  
A: Windows / macOS / Linux (需要Python 3.8+)

**Q: Web应用端口被占用怎么办?**  
A: 修改 `web_app/app.py` 中的 `port=5000` 为其他端口

## 🆘 获取帮助

1. 查看对应文件的详细注释
2. 阅读 DEPLOYMENT_GUIDE.md
3. 检查 logs/ 目录的错误日志
4. 参考 COMPLETE_GUIDE.py 中的示例

## ✅ 验证安装

运行以下命令验证安装成功:

```bash
# 测试模块导入
python -c "from emotion_analyzer.models.hybrid_models import create_model; print('✅ OK')"

# 测试Flask
python -c "import flask; print('✅ Flask OK')"

# 测试TensorFlow
python -c "import tensorflow; print('✅ TensorFlow OK')"
```

## 🎉 开始使用

```bash
# 最简单的开始方式
cd web_app
python app.py
```

然后打开浏览器: **http://localhost:5000**

祝你使用愉快! 🚀
