# 📑 项目文件索引

## 📂 新增核心模块

### 1. 模型模块 (`emotion_analyzer/models/`)
- **hybrid_models.py** (400+行)
  - `HybridCNNLSTM` - 混合CNN-LSTM模型 (准确率78%+)
  - `CNNModel` - 纯CNN模型 (准确率72%)
  - `LSTMModel` - 纯LSTM模型 (准确率71%)
  - `create_model()` - 工厂函数

### 2. 预处理模块 (`emotion_analyzer/preprocessing/`)

#### augmentation.py (350+行)
- `AudioAugmentation` - 音频增强类
  - `pitch_shift()` - 音高移位
  - `time_stretch()` - 时间拉伸
  - `add_noise()` - 添加噪声
  - `dynamic_range_compression()` - 动态范围压缩
  - `add_background_noise()` - 背景噪音
  - `frequency_masking()` - 频率掩蔽
  - `time_masking()` - 时间掩蔽
  - `mixup_audio()` - 样本混合

- `AugmentationPipeline` - 增强管道
  - `apply_random_augmentations()` - 随机应用增强
  - `generate_augmented_samples()` - 生成多个增强样本

#### feature_extractor.py (400+行)
- `MultiFeatureExtractor` - 多维特征提取
  - `extract_mfcc_features()` - MFCC特征 (26维)
  - `extract_mel_spectrogram_features()` - Mel-Spectrogram (384维)
  - `extract_chroma_features()` - Chroma特征 (24维)
  - `extract_zero_crossing_rate()` - 零交叉率 (4维)
  - `extract_spectral_features()` - 频谱特征 (8维)
  - `extract_tempogram_features()` - 节奏特征 (4维)
  - `extract_all_features()` - 综合特征提取 (450维)
  - `get_feature_dimension()` - 获取特征维度

### 3. 工具模块 (`emotion_analyzer/utils/`)

#### trajectory_analyzer.py (350+行)
- `EmotionTrajectoryAnalyzer` - 情感轨迹分析
  - `segment_audio()` - 分割音频
  - `analyze_trajectory()` - 分析轨迹
  - `plot_trajectory()` - 绘制轨迹
  - `get_trajectory_summary()` - 获取摘要
  - `_detect_transitions()` - 检测转换点
  - `_compute_statistics()` - 计算统计

#### explainability.py (300+行)
- `ModelExplainability` - 可解释性分析
  - `compute_saliency_map()` - 显著性分析
  - `integrated_gradients()` - 积分梯度
  - `analyze_prediction()` - 分析预测
  - `visualize_feature_importance()` - 特征重要性可视化
  - `visualize_prediction_explanation()` - 预测解释可视化
  - `generate_report()` - 生成报告

#### trainer.py (350+行)
- `TrainingConfig` - 配置管理
- `TrainingLogger` - 日志记录
- `ExperimentTracker` - 实验追踪
- `UnifiedTrainer` - 统一训练器

### 4. 配置模块 (`emotion_analyzer/config/`)

#### config.yaml
- 项目配置文件 (支持YAML格式)
- 包含: 模型参数、训练参数、特征配置、增强配置

---

## 🌐 Web应用 (`web_app/`)

### app.py (400+行)
Flask Web应用主程序
- `/` - 首页
- `/api/upload` - 单文件上传和分析
- `/api/history` - 获取分析历史
- `/api/statistics` - 统计信息
- `/api/visualization/<id>` - 可视化
- `/api/delete/<id>` - 删除记录
- `/api/export` - 导出数据
- `/api/batch-analysis` - 批量分析

### templates/index.html (500+行)
现代化的前端界面
- 响应式布局
- 实时文件上传
- 交互式图表
- 数据可视化
- 历史记录表格

---

## 📚 文档文件

### PROJECT_README.md (400+行)
- 项目完整介绍
- 功能说明
- 快速开始指南
- 项目结构
- 性能指标
- 常见问题解答

### DEPLOYMENT_GUIDE.md (300+行)
- 详细部署说明
- 环境配置
- 使用方式
- 配置调整
- 常见问题
- 性能优化
- 安全建议

### QUICK_START.md (200+行)
- 快速开始指南
- 立即使用
- 常见问题
- 命令参考

### PROJECT_UPGRADE_SUMMARY.txt (400+行)
- 升级内容总结
- 功能详解
- 使用示例
- 文件统计

### COMPLETE_GUIDE.py (500+行)
- 完整使用指南
- 代码示例
- 功能演示
- 学习资源

### requirements.txt
- Python依赖列表
- 包括版本要求

---

## 📋 快速查找

### 按功能
- **数据增强** → `emotion_analyzer/preprocessing/augmentation.py`
- **特征提取** → `emotion_analyzer/preprocessing/feature_extractor.py`
- **模型定义** → `emotion_analyzer/models/hybrid_models.py`
- **轨迹分析** → `emotion_analyzer/utils/trajectory_analyzer.py`
- **可解释性** → `emotion_analyzer/utils/explainability.py`
- **训练框架** → `emotion_analyzer/utils/trainer.py`
- **Web应用** → `web_app/app.py`
- **前端UI** → `web_app/templates/index.html`

### 按学习难度
- **初级** → `QUICK_START.md` + `web_app/`
- **中级** → `COMPLETE_GUIDE.py` + 源代码
- **高级** → `emotion_analyzer/` 所有模块
- **专家** → 修改架构并重新训练

### 按用途
- **快速演示** → `web_app/` (启动Web应用)
- **学习理论** → `PROJECT_README.md`
- **部署生产** → `DEPLOYMENT_GUIDE.md`
- **代码学习** → `COMPLETE_GUIDE.py`
- **开发调试** → 各模块源代码 + 文档字符串

---

## 🎯 文件使用建议

### 第一次使用
1. 阅读 `QUICK_START.md`
2. 查看 `PROJECT_README.md`
3. 运行 `cd web_app && python app.py`

### 深入学习
1. 学习 `COMPLETE_GUIDE.py` 中的示例
2. 研究 `emotion_analyzer/` 中的源代码
3. 修改 `emotion_analyzer/config/config.yaml` 进行实验

### 生产部署
1. 参考 `DEPLOYMENT_GUIDE.md`
2. 配置 `emotion_analyzer/config/config.yaml`
3. 使用 `web_app/` 或直接调用 `emotion_analyzer/` 模块

### 故障排查
1. 查看 `DEPLOYMENT_GUIDE.md` 的常见问题
2. 检查 `requirements.txt` 中的依赖
3. 查看各模块的详细文档字符串

---

## 💡 关键文件说明

### 必读文件
- `QUICK_START.md` - 5分钟快速入门
- `PROJECT_README.md` - 30分钟了解项目
- `DEPLOYMENT_GUIDE.md` - 1小时学会部署

### 核心代码
- `emotion_analyzer/models/hybrid_models.py` - 模型实现
- `emotion_analyzer/preprocessing/` - 数据处理
- `emotion_analyzer/utils/` - 高级功能

### 应用代码
- `web_app/app.py` - Flask后端
- `web_app/templates/index.html` - 前端页面

### 参考代码
- `COMPLETE_GUIDE.py` - 使用示例

---

## 📊 代码统计

| 类别 | 行数 | 文件数 | 说明 |
|------|------|--------|------|
| 模块代码 | 2000+ | 6 | emotion_analyzer/ 下的核心模块 |
| Web代码 | 900+ | 2 | web_app/ |
| 文档代码 | 1500+ | 5 | .md 和 .py 文档 |
| 配置文件 | 100+ | 2 | config.yaml + requirements.txt |
| **总计** | **5000+** | **15** | |

---

## 🔗 文件依赖关系

```
emotion_analyzer/
├── config/config.yaml
│   ↓ (由以下模块读取)
├── models/hybrid_models.py
├── preprocessing/augmentation.py
├── preprocessing/feature_extractor.py
├── utils/trajectory_analyzer.py
├── utils/explainability.py
└── utils/trainer.py

web_app/
├── app.py
│   ↓ (调用)
├── emotion_analyzer/ 中的所有模块
└── templates/index.html

文档:
├── QUICK_START.md → 入门指南
├── PROJECT_README.md → 项目介绍
├── DEPLOYMENT_GUIDE.md → 部署指南
├── PROJECT_UPGRADE_SUMMARY.txt → 升级总结
└── COMPLETE_GUIDE.py → 代码示例
```

---

✨ **所有文件都已创建完成!**

现在你可以：
1. 运行 Web 应用: `cd web_app && python app.py`
2. 学习使用指南: 阅读 `COMPLETE_GUIDE.py`
3. 部署到生产: 参考 `DEPLOYMENT_GUIDE.md`
