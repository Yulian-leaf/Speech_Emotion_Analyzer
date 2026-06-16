"""
完整的演示和使用指南
"""

# ============================================================================
# Speech Emotion Analyzer Pro - 完整使用指南
# ============================================================================

# 一、安装依赖
# pip install numpy librosa tensorflow keras scikit-learn pandas matplotlib
# pip install pyyaml soundfile flask scipy

# ============================================================================
# 二、导入模块
# ============================================================================

from emotion_analyzer.preprocessing.augmentation import AugmentationPipeline, AudioAugmentation
from emotion_analyzer.preprocessing.feature_extractor import MultiFeatureExtractor
from emotion_analyzer.models.hybrid_models import create_model, HybridCNNLSTM
from emotion_analyzer.utils.trajectory_analyzer import EmotionTrajectoryAnalyzer
from emotion_analyzer.utils.explainability import ModelExplainability
from emotion_analyzer.utils.trainer import TrainingConfig, UnifiedTrainer, ExperimentTracker

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# 三、快速开始示例
# ============================================================================

def quick_start_example():
    """快速开始示例"""
    
    print("=" * 60)
    print("🎤 Speech Emotion Analyzer - 快速开始")
    print("=" * 60)
    
    # 1. 加载配置
    config = TrainingConfig()
    print(f"\n✓ 配置已加载")
    print(f"  模型类型: {config.get('model.type')}")
    print(f"  样本率: {config.get('audio.sample_rate')} Hz")
    print(f"  特征维度: MFCC={config.get('features.n_mfcc')}, Mel={config.get('features.n_mel')}")
    
    # 2. 初始化特征提取器
    feature_extractor = MultiFeatureExtractor(
        sample_rate=config.get('audio.sample_rate'),
        duration=config.get('audio.duration')
    )
    feature_dim = feature_extractor.get_feature_dimension()
    print(f"\n✓ 特征提取器已初始化")
    print(f"  总特征维度: {feature_dim}")
    
    # 3. 初始化数据增强管道
    augmentation = AugmentationPipeline(sample_rate=config.get('audio.sample_rate'))
    print(f"\n✓ 数据增强管道已初始化")
    print(f"  支持特性:")
    print(f"  - 音高移位 (Pitch Shifting)")
    print(f"  - 时间拉伸 (Time Stretching)")
    print(f"  - 噪声添加 (Noise Injection)")
    print(f"  - 动态范围压缩 (Dynamic Range Compression)")
    
    # 4. 创建混合模型
    input_shape = (feature_dim, 1)  # 或根据需要调整
    num_classes = len(config.get('emotions.classes', []))
    
    model_obj = create_model(
        model_type=config.get('model.type', 'hybrid_cnn_lstm'),
        input_shape=input_shape,
        num_classes=num_classes,
        cnn_filters=config.get('model.cnn_filters'),
        lstm_units=config.get('model.lstm_units'),
        dropout=config.get('model.dropout')
    )
    
    print(f"\n✓ {config.get('model.type')} 模型已创建")
    print(f"  输入形状: {input_shape}")
    print(f"  输出类别: {num_classes}")
    
    return config, feature_extractor, augmentation, model_obj


# ============================================================================
# 四、数据增强演示
# ============================================================================

def augmentation_demo(audio_path: str):
    """数据增强演示"""
    
    print("\n" + "=" * 60)
    print("🔄 数据增强演示")
    print("=" * 60)
    
    # 加载音频
    y, sr = librosa.load(audio_path, sr=44100)
    
    # 创建增强管道
    augmentor = AugmentationPipeline(sample_rate=sr)
    
    # 生成多个增强样本
    augmented_samples = augmentor.generate_augmented_samples(
        y,
        num_augmentations=5,
        pitch_range=(-2, 2),
        time_stretch_range=(0.8, 1.2),
        noise_factor=0.005
    )
    
    print(f"\n✓ 已生成 {len(augmented_samples)} 个增强样本")
    print(f"  包括 1 个原始样本 + 5 个增强样本")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, sample in enumerate(augmented_samples[:6]):
        ax = axes[i]
        ax.plot(sample[:sr*2])  # 绘制前2秒
        ax.set_title(f'Sample {i}' if i == 0 else f'Augmented {i}')
        ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    return augmented_samples


# ============================================================================
# 五、多维特征提取演示
# ============================================================================

def feature_extraction_demo(audio_path: str):
    """多维特征提取演示"""
    
    print("\n" + "=" * 60)
    print("🎵 多维特征提取演示")
    print("=" * 60)
    
    # 加载音频
    y, sr = librosa.load(audio_path, sr=44100)
    
    # 创建特征提取器
    extractor = MultiFeatureExtractor(sample_rate=sr)
    
    # 提取所有特征
    features = extractor.extract_all_features(y)
    
    print(f"\n✓ 特征提取完成")
    print(f"  总特征维度: {len(features)}")
    print(f"  特征类型:")
    print(f"    - MFCC: 26维")
    print(f"    - Mel-Spectrogram: 384维")
    print(f"    - Chroma: 24维")
    print(f"    - Zero Crossing Rate: 4维")
    print(f"    - Spectral Features: 8维")
    print(f"    - Tempogram: 4维")
    
    # 显示特征统计
    print(f"\n📊 特征统计:")
    print(f"  均值: {np.mean(features):.6f}")
    print(f"  标准差: {np.std(features):.6f}")
    print(f"  最大值: {np.max(features):.6f}")
    print(f"  最小值: {np.min(features):.6f}")
    
    return features


# ============================================================================
# 六、情感轨迹分析演示
# ============================================================================

def trajectory_analysis_demo(audio_path: str, model, feature_extractor):
    """情感轨迹分析演示"""
    
    print("\n" + "=" * 60)
    print("📈 情感轨迹分析演示")
    print("=" * 60)
    
    # 创建轨迹分析器
    analyzer = EmotionTrajectoryAnalyzer(model, feature_extractor)
    
    # 分析轨迹
    trajectory_data = analyzer.analyze_trajectory(
        audio_path,
        segment_duration=0.5,
        overlap=0.1
    )
    
    print(f"\n✓ 轨迹分析完成")
    print(f"  总时长: {trajectory_data['total_duration']:.2f} 秒")
    print(f"  分段数: {len(trajectory_data['emotions'])}")
    print(f"  平均置信度: {trajectory_data['statistics']['average_confidence']:.3f}")
    print(f"  情感转换次数: {trajectory_data['statistics']['num_transitions']}")
    
    # 打印摘要
    print(analyzer.get_trajectory_summary(trajectory_data))
    
    # 绘制轨迹
    analyzer.plot_trajectory(trajectory_data, save_path='trajectory_analysis.png')
    
    return trajectory_data


# ============================================================================
# 七、模型可解释性演示
# ============================================================================

def explainability_demo(X_sample, y_sample, model, feature_names=None):
    """模型可解释性演示"""
    
    print("\n" + "=" * 60)
    print("🔍 模型可解释性演示")
    print("=" * 60)
    
    # 创建可解释性分析器
    explainer = ModelExplainability(model, feature_names)
    
    # 生成报告
    emotion_labels = [
        'female_angry', 'female_calm', 'female_fearful', 'female_happy',
        'female_sad', 'male_angry', 'male_calm', 'male_fearful',
        'male_happy', 'male_sad'
    ]
    
    report = explainer.generate_report(X_sample, y_sample, emotion_labels)
    print(report)
    
    # 可视化
    print("\n📊 生成可视化...")
    explainer.visualize_prediction_explanation(
        X_sample, y_sample, emotion_labels,
        save_path='prediction_explanation.png'
    )
    
    return report


# ============================================================================
# 八、Web 仪表板启动
# ============================================================================

def start_web_dashboard():
    """启动Web可视化仪表板"""
    
    print("\n" + "=" * 60)
    print("🌐 Web 可视化仪表板")
    print("=" * 60)
    
    print(f"""
    ✓ Web 应用配置:
    
    位置: web_app/app.py
    启动命令:
    
        cd web_app
        python app.py
    
    访问地址: http://localhost:5000
    
    功能特性:
    ✓ 实时音频上传和分析
    ✓ 情感分布可视化
    ✓ 分析历史管理
    ✓ 批量文件处理
    ✓ 数据导出 (CSV)
    ✓ 交互式仪表板
    """)


# ============================================================================
# 九、实验跟踪演示
# ============================================================================

def experiment_tracking_demo():
    """实验跟踪演示"""
    
    print("\n" + "=" * 60)
    print("📊 实验跟踪和对比")
    print("=" * 60)
    
    tracker = ExperimentTracker()
    
    # 模拟多个实验
    experiments_config = [
        {
            'name': 'CNN-only',
            'model_type': 'cnn',
            'result': {'accuracy': 0.72, 'loss': 0.68}
        },
        {
            'name': 'LSTM-only',
            'model_type': 'lstm',
            'result': {'accuracy': 0.71, 'loss': 0.71}
        },
        {
            'name': 'Hybrid CNN-LSTM',
            'model_type': 'hybrid_cnn_lstm',
            'result': {'accuracy': 0.78, 'loss': 0.55}
        }
    ]
    
    for exp_config in experiments_config:
        exp_id = tracker.start_experiment(
            exp_config['name'],
            {'model_type': exp_config['model_type']}
        )
        
        for key, value in exp_config['result'].items():
            tracker.log_result(exp_id, key, value)
        
        tracker.complete_experiment(exp_id)
    
    # 显示对比
    print("\n📈 实验对比结果:")
    comparison = tracker.compare_experiments()
    print(comparison.to_string())
    
    # 保存
    tracker.save_experiments()
    
    return tracker


# ============================================================================
# 十、完整工作流程
# ============================================================================

def complete_workflow():
    """完整工作流程演示"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Speech Emotion Analyzer Pro - 完整工作流程演示        ║
    ╚══════════════════════════════════════════════════════════╝
    
    本框架包含6个创新功能:
    
    1️⃣  情感轨迹分析系统
        ✓ 分析长音频中的情感动态变化
        ✓ 绘制情感变化曲线
        ✓ 检测情感转换点
        
    2️⃣  Web 可视化仪表板
        ✓ 实时音频上传和分析
        ✓ 交互式结果展示
        ✓ 历史记录管理
        
    3️⃣  高级数据增强
        ✓ Pitch shifting / Time stretching
        ✓ Noise injection / Dynamic compression
        ✓ Frequency/Time masking (SpecAugment)
        
    4️⃣  Hybrid CNN-LSTM 模型
        ✓ 融合频域和时序特征
        ✓ 目标准确率 75%+
        ✓ 更好的泛化能力
        
    5️⃣  可解释性 AI 系统
        ✓ 显著性分析 (Saliency Maps)
        ✓ 特征重要性可视化
        ✓ 预测决策解释
        
    6️⃣  模块化训练框架
        ✓ 配置化的实验管理
        ✓ 多实验对比
        ✓ 详细的日志记录
    
    ════════════════════════════════════════════════════════════
    
    快速开始步骤:
    
    Step 1: 准备数据
            - 下载 RAVDESS 和 SAVEE 数据集
            - 解压到 RawData/ 目录
    
    Step 2: 数据处理
            - 特征提取
            - 数据增强
            - 数据分割
    
    Step 3: 模型训练
            - 创建混合模型
            - 配置参数
            - 开始训练
    
    Step 4: 评估和分析
            - 评估模型性能
            - 分析情感轨迹
            - 生成可解释性报告
    
    Step 5: 部署应用
            - 启动 Web 仪表板
            - 进行实时预测
            - 查看分析结果
    
    ════════════════════════════════════════════════════════════
    """)
    
    print(\"\\n🚀 现在你可以开始了！\")
    print(\"\\n推荐阅读:\")
    print(\"  - README.md          (项目介绍)\")
    print(\"  - 本文件                (使用指南)\")
    print(\"  - emotion_analyzer/    (核心模块)\")


if __name__ == '__main__':
    # 打印完整工作流程
    complete_workflow()
    
    # 根据需要运行具体演示
    # quick_start_example()
    # experiment_tracking_demo()
    # start_web_dashboard()
