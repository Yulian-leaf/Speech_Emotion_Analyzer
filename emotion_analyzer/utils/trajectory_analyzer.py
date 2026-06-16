"""
情感轨迹分析系统 - 分析长音频中情感的动态变化
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd
from scipy import interpolate


class EmotionTrajectoryAnalyzer:
    """情感轨迹分析器 - 追踪音频中的情感变化"""
    
    def __init__(self, model, feature_extractor, sample_rate: int = 44100):
        """
        初始化
        
        Args:
            model: 训练好的情感分类模型
            feature_extractor: 特征提取器
            sample_rate: 采样率
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.emotion_classes = [
            "female_angry", "female_calm", "female_fearful", "female_happy",
            "female_sad", "male_angry", "male_calm", "male_fearful",
            "male_happy", "male_sad"
        ]
    
    def segment_audio(self, y: np.ndarray, segment_duration: float = 0.5,
                     overlap: float = 0.1) -> Tuple[List[np.ndarray], List[float]]:
        """
        分割音频为重叠的片段
        
        Args:
            y: 音频信号
            segment_duration: 每段时长（秒）
            overlap: 重叠比例 (0-1)
        
        Returns:
            (音频片段列表, 时间戳列表)
        """
        segment_samples = int(segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        timestamps = []
        
        for start in range(0, len(y) - segment_samples, hop_samples):
            segment = y[start:start + segment_samples]
            if len(segment) == segment_samples:
                segments.append(segment)
                # 使用片段中点的时间戳
                timestamp = (start + segment_samples // 2) / self.sample_rate
                timestamps.append(timestamp)
        
        return segments, timestamps
    
    def analyze_trajectory(self, audio_path: str, segment_duration: float = 0.5,
                          overlap: float = 0.1) -> Dict:
        """
        分析音频的情感轨迹
        
        Args:
            audio_path: 音频文件路径
            segment_duration: 分段时长
            overlap: 重叠比例
        
        Returns:
            {
                'timestamps': 时间戳,
                'emotions': 预测情感标签,
                'confidences': 各类别的置信度,
                'dominant_emotion': 主要情感及百分比,
                'emotion_transitions': 情感转换点,
                'statistics': 统计信息
            }
        """
        # 加载音频
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 分割音频
        segments, timestamps = self.segment_audio(y, segment_duration, overlap)
        
        if len(segments) == 0:
            raise ValueError("音频太短或分段参数不合适")
        
        # 特征提取和预测
        predictions = []
        confidences = []
        
        for segment in segments:
            features = self.feature_extractor.extract_all_features(segment)
            features = np.expand_dims(features, axis=0)  # 添加batch维度
            
            pred = self.model.predict(features, verbose=0)[0]
            predictions.append(np.argmax(pred))
            confidences.append(pred)
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # 转换为情感标签
        emotions = [self.emotion_classes[p] for p in predictions]
        
        # 计算情感转换点
        emotion_transitions = self._detect_transitions(predictions, timestamps)
        
        # 统计信息
        statistics = self._compute_statistics(emotions, confidences)
        
        return {
            'timestamps': np.array(timestamps),
            'emotions': emotions,
            'predictions': predictions,
            'confidences': confidences,
            'emotion_transitions': emotion_transitions,
            'statistics': statistics,
            'audio_path': audio_path,
            'total_duration': len(y) / self.sample_rate
        }
    
    def _detect_transitions(self, predictions: np.ndarray, 
                           timestamps: List[float]) -> List[Dict]:
        """
        检测情感转换点
        
        Args:
            predictions: 预测的类别索引
            timestamps: 时间戳
        
        Returns:
            转换点列表
        """
        transitions = []
        
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i-1]:
                transition = {
                    'time': timestamps[i],
                    'from_emotion': self.emotion_classes[predictions[i-1]],
                    'to_emotion': self.emotion_classes[predictions[i]],
                    'index': i
                }
                transitions.append(transition)
        
        return transitions
    
    def _compute_statistics(self, emotions: List[str], 
                          confidences: np.ndarray) -> Dict:
        """计算统计信息"""
        stats = {}
        
        # 情感分布
        unique_emotions, counts = np.unique(emotions, return_counts=True)
        emotion_distribution = dict(zip(unique_emotions, counts))
        stats['emotion_distribution'] = emotion_distribution
        
        # 平均置信度
        stats['average_confidence'] = float(np.mean(np.max(confidences, axis=1)))
        stats['confidence_std'] = float(np.std(np.max(confidences, axis=1)))
        
        # 主要情感
        main_emotion = max(emotion_distribution, key=emotion_distribution.get)
        main_percentage = emotion_distribution[main_emotion] / len(emotions) * 100
        stats['main_emotion'] = f"{main_emotion} ({main_percentage:.1f}%)"
        
        # 情感转换次数
        stats['num_transitions'] = len(emotions) - 1 - sum(
            1 for i in range(1, len(emotions)) if emotions[i] == emotions[i-1]
        )
        
        return stats
    
    def plot_trajectory(self, trajectory_data: Dict, save_path: str = None):
        """
        绘制情感轨迹图
        
        Args:
            trajectory_data: analyze_trajectory返回的数据
            save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        timestamps = trajectory_data['timestamps']
        confidences = trajectory_data['confidences']
        emotions = trajectory_data['emotions']
        predictions = trajectory_data['predictions']
        
        # 1. 情感变化时间序列
        ax1 = axes[0]
        colors = plt.cm.tab10(predictions / 10)
        scatter = ax1.scatter(timestamps, predictions, c=predictions, cmap='tab10',
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.plot(timestamps, predictions, 'o-', alpha=0.3, linewidth=1)
        ax1.set_ylabel('Emotion Index', fontsize=12)
        ax1.set_title('Emotion Trajectory Over Time', fontsize=14, fontweight='bold')
        ax1.set_yticks(range(10))
        ax1.set_yticklabels([f"E{i}" for i in range(10)])
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Emotion')
        
        # 2. 置信度变化
        ax2 = axes[1]
        max_confidence = np.max(confidences, axis=1)
        ax2.plot(timestamps, max_confidence, 'g-o', linewidth=2, markersize=6, label='Max Confidence')
        
        # 添加置信度区间
        mean_confidence = np.mean(confidences, axis=1)
        ax2.fill_between(timestamps, mean_confidence - np.std(confidences, axis=1),
                        mean_confidence + np.std(confidences, axis=1),
                        alpha=0.3, color='green', label='Std Range')
        
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 每个类别的概率变化
        ax3 = axes[2]
        for i in range(len(self.emotion_classes)):
            class_confidence = confidences[:, i]
            ax3.plot(timestamps, class_confidence, 'o-', label=self.emotion_classes[i],
                    alpha=0.7, linewidth=1.5)
        
        ax3.set_xlabel('Time (seconds)', fontsize=12)
        ax3.set_ylabel('Class Probability', fontsize=12)
        ax3.set_title('All Emotion Classes Probability Over Time', fontsize=14, fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"轨迹图已保存到: {save_path}")
        
        plt.show()
    
    def get_trajectory_summary(self, trajectory_data: Dict) -> str:
        """获取轨迹摘要"""
        summary = f"""
╔══════════════════════════════════════════════════════════╗
║         EMOTION TRAJECTORY ANALYSIS REPORT              ║
╚══════════════════════════════════════════════════════════╝

📁 Audio File: {trajectory_data['audio_path']}
⏱️  Total Duration: {trajectory_data['total_duration']:.2f} seconds

📊 Emotion Distribution:
{self._format_distribution(trajectory_data['statistics']['emotion_distribution'])}

🎯 Main Emotion: {trajectory_data['statistics']['main_emotion']}
🎯 Average Confidence: {trajectory_data['statistics']['average_confidence']:.3f}
📈 Confidence Std Dev: {trajectory_data['statistics']['confidence_std']:.3f}
🔄 Emotion Transitions: {trajectory_data['statistics']['num_transitions']}

🔀 Emotion Transitions:
{self._format_transitions(trajectory_data['emotion_transitions'])}
"""
        return summary
    
    def _format_distribution(self, distribution: Dict) -> str:
        total = sum(distribution.values())
        lines = []
        for emotion, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total * 100
            bar = "█" * int(percentage / 5)
            lines.append(f"   {emotion:20s} {bar:20s} {percentage:5.1f}% ({count})")
        return "\n".join(lines)
    
    def _format_transitions(self, transitions: List[Dict]) -> str:
        if not transitions:
            return "   No emotion transitions detected"
        
        lines = []
        for i, trans in enumerate(transitions[:10], 1):  # 显示前10个转换
            time_str = f"{trans['time']:.2f}s"
            from_em = trans['from_emotion'][:15]
            to_em = trans['to_emotion'][:15]
            lines.append(f"   {i}. {time_str:>8s}: {from_em:20s} → {to_em:20s}")
        
        if len(transitions) > 10:
            lines.append(f"   ... and {len(transitions) - 10} more transitions")
        
        return "\n".join(lines)
