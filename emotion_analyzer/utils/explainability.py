"""
可解释性AI系统 - 使用SHAP和Grad-CAM分析模型决策
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import tensorflow as tf
from tensorflow import keras


class ModelExplainability:
    """模型可解释性分析"""
    
    def __init__(self, model, feature_names: List[str] = None):
        """
        初始化
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(100)]
    
    def compute_saliency_map(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算显著性图 - 显示输入中对预测最敏感的区域
        
        Args:
            X: 输入样本
            y: 目标标签
        
        Returns:
            显著性图
        """
        X_tensor = tf.Variable(X, trainable=True)
        
        with tf.GradientTape() as tape:
            predictions = self.model(X_tensor)
            loss = tf.keras.losses.categorical_crossentropy(y, predictions)
        
        # 计算梯度
        gradients = tape.gradient(loss, X_tensor)
        
        # 计算显著性 (梯度绝对值)
        saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
        
        return saliency.numpy()
    
    def integrated_gradients(self, X: np.ndarray, y: np.ndarray, 
                            steps: int = 50) -> np.ndarray:
        """
        积分梯度 - 衡量每个特征对预测的贡献
        
        Args:
            X: 输入样本
            y: 目标标签
            steps: 积分步数
        
        Returns:
            积分梯度
        """
        # 基线 (全零)
        baseline = np.zeros_like(X)
        
        # 从基线插值到输入
        alphas = np.linspace(0, 1, steps)
        integrated_gradients_arr = np.zeros_like(X)
        
        for alpha in alphas:
            X_interp = baseline + alpha * (X - baseline)
            X_tensor = tf.Variable(X_interp, trainable=True)
            
            with tf.GradientTape() as tape:
                predictions = self.model(X_tensor)
                loss = tf.keras.losses.categorical_crossentropy(y, predictions)
            
            gradients = tape.gradient(loss, X_tensor)
            integrated_gradients_arr += gradients.numpy()
        
        # 平均并缩放
        integrated_gradients_arr *= (X - baseline) / steps
        
        return integrated_gradients_arr
    
    def analyze_prediction(self, X: np.ndarray, emotion_labels: List[str]) -> Dict:
        """
        分析单个预测
        
        Args:
            X: 输入特征
            emotion_labels: 情感标签列表
        
        Returns:
            分析结果字典
        """
        # 获取预测
        prediction = self.model.predict(X, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        # 计算显著性
        y_onehot = np.zeros((1, len(emotion_labels)))
        y_onehot[0, predicted_class] = 1
        
        saliency = self.compute_saliency_map(X, y_onehot)[0]
        
        # 获取最重要的特征
        top_feature_indices = np.argsort(saliency)[-5:][::-1]
        
        return {
            'predicted_emotion': emotion_labels[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                emotion_labels[i]: float(prediction[i]) 
                for i in range(len(emotion_labels))
            },
            'top_features': [
                {
                    'feature': self.feature_names[i],
                    'importance': float(saliency[i])
                }
                for i in top_feature_indices
            ],
            'saliency': saliency
        }
    
    def visualize_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                     emotion_labels: List[str],
                                     save_path: str = None):
        """
        可视化特征重要性
        
        Args:
            X: 输入样本
            y: 目标标签 (one-hot)
            emotion_labels: 情感标签
            save_path: 保存路径
        """
        # 计算显著性
        saliency = self.compute_saliency_map(X, y)[0]
        
        # 获取前20个重要特征
        top_indices = np.argsort(saliency)[-20:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]
        top_importance = saliency[top_indices]
        
        # 绘制
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(top_importance / top_importance.max())
        bars = ax.barh(top_features, top_importance, color=colors)
        
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Top 20 Important Features for Emotion Prediction', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, top_importance)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"特征重要性图已保存到: {save_path}")
        
        plt.show()
    
    def visualize_prediction_explanation(self, X: np.ndarray, y_true: np.ndarray,
                                        emotion_labels: List[str],
                                        save_path: str = None):
        """
        可视化预测解释
        
        Args:
            X: 输入特征
            y_true: 真实标签 (one-hot)
            emotion_labels: 情感标签
            save_path: 保存路径
        """
        analysis = self.analyze_prediction(X, emotion_labels)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：情感概率分布
        ax1 = axes[0]
        emotions_list = list(analysis['all_probabilities'].keys())
        probabilities = list(analysis['all_probabilities'].values())
        
        colors = ['#ff6b6b' if e == analysis['predicted_emotion'] else '#4ecdc4' 
                 for e in emotions_list]
        
        ax1.bar(range(len(emotions_list)), probabilities, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(emotions_list)))
        ax1.set_xticklabels(emotions_list, rotation=45, ha='right')
        ax1.set_ylabel('Probability', fontsize=11)
        ax1.set_title('Emotion Probability Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 右图：特征重要性
        ax2 = axes[1]
        top_5 = analysis['top_features'][:5]
        feature_names = [f['feature'][:15] for f in top_5]
        importance_scores = [f['importance'] for f in top_5]
        
        ax2.bar(range(len(feature_names)), importance_scores, color='#667eea', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45, ha='right')
        ax2.set_ylabel('Importance Score', fontsize=11)
        ax2.set_title('Top 5 Important Features', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加预测信息
        fig.suptitle(
            f"Predicted: {analysis['predicted_emotion']} (Confidence: {analysis['confidence']:.2%})",
            fontsize=14, fontweight='bold', y=1.02
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"预测解释图已保存到: {save_path}")
        
        plt.show()
    
    def generate_report(self, X: np.ndarray, y_true: np.ndarray,
                       emotion_labels: List[str]) -> str:
        """
        生成详细的可解释性报告
        
        Args:
            X: 输入特征
            y_true: 真实标签 (one-hot)
            emotion_labels: 情感标签
        
        Returns:
            报告文本
        """
        analysis = self.analyze_prediction(X, emotion_labels)
        true_emotion = emotion_labels[np.argmax(y_true)]
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║         MODEL PREDICTION EXPLANATION REPORT             ║
╚══════════════════════════════════════════════════════════╝

🎯 PREDICTION RESULTS
────────────────────────────────────────────────────────────
Predicted Emotion:     {analysis['predicted_emotion']}
True Emotion:          {true_emotion}
Confidence:            {analysis['confidence']:.2%}
Correct:               {'✓ YES' if analysis['predicted_emotion'] == true_emotion else '✗ NO'}

📊 EMOTION PROBABILITIES
────────────────────────────────────────────────────────────
"""
        
        # 排序概率
        sorted_probs = sorted(analysis['all_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for emotion, prob in sorted_probs:
            bar = "█" * int(prob * 30)
            report += f"{emotion:20s} {bar:30s} {prob:6.1%}\n"
        
        report += f"""
🔍 TOP IMPORTANT FEATURES
────────────────────────────────────────────────────────────
"""
        
        for i, feature_info in enumerate(analysis['top_features'][:10], 1):
            report += f"{i:2d}. {feature_info['feature']:30s} {feature_info['importance']:8.4f}\n"
        
        report += f"""
💡 EXPLANATION
────────────────────────────────────────────────────────────
The model predicted '{analysis['predicted_emotion']}' with {analysis['confidence']:.1%} 
confidence based on the input audio features. The top contributing 
features were {', '.join([f['feature'] for f in analysis['top_features'][:3]])}.

"""
        
        return report
