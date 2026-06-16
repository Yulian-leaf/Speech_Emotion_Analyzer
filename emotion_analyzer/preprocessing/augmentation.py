"""
高级数据增强模块 - 提升模型对抗鲁棒性
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List


class AudioAugmentation:
    """音频数据增强类"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def pitch_shift(self, y: np.ndarray, n_steps: int) -> np.ndarray:
        """
        音高移位 - 改变音调而保持速度
        
        Args:
            y: 音频数组
            n_steps: 移位的半音数 (正数升高，负数降低)
        
        Returns:
            处理后的音频
        """
        return librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=n_steps)
    
    def time_stretch(self, y: np.ndarray, rate: float) -> np.ndarray:
        """
        时间拉伸 - 改变速度而保持音调
        
        Args:
            y: 音频数组
            rate: 拉伸比例 (0.8=变慢, 1.2=变快)
        
        Returns:
            处理后的音频
        """
        return librosa.effects.time_stretch(y, rate=rate)
    
    def add_noise(self, y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        添加高斯噪声 - 模拟嘈杂环境
        
        Args:
            y: 音频数组
            noise_factor: 噪声强度因子
        
        Returns:
            添加噪声后的音频
        """
        noise = np.random.normal(0, noise_factor, len(y))
        return y + noise
    
    def dynamic_range_compression(self, y: np.ndarray, threshold: float = -20, ratio: float = 4) -> np.ndarray:
        """
        动态范围压缩 - 平衡音量
        
        Args:
            y: 音频数组
            threshold: 压缩阈值 (dB)
            ratio: 压缩比
        
        Returns:
            压缩后的音频
        """
        # 转换为dB
        S = librosa.stft(y)
        magnitude = np.abs(S)
        magnitude_db = librosa.power_to_db(magnitude)
        
        # 应用压缩
        compressed = np.where(
            magnitude_db > threshold,
            threshold + (magnitude_db - threshold) / ratio,
            magnitude_db
        )
        
        # 转回线性刻度
        magnitude_linear = librosa.db_to_power(compressed)
        phase = np.angle(S)
        S_compressed = magnitude_linear * np.exp(1j * phase)
        
        return librosa.istft(S_compressed)
    
    def add_background_noise(self, y: np.ndarray, noise_audio: np.ndarray, 
                            noise_factor: float = 0.1) -> np.ndarray:
        """
        添加背景噪音 - 使用真实的背景噪音音频
        
        Args:
            y: 原始音频
            noise_audio: 背景噪音
            noise_factor: 噪音混合比例
        
        Returns:
            混合后的音频
        """
        # 将噪音调整到相同长度
        if len(noise_audio) > len(y):
            noise_audio = noise_audio[:len(y)]
        else:
            noise_audio = np.tile(noise_audio, len(y) // len(noise_audio) + 1)[:len(y)]
        
        # 混合
        return y + noise_factor * noise_audio
    
    def frequency_masking(self, S: np.ndarray, max_freq_width: int = 20) -> np.ndarray:
        """
        频率掩蔽 - SpecAugment技术
        
        Args:
            S: Mel-Spectrogram (频率 × 时间)
            max_freq_width: 最大掩蔽频率宽度
        
        Returns:
            掩蔽后的频谱
        """
        freq_width = np.random.randint(0, max_freq_width)
        freq_start = np.random.randint(0, S.shape[0] - freq_width)
        S[freq_start:freq_start + freq_width, :] = 0
        return S
    
    def time_masking(self, S: np.ndarray, max_time_width: int = 20) -> np.ndarray:
        """
        时间掩蔽 - SpecAugment技术
        
        Args:
            S: Mel-Spectrogram (频率 × 时间)
            max_time_width: 最大掩蔽时间宽度
        
        Returns:
            掩蔽后的频谱
        """
        time_width = np.random.randint(0, max_time_width)
        time_start = np.random.randint(0, S.shape[1] - time_width)
        S[:, time_start:time_start + time_width] = 0
        return S
    
    def mixup_audio(self, y1: np.ndarray, y2: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, float]:
        """
        Mixup数据增强 - 混合两个音频样本
        
        Args:
            y1: 第一个音频
            y2: 第二个音频
            alpha: Beta分布参数，控制混合比例
        
        Returns:
            混合后的音频和混合权重
        """
        lam = np.random.beta(alpha, alpha)
        
        # 调整长度
        min_len = min(len(y1), len(y2))
        y1_trim = y1[:min_len]
        y2_trim = y2[:min_len]
        
        mixed = lam * y1_trim + (1 - lam) * y2_trim
        return mixed, lam


class AugmentationPipeline:
    """增强管道 - 组织多个增强策略"""
    
    def __init__(self, sample_rate: int = 44100):
        self.augmentor = AudioAugmentation(sample_rate)
        self.sample_rate = sample_rate
    
    def apply_random_augmentations(self, y: np.ndarray, 
                                   pitch_range: Tuple[int, int] = (-2, 2),
                                   time_stretch_range: Tuple[float, float] = (0.8, 1.2),
                                   noise_factor: float = 0.005,
                                   apply_compression: bool = True) -> np.ndarray:
        """
        随机应用多个增强
        
        Args:
            y: 音频数组
            pitch_range: 音高移位范围
            time_stretch_range: 时间拉伸范围
            noise_factor: 噪声因子
            apply_compression: 是否应用压缩
        
        Returns:
            增强后的音频
        """
        augmented = y.copy()
        
        # 随机选择应用的增强方法
        if np.random.rand() > 0.5:
            pitch_steps = np.random.randint(pitch_range[0], pitch_range[1])
            augmented = self.augmentor.pitch_shift(augmented, pitch_steps)
        
        if np.random.rand() > 0.5:
            stretch_rate = np.random.uniform(time_stretch_range[0], time_stretch_range[1])
            augmented = self.augmentor.time_stretch(augmented, stretch_rate)
        
        if np.random.rand() > 0.5:
            augmented = self.augmentor.add_noise(augmented, noise_factor)
        
        if apply_compression and np.random.rand() > 0.5:
            augmented = self.augmentor.dynamic_range_compression(augmented)
        
        return augmented
    
    def generate_augmented_samples(self, y: np.ndarray, num_augmentations: int = 3,
                                  **kwargs) -> List[np.ndarray]:
        """
        生成多个增强样本
        
        Args:
            y: 原始音频
            num_augmentations: 生成的增强样本数
            **kwargs: 传递给apply_random_augmentations的参数
        
        Returns:
            增强样本列表
        """
        augmented_list = [y]  # 包含原始样本
        
        for _ in range(num_augmentations):
            augmented = self.apply_random_augmentations(y, **kwargs)
            augmented_list.append(augmented)
        
        return augmented_list
