"""
多维特征提取模块 - 综合特征提升准确率
"""

import numpy as np
import librosa
from typing import Dict, Tuple


class MultiFeatureExtractor:
    """多维特征提取器"""
    
    def __init__(self, sample_rate: int = 44100, duration: float = 2.5):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_fft = 2048
        self.hop_length = 512
    
    def extract_mfcc_features(self, y: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            y: 音频信号
            n_mfcc: MFCC系数个数
        
        Returns:
            MFCC特征 (n_mfcc,)
        """
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=self.sample_rate, 
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # 计算统计量
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])
    
    def extract_mel_spectrogram_features(self, y: np.ndarray, n_mel: int = 128) -> np.ndarray:
        """
        提取Mel-Spectrogram特征
        
        Args:
            y: 音频信号
            n_mel: Mel频带数
        
        Returns:
            Mel-Spectrogram特征
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=n_mel
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 统计特征
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_std = np.std(mel_spec_db, axis=1)
        mel_max = np.max(mel_spec_db, axis=1)
        
        return np.concatenate([mel_mean, mel_std, mel_max])
    
    def extract_chroma_features(self, y: np.ndarray) -> np.ndarray:
        """
        提取Chroma特征 (音高能量分布)
        
        Args:
            y: 音频信号
        
        Returns:
            Chroma特征 (12,) - 12个音符的能量
        """
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # 统计
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        return np.concatenate([chroma_mean, chroma_std])
    
    def extract_zero_crossing_rate(self, y: np.ndarray) -> np.ndarray:
        """
        提取零交叉率 - 衡量声音的"周期性"
        
        Args:
            y: 音频信号
        
        Returns:
            ZCR特征 (4,) - 均值、std、最大、最小
        """
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        zcr_flat = zcr.flatten()
        
        return np.array([
            np.mean(zcr_flat),
            np.std(zcr_flat),
            np.max(zcr_flat),
            np.min(zcr_flat)
        ])
    
    def extract_spectral_features(self, y: np.ndarray) -> np.ndarray:
        """
        提取频谱特征 - 中心频率、带宽、滚转频率等
        
        Args:
            y: 音频信号
        
        Returns:
            频谱特征 (8,)
        """
        spec_cent = librosa.feature.spectral_centroid(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        spec_rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        spec_bw = librosa.feature.spectral_bandwidth(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return np.array([
            np.mean(spec_cent), np.std(spec_cent),
            np.mean(spec_rolloff), np.std(spec_rolloff),
            np.mean(spec_bw), np.std(spec_bw),
            np.max(spec_cent), np.min(spec_cent)
        ])
    
    def extract_tempogram_features(self, y: np.ndarray) -> np.ndarray:
        """
        提取节奏/速度特征
        
        Args:
            y: 音频信号
        
        Returns:
            节奏特征 (4,)
        """
        # 提取节拍
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate)
        tempogram = librosa.feature.tempogram(onset_env=onset_env, sr=self.sample_rate)
        
        return np.array([
            np.mean(tempogram),
            np.std(tempogram),
            np.max(tempogram),
            np.min(tempogram)
        ])
    
    def extract_all_features(self, y: np.ndarray, config: Dict = None) -> np.ndarray:
        """
        提取所有特征的综合特征向量
        
        Args:
            y: 音频信号
            config: 配置字典，指定要提取的特征
        
        Returns:
            综合特征向量
        """
        if config is None:
            config = {
                'mfcc': True,
                'mel_spec': True,
                'chroma': True,
                'zcr': True,
                'spectral': True,
                'tempogram': True
            }
        
        features = []
        
        if config.get('mfcc', True):
            mfcc_feat = self.extract_mfcc_features(y)
            features.append(mfcc_feat)
        
        if config.get('mel_spec', True):
            mel_feat = self.extract_mel_spectrogram_features(y)
            features.append(mel_feat)
        
        if config.get('chroma', True):
            chroma_feat = self.extract_chroma_features(y)
            features.append(chroma_feat)
        
        if config.get('zcr', True):
            zcr_feat = self.extract_zero_crossing_rate(y)
            features.append(zcr_feat)
        
        if config.get('spectral', True):
            spec_feat = self.extract_spectral_features(y)
            features.append(spec_feat)
        
        if config.get('tempogram', True):
            tempo_feat = self.extract_tempogram_features(y)
            features.append(tempo_feat)
        
        # 拼接所有特征
        combined_features = np.concatenate(features)
        
        return combined_features
    
    def get_feature_dimension(self, config: Dict = None) -> int:
        """获取特征维度"""
        dimensions = {
            'mfcc': 26,           # 13 mean + 13 std
            'mel_spec': 384,       # 128*3 (mean, std, max)
            'chroma': 24,          # 12*2 (mean, std)
            'zcr': 4,              # mean, std, max, min
            'spectral': 8,         # 8个频谱特征
            'tempogram': 4         # 4个节奏特征
        }
        
        if config is None:
            config = {k: True for k in dimensions.keys()}
        
        total_dim = sum(v for k, v in dimensions.items() if config.get(k, True))
        return total_dim
