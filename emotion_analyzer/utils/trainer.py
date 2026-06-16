"""
综合训练框架 - 支持配置化的模型训练和实验管理
"""

import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime
import pickle


class TrainingConfig:
    """训练配置管理"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置
        
        Args:
            config_path: YAML配置文件路径
        """
        if config_path is None:
            config_path = 'emotion_analyzer/config/config.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any):
        """更新配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: str):
        """保存配置到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = 'logs'):
        """初始化"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_id = timestamp
        
        self.history = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'metrics': {}
        }
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """记录每个epoch的信息"""
        self.history['epochs'].append({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_metric(self, name: str, value: float):
        """记录单个指标"""
        if name not in self.history['metrics']:
            self.history['metrics'][name] = []
        
        self.history['metrics'][name].append(value)
    
    def save_history(self):
        """保存训练历史"""
        output_path = self.log_dir / f'training_history_{self.session_id}.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"训练历史已保存到: {output_path}")
        return output_path
    
    def get_summary(self) -> str:
        """获取训练摘要"""
        summary = f"""
╔══════════════════════════════════════════════════════════╗
║            TRAINING SESSION SUMMARY                     ║
╚══════════════════════════════════════════════════════════╝

Session ID:            {self.session_id}
Start Time:            {self.history['start_time']}
Total Epochs:          {len(self.history['epochs'])}

Final Metrics:
"""
        
        if self.history['epochs']:
            final_metrics = self.history['epochs'][-1]['metrics']
            for key, value in final_metrics.items():
                summary += f"  {key:30s}: {value:8.6f}\n"
        
        return summary


class ExperimentTracker:
    """实验追踪器 - 记录所有实验信息"""
    
    def __init__(self, experiments_dir: str = 'experiments'):
        """初始化"""
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        
        self.experiments = []
    
    def start_experiment(self, name: str, config: Dict) -> str:
        """开始一个新实验"""
        exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        experiment = {
            'exp_id': exp_id,
            'name': name,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'results': {}
        }
        
        self.experiments.append(experiment)
        
        print(f"\n📊 开始实验: {name} (ID: {exp_id})")
        return exp_id
    
    def log_result(self, exp_id: str, key: str, value: Any):
        """记录实验结果"""
        for exp in self.experiments:
            if exp['exp_id'] == exp_id:
                exp['results'][key] = value
                break
    
    def complete_experiment(self, exp_id: str, status: str = 'completed'):
        """完成实验"""
        for exp in self.experiments:
            if exp['exp_id'] == exp_id:
                exp['status'] = status
                exp['end_time'] = datetime.now().isoformat()
                break
        
        print(f"✓ 实验已完成: {exp_id}")
    
    def save_experiments(self):
        """保存所有实验信息"""
        output_path = self.experiments_dir / 'experiments_log.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        print(f"实验日志已保存到: {output_path}")
        return output_path
    
    def compare_experiments(self) -> pd.DataFrame:
        """比较所有实验"""
        data = []
        
        for exp in self.experiments:
            row = {
                'Experiment ID': exp['exp_id'],
                'Name': exp['name'],
                'Status': exp['status'],
                'Model Type': exp['config'].get('model', {}).get('type', 'unknown'),
                **exp['results']
            }
            data.append(row)
        
        return pd.DataFrame(data)


class UnifiedTrainer:
    """统一训练器 - 整合所有功能"""
    
    def __init__(self, config_path: str = None):
        """初始化"""
        self.config = TrainingConfig(config_path)
        self.logger = TrainingLogger()
        self.tracker = ExperimentTracker()
    
    def train_model(self, model, X_train, y_train, X_val, y_val,
                   exp_name: str = 'experiment', **kwargs):
        """
        训练模型
        
        Args:
            model: 模型对象
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            exp_name: 实验名称
            **kwargs: 其他参数
        
        Returns:
            训练历史
        """
        # 开始实验
        exp_id = self.tracker.start_experiment(
            exp_name,
            {'model': self.config.get('model')}
        )
        
        try:
            # 训练
            epochs = self.config.get('training.epochs', 100)
            batch_size = self.config.get('training.batch_size', 32)
            
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # 记录结果
            if hasattr(history, 'history'):
                final_metrics = {
                    'final_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history['val_loss'][-1]),
                    'final_accuracy': float(history.history['accuracy'][-1]),
                    'final_val_accuracy': float(history.history['val_accuracy'][-1])
                }
            else:
                final_metrics = history
            
            self.tracker.log_result(exp_id, 'metrics', final_metrics)
            self.tracker.complete_experiment(exp_id)
            
            print(self.logger.get_summary())
            
            return history
        
        except Exception as e:
            self.tracker.complete_experiment(exp_id, status='failed')
            print(f"❌ 实验失败: {str(e)}")
            raise
    
    def save_experiment_summary(self):
        """保存实验总结"""
        self.logger.save_history()
        self.tracker.save_experiments()
        
        # 生成对比表
        print("\n📈 实验对比:")
        print(self.tracker.compare_experiments().to_string())


# 使用示例
if __name__ == '__main__':
    # 创建配置
    config = TrainingConfig()
    
    # 创建训练器
    trainer = UnifiedTrainer()
    
    # 查看配置
    print("模型类型:", config.get('model.type'))
    print("训练轮数:", config.get('training.epochs'))
    print("批大小:", config.get('training.batch_size'))
    
    # 修改配置
    config.update('training.epochs', 50)
    config.save('emotion_analyzer/config/config_modified.yaml')
    
    print("✓ 配置已保存")
