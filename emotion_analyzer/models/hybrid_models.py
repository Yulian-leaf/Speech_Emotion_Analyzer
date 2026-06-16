"""
Hybrid CNN-LSTM 模型 - 融合频域和时序特征
"""

import numpy as np
from tensorflow import keras
from keras.layers import (
    Conv1D, MaxPooling1D, AveragePooling1D, LSTM, 
    Dense, Dropout, Flatten, Reshape, Input, 
    Bidirectional, Attention, Add, BatchNormalization
)
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


class HybridCNNLSTM:
    """
    Hybrid CNN-LSTM 模型
    - CNN层：提取频域特征（短时特征）
    - LSTM层：学习时序相关性（长期依赖）
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, 
                 cnn_filters: list = [32, 64, 128], lstm_units: int = 128,
                 dropout: float = 0.3, l2_reg: float = 0.001):
        """
        初始化模型
        
        Args:
            input_shape: 输入形状 (时间步, 特征维度)
            num_classes: 分类数
            cnn_filters: CNN各层过滤器数
            lstm_units: LSTM单元数
            dropout: Dropout比率
            l2_reg: L2正则化系数
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.model = None
        self.history = None
    
    def build_model(self) -> Model:
        """构建混合模型"""
        
        # 输入层
        inputs = Input(shape=self.input_shape, name='input')
        
        # CNN 分支 - 提取局部频域特征
        x = inputs
        for i, filters in enumerate(self.cnn_filters):
            x = Conv1D(
                filters=filters,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'conv1d_{i+1}'
            )(x)
            x = BatchNormalization(name=f'bn_{i+1}')(x)
            x = MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
            x = Dropout(self.dropout, name=f'dropout_cnn_{i+1}')(x)
        
        # LSTM 分支 - 学习时序依赖
        lstm_out = Bidirectional(LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_regularizer=keras.regularizers.l2(self.l2_reg),
            name='lstm_1'
        ))(x)
        
        lstm_out = Bidirectional(LSTM(
            self.lstm_units // 2,
            return_sequences=False,
            dropout=self.dropout,
            name='lstm_2'
        ))(lstm_out)
        
        # 融合层
        lstm_out = Dropout(self.dropout, name='dropout_lstm')(lstm_out)
        
        # 全连接层
        dense = Dense(256, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                     name='dense_1')(lstm_out)
        dense = BatchNormalization(name='bn_dense')(dense)
        dense = Dropout(self.dropout, name='dropout_dense')(dense)
        
        dense = Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                     name='dense_2')(dense)
        dense = Dropout(self.dropout / 2, name='dropout_dense2')(dense)
        
        # 输出层
        outputs = Dense(self.num_classes, activation='softmax', name='output')(dense)
        
        # 构建模型
        self.model = Model(inputs=inputs, outputs=outputs, name='HybridCNNLSTM')
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """编译模型"""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32,
             callbacks_list: list = None) -> Dict:
        """
        训练模型
        
        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            epochs: 训练轮数
            batch_size: 批大小
            callbacks_list: 回调函数列表
        
        Returns:
            训练历史
        """
        if callbacks_list is None:
            callbacks_list = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(X, verbose=0)
    
    def get_summary(self):
        """获取模型摘要"""
        return self.model.summary()


class CNNModel:
    """标准CNN模型"""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int,
                 filters: list = [32, 64, 128], dropout: float = 0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.dropout = dropout
        self.model = None
    
    def build_model(self) -> Model:
        """构建CNN模型"""
        model = Sequential([
            Conv1D(self.filters[0], 3, padding='same', activation='relu',
                  input_shape=self.input_shape, name='conv1'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(self.dropout),
            
            Conv1D(self.filters[1], 3, padding='same', activation='relu', name='conv2'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(self.dropout),
            
            Conv1D(self.filters[2], 3, padding='same', activation='relu', name='conv3'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(self.dropout),
            
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(self.dropout),
            Dense(128, activation='relu'),
            Dropout(self.dropout),
            Dense(self.num_classes, activation='softmax')
        ], name='CNNModel')
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


class LSTMModel:
    """标准LSTM模型"""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int,
                 lstm_units: int = 128, dropout: float = 0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
    
    def build_model(self) -> Model:
        """构建LSTM模型"""
        model = Sequential([
            Bidirectional(LSTM(self.lstm_units, return_sequences=True, 
                             dropout=self.dropout), input_shape=self.input_shape),
            Bidirectional(LSTM(self.lstm_units // 2, dropout=self.dropout)),
            Dense(256, activation='relu'),
            Dropout(self.dropout),
            Dense(128, activation='relu'),
            Dropout(self.dropout),
            Dense(self.num_classes, activation='softmax')
        ], name='LSTMModel')
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


def create_model(model_type: str = 'hybrid_cnn_lstm', 
                 input_shape: Tuple[int, int] = (100, 26),
                 num_classes: int = 10, **kwargs) -> Model:
    """
    工厂函数 - 创建指定类型的模型
    
    Args:
        model_type: 模型类型 ('cnn', 'lstm', 'hybrid_cnn_lstm')
        input_shape: 输入形状
        num_classes: 分类数
        **kwargs: 其他参数
    
    Returns:
        编译后的模型
    """
    if model_type == 'hybrid_cnn_lstm':
        model_obj = HybridCNNLSTM(input_shape, num_classes, **kwargs)
    elif model_type == 'cnn':
        model_obj = CNNModel(input_shape, num_classes, **kwargs)
    elif model_type == 'lstm':
        model_obj = LSTMModel(input_shape, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_obj.build_model()
    model_obj.compile_model()
    
    return model_obj
