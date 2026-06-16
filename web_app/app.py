"""
Web 可视化仪表板 - Flask 应用主程序
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 限制
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'ogg'}

# 创建必要的目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)

# 全局变量 - 在实际应用中应使用数据库
analysis_history = []
current_model = None
current_feature_extractor = None


def allowed_file(filename):
    """检查文件类型"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    处理音频上传和情感分析
    """
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'不支持的文件格式。允许: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400
    
    try:
        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
        file.save(filepath)
        
        # 进行情感分析（这里需要调用实际的分析函数）
        # result = analyze_emotion(filepath, current_model, current_feature_extractor)
        
        # 示例结果
        result = {
            'filename': filename,
            'upload_time': datetime.now().isoformat(),
            'emotion': 'happy',
            'confidence': 0.87,
            'gender': 'male',
            'file_path': filepath
        }
        
        # 保存到历史记录
        analysis_history.append(result)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def get_history():
    """获取分析历史"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    start = (page - 1) * per_page
    end = start + per_page
    
    total = len(analysis_history)
    items = analysis_history[start:end]
    
    return jsonify({
        'items': items,
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page
    })


@app.route('/api/statistics')
def get_statistics():
    """获取统计信息"""
    if not analysis_history:
        return jsonify({
            'total_analyses': 0,
            'emotion_distribution': {},
            'gender_distribution': {},
            'average_confidence': 0
        })
    
    emotions = {}
    genders = {}
    confidences = []
    
    for record in analysis_history:
        # 情感分布
        emotion = record.get('emotion', 'unknown')
        emotions[emotion] = emotions.get(emotion, 0) + 1
        
        # 性别分布
        gender = record.get('gender', 'unknown')
        genders[gender] = genders.get(gender, 0) + 1
        
        # 置信度
        confidence = record.get('confidence', 0)
        if isinstance(confidence, (int, float)):
            confidences.append(confidence)
    
    return jsonify({
        'total_analyses': len(analysis_history),
        'emotion_distribution': emotions,
        'gender_distribution': genders,
        'average_confidence': float(np.mean(confidences)) if confidences else 0,
        'confidence_std': float(np.std(confidences)) if confidences else 0
    })


@app.route('/api/visualization/<int:record_id>')
def get_visualization(record_id):
    """获取指定分析的可视化"""
    if record_id >= len(analysis_history):
        return jsonify({'error': '记录不存在'}), 404
    
    record = analysis_history[record_id]
    
    # 生成图表 (示例)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    emotions = ['angry', 'calm', 'fearful', 'happy', 'sad']
    scores = np.random.rand(5)
    
    ax.bar(emotions, scores, color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_ylabel('Confidence Score')
    ax.set_title(f'Emotion Analysis: {record.get("filename", "Unknown")}')
    ax.set_ylim([0, 1])
    
    # 转换为PNG
    img = io.BytesIO()
    FigureCanvasAgg(fig).print_png(img)
    img.seek(0)
    
    plt.close(fig)
    
    # 转换为base64
    img_base64 = base64.b64encode(img.getvalue()).decode()
    
    return jsonify({
        'image': f'data:image/png;base64,{img_base64}',
        'record': record
    })


@app.route('/api/delete/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    """删除分析记录"""
    if record_id >= len(analysis_history):
        return jsonify({'error': '记录不存在'}), 404
    
    record = analysis_history.pop(record_id)
    
    # 删除文件
    if 'file_path' in record and os.path.exists(record['file_path']):
        os.remove(record['file_path'])
    
    return jsonify({'success': True})


@app.route('/api/export')
def export_data():
    """导出分析数据为CSV"""
    import csv
    
    output = io.StringIO()
    if analysis_history:
        writer = csv.DictWriter(output, fieldnames=analysis_history[0].keys())
        writer.writeheader()
        writer.writerows(analysis_history)
    
    response_data = output.getvalue()
    
    return response_data, 200, {
        'Content-Disposition': 'attachment; filename=emotion_analysis_history.csv',
        'Content-Type': 'text/csv'
    }


@app.route('/api/batch-analysis', methods=['POST'])
def batch_analysis():
    """
    批量分析多个文件
    """
    if 'files' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
                file.save(filepath)
                
                # 分析
                result = {
                    'filename': filename,
                    'status': 'success',
                    'emotion': 'happy',
                    'confidence': 0.85
                }
                
                analysis_history.append(result)
                results.append(result)
            
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': str(e)
                })
    
    return jsonify({
        'total': len(files),
        'successful': sum(1 for r in results if r.get('status') == 'success'),
        'results': results
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '页面不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
