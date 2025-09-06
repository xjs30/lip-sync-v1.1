import os
import yaml
import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

def setup_logging(log_dir="logs", log_name="training.log"):
    """设置日志系统"""
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件路径
    log_path = os.path.join(log_dir, log_name)
    
    # 创建日志器
    logger = logging.getLogger("lip_sync")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def set_seed(seed=42):
    """设置随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况下
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path

def save_checkpoint(model, optimizer, epoch, loss, config, is_best=False):
    """保存模型检查点"""
    checkpoint_dir = ensure_dir(os.path.join(
        config['training']['checkpoint_dir'], 
        config['model']['name']
    ))
    
    # 检查点文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # 保存检查点
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, checkpoint_path)
    
    # 如果是最佳模型，创建软链接
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        if os.path.exists(best_path):
            os.remove(best_path)
        os.symlink(checkpoint_name, best_path)
    
    return checkpoint_path

def load_checkpoint(model, optimizer, config=None, checkpoint_path=None):
    """加载模型检查点"""
    # 如果未指定路径，尝试加载最佳检查点
    if checkpoint_path is None and config is not None:
        checkpoint_dir = os.path.join(
            config['training']['checkpoint_dir'], 
            config['model']['name']
        )
        checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        
        # 如果最佳检查点不存在，尝试加载最新的检查点
        if not os.path.exists(checkpoint_path):
            checkpoints = [f for f in os.listdir(checkpoint_dir) 
                         if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort(reverse=True)
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器参数（如果提供了优化器）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))

def plot_metrics(metrics_history, save_dir, title="训练指标"):
    """绘制训练指标图表"""
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, "training_metrics.png")
    
    # 创建一个包含多个子图的图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)
    
    # 1. 训练损失和验证损失
    ax1 = axes[0, 0]
    ax1.plot(metrics_history['train_loss'], label='训练损失')
    ax1.plot(metrics_history['val_loss'], label='验证损失')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 同步分数
    if 'sync_score' in metrics_history and metrics_history['sync_score']:
        ax2 = axes[0, 1]
        ax2.plot(metrics_history['sync_score'], label='同步分数', color='green')
        ax2.set_title('同步分数曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('同步分数')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)
    
    # 3. SSIM
    if 'ssim' in metrics_history and metrics_history['ssim']:
        ax3 = axes[1, 0]
        ax3.plot(metrics_history['ssim'], label='SSIM', color='orange')
        ax3.set_title('结构相似性曲线')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('SSIM')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def preprocess_video_for_inference(video_path, processor, max_length=None):
    """预处理推理用的视频"""
    try:
        # 加载视频
        frames, fps = processor.load_video(video_path)
        
        # 提取面部区域
        face_frames, landmarks = processor.extract_face_region(frames)
        
        # 如果没有检测到人脸，使用原始帧
        if face_frames is None:
            face_frames = frames
            landmarks = np.zeros((len(frames), 136))  # 68点×2坐标
        
        # 预处理帧
        face_tensor = processor.preprocess_frames(face_frames)
        
        # 调整长度
        if max_length and len(face_tensor) > max_length:
            face_tensor = face_tensor[:max_length]
            landmarks = landmarks[:max_length]
        
        return {
            'video': face_tensor,
            'landmarks': landmarks,
            'fps': fps,
            'length': len(face_tensor)
        }
    except Exception as e:
        print(f"视频预处理失败: {str(e)}")
        return None

def preprocess_audio_for_inference(audio_path, processor, target_length=None):
    """预处理推理用的音频"""
    try:
        # 加载音频
        audio, sr = processor.load_audio(audio_path)
        
        # 提取特征
        audio_features = processor.extract_features_from_waveform(audio, sr)
        
        # 调整长度以匹配视频
        if target_length is not None:
            # 如果音频特征长度小于目标长度，重复填充
            if len(audio_features) < target_length:
                repeat_times = (target_length // len(audio_features)) + 1
                audio_features = np.tile(audio_features, (repeat_times, 1))[:target_length]
            # 如果音频特征长度大于目标长度，截断
            elif len(audio_features) > target_length:
                audio_features = audio_features[:target_length]
        
        return {
            'features': audio_features,
            'waveform': audio,
            'sample_rate': sr
        }
    except Exception as e:
        print(f"音频预处理失败: {str(e)}")
        return None

def save_generated_video(frames, output_path, fps=30, audio_path=None):
    """保存生成的视频，可选添加音频"""
    try:
        import cv2
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        # 确保输出目录存在
        ensure_dir(os.path.dirname(output_path))
        
        # 临时视频文件（无音频）
        temp_video = output_path.replace('.mp4', '_temp.mp4')
        
        # 获取帧大小
        height, width = frames[0].shape[:2]
        
        # 写入视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        for frame in frames:
            # 转换为BGR格式（OpenCV默认）
            if frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        
        # 如果提供了音频，合并音频和视频
        if audio_path and os.path.exists(audio_path):
            video_clip = VideoFileClip(temp_video)
            audio_clip = AudioFileClip(audio_path)
            
            # 确保音频长度与视频匹配
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            elif audio_clip.duration < video_clip.duration:
                # 如果音频较短，循环播放
                audio_clip = audio_clip.loop(duration=video_clip.duration)
            
            # 合并并保存
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            # 清理
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            # 删除临时文件
            if os.path.exists(temp_video):
                os.remove(temp_video)
        else:
            # 没有音频，直接重命名临时文件
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(temp_video, output_path)
        
        return output_path
    except Exception as e:
        print(f"保存视频失败: {str(e)}")
        return None
