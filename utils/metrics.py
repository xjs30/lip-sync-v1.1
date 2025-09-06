import numpy as np
import cv2
from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score

class LipSyncMetrics:
    """唇同步模型评估指标计算工具"""
    
    def __init__(self):
        # 唇部特征点索引（68点模型中的唇部区域）
        self.lip_landmark_indices = list(range(48, 68))
        
    def compute_landmark_distance(self, pred_landmarks, gt_landmarks):
        """
        计算预测和真实面部特征点之间的平均距离
        
        参数:
            pred_landmarks: 预测的面部特征点，形状为 (T, 136) 或 (B, T, 136)
            gt_landmarks: 真实的面部特征点，形状同上
            
        返回:
            平均距离值
        """
        # 确保输入是numpy数组
        pred = np.asarray(pred_landmarks)
        gt = np.asarray(gt_landmarks)
        
        # 处理批次维度
        if pred.ndim == 3:
            # 批次处理
            total_dist = 0.0
            count = 0
            for b in range(pred.shape[0]):
                for t in range(pred.shape[1]):
                    dist = self._compute_single_landmark_distance(
                        pred[b, t], gt[b, t]
                    )
                    total_dist += dist
                    count += 1
            return total_dist / count if count > 0 else 0.0
        else:
            # 单样本处理
            total_dist = 0.0
            count = 0
            for t in range(pred.shape[0]):
                dist = self._compute_single_landmark_distance(
                    pred[t], gt[t]
                )
                total_dist += dist
                count += 1
            return total_dist / count if count > 0 else 0.0
    
    def _compute_single_landmark_distance(self, pred, gt):
        """计算单帧特征点距离"""
        # 重塑为 (68, 2) 格式
        pred_points = pred.reshape(-1, 2)
        gt_points = gt.reshape(-1, 2)
        
        # 仅计算唇部特征点的距离
        lip_pred = pred_points[self.lip_landmark_indices]
        lip_gt = gt_points[self.lip_landmark_indices]
        
        # 计算欧氏距离
        dist = np.mean([
            distance.euclidean(p, g) 
            for p, g in zip(lip_pred, lip_gt)
        ])
        
        return dist
    
    def compute_ssim(self, pred_frames, gt_frames):
        """
        计算预测和真实视频帧之间的结构相似性指数(SSIM)
        
        参数:
            pred_frames: 预测的视频帧，形状为 (T, H, W, C) 或 (B, T, H, W, C)
            gt_frames: 真实的视频帧，形状同上
            
        返回:
            平均SSIM值
        """
        pred = np.asarray(pred_frames)
        gt = np.asarray(gt_frames)
        
        # 确保输入是RGB格式且值在0-255范围内
        if pred.max() <= 1.0:
            pred = (pred * 255).astype(np.uint8)
        if gt.max() <= 1.0:
            gt = (gt * 255).astype(np.uint8)
        
        # 转为灰度图处理
        def to_gray(frame):
            if frame.shape[-1] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return frame
        
        # 处理批次维度
        if pred.ndim == 5:
            # 批次处理
            total_ssim = 0.0
            count = 0
            for b in range(pred.shape[0]):
                for t in range(pred.shape[1]):
                    pred_gray = to_gray(pred[b, t])
                    gt_gray = to_gray(gt[b, t])
                    
                    # 确保尺寸一致
                    if pred_gray.shape != gt_gray.shape:
                        pred_gray = cv2.resize(pred_gray, (gt_gray.shape[1], gt_gray.shape[0]))
                        
                    s = ssim(pred_gray, gt_gray)
                    total_ssim += s
                    count += 1
            return total_ssim / count if count > 0 else 0.0
        else:
            # 单样本处理
            total_ssim = 0.0
            count = 0
            for t in range(pred.shape[0]):
                pred_gray = to_gray(pred[t])
                gt_gray = to_gray(gt[t])
                
                # 确保尺寸一致
                if pred_gray.shape != gt_gray.shape:
                    pred_gray = cv2.resize(pred_gray, (gt_gray.shape[1], gt_gray.shape[0]))
                    
                s = ssim(pred_gray, gt_gray)
                total_ssim += s
                count += 1
            return total_ssim / count if count > 0 else 0.0
    
    def compute_sync_score(self, audio_features, video_features, labels=None):
        """
        计算音频和视频的同步分数
        
        参数:
            audio_features: 音频特征，形状为 (B, T, D)
            video_features: 视频特征，形状为 (B, T, D)
            labels: 真实标签，1表示同步，0表示不同步
            
        返回:
            同步分数（余弦相似度均值或分类准确率）
        """
        audio = np.asarray(audio_features)
        video = np.asarray(video_features)
        
        # 计算余弦相似度
        if audio.ndim == 3 and video.ndim == 3:
            # 批次和时间维度
            cos_sim = []
            for b in range(audio.shape[0]):
                for t in range(audio.shape[1]):
                    a = audio[b, t]
                    v = video[b, t]
                    # 计算余弦相似度
                    sim = np.dot(a, v) / (np.linalg.norm(a) * np.linalg.norm(v) + 1e-8)
                    cos_sim.append(sim)
            
            if labels is not None:
                # 如果有标签，使用分类准确率
                labels = np.asarray(labels).flatten()
                preds = np.array(cos_sim) > 0.5  # 以0.5为阈值
                return accuracy_score(labels, preds)
            else:
                # 否则返回平均余弦相似度
                return np.mean(cos_sim) if cos_sim else 0.0
        else:
            return 0.0
    
    def compute_temporal_consistency(self, frames):
        """
        计算视频帧序列的时间一致性
        
        参数:
            frames: 视频帧序列，形状为 (T, H, W, C) 或 (B, T, H, W, C)
            
        返回:
            时间一致性分数（相邻帧差异的均值）
        """
        frames = np.asarray(frames)
        
        # 转为灰度图处理
        def to_gray(frame):
            if frame.shape[-1] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return frame / frame.max() if frame.max() > 0 else frame
        
        # 处理批次维度
        if frames.ndim == 5:
            # 批次处理
            total_diff = 0.0
            count = 0
            for b in range(frames.shape[0]):
                seq_diff = self._compute_sequence_temporal_diff(frames[b], to_gray)
                total_diff += seq_diff
                count += 1
            return 1.0 - (total_diff / count) if count > 0 else 0.0  # 1减去差异，得到一致性分数
        else:
            # 单样本处理
            seq_diff = self._compute_sequence_temporal_diff(frames, to_gray)
            return 1.0 - seq_diff  # 1减去差异，得到一致性分数
    
    def _compute_sequence_temporal_diff(self, sequence, preprocess_fn):
        """计算单个序列的时间差异"""
        if sequence.shape[0] < 2:
            return 0.0  # 序列太短，无法计算
            
        total_diff = 0.0
        for t in range(sequence.shape[0] - 1):
            frame1 = preprocess_fn(sequence[t])
            frame2 = preprocess_fn(sequence[t+1])
            
            # 确保尺寸一致
            if frame1.shape != frame2.shape:
                frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
                
            # 计算帧差异
            diff = np.mean(np.abs(frame1 - frame2)) / 255.0 if frame1.max() > 1 else np.mean(np.abs(frame1 - frame2))
            total_diff += diff
            
        return total_diff / (sequence.shape[0] - 1)
    
    def compute_lip_movement_similarity(self, pred_landmarks, gt_landmarks):
        """
        计算唇部运动模式的相似度
        
        参数:
            pred_landmarks: 预测的面部特征点，形状为 (T, 136) 或 (B, T, 136)
            gt_landmarks: 真实的面部特征点，形状同上
            
        返回:
            唇部运动相似度分数
        """
        pred = np.asarray(pred_landmarks)
        gt = np.asarray(gt_landmarks)
        
        # 处理批次维度
        if pred.ndim == 3:
            # 批次处理
            total_sim = 0.0
            count = 0
            for b in range(pred.shape[0]):
                sim = self._compute_single_sequence_movement_similarity(
                    pred[b], gt[b]
                )
                total_sim += sim
                count += 1
            return total_sim / count if count > 0 else 0.0
        else:
            # 单样本处理
            return self._compute_single_sequence_movement_similarity(pred, gt)
    
    def _compute_single_sequence_movement_similarity(self, pred, gt):
        """计算单个序列的唇部运动相似度"""
        if pred.shape[0] < 2:
            return 0.0  # 序列太短，无法计算
            
        # 提取唇部特征点
        def extract_lip_points(landmarks):
            points = landmarks.reshape(-1, 2)
            return points[self.lip_landmark_indices]
            
        # 计算运动向量（相邻帧差异）
        pred_movement = []
        gt_movement = []
        
        for t in range(pred.shape[0] - 1):
            # 预测运动
            pred_lip_prev = extract_lip_points(pred[t])
            pred_lip_next = extract_lip_points(pred[t+1])
            pred_move = np.mean(pred_lip_next - pred_lip_prev, axis=0)
            pred_movement.append(pred_move)
            
            # 真实运动
            gt_lip_prev = extract_lip_points(gt[t])
            gt_lip_next = extract_lip_points(gt[t+1])
            gt_move = np.mean(gt_lip_next - gt_lip_prev, axis=0)
            gt_movement.append(gt_move)
        
        # 计算运动向量的余弦相似度
        pred_movement = np.array(pred_movement)
        gt_movement = np.array(gt_movement)
        
        # 标准化
        pred_norm = np.linalg.norm(pred_movement, axis=1, keepdims=True) + 1e-8
        gt_norm = np.linalg.norm(gt_movement, axis=1, keepdims=True) + 1e-8
        
        pred_normalized = pred_movement / pred_norm
        gt_normalized = gt_movement / gt_norm
        
        # 计算余弦相似度
        cos_sim = np.mean([
            np.dot(p, g) for p, g in zip(pred_normalized, gt_normalized)
        ])
        
        return (cos_sim + 1) / 2  # 归一化到0-1范围
    
    def compute_all_metrics(self, pred_data, gt_data, audio_features=None):
        """
        计算所有评估指标
        
        参数:
            pred_data: 预测数据字典，包含 'video', 'landmarks', 'features'
            gt_data: 真实数据字典，包含 'video', 'landmarks'
            audio_features: 音频特征，用于计算同步分数
            
        返回:
            包含所有指标的字典
        """
        metrics = {}
        
        # 1. 特征点距离
        metrics['avg_landmark_distance'] = self.compute_landmark_distance(
            pred_data['landmarks'], 
            gt_data['landmarks']
        )
        
        # 2. SSIM
        metrics['avg_ssim'] = self.compute_ssim(
            pred_data['video'], 
            gt_data['video']
        )
        
        # 3. 时间一致性
        metrics['temporal_consistency'] = self.compute_temporal_consistency(
            pred_data['video']
        )
        
        # 4. 唇部运动相似度
        metrics['lip_movement_similarity'] = self.compute_lip_movement_similarity(
            pred_data['landmarks'], 
            gt_data['landmarks']
        )
        
        # 5. 同步分数（如果提供了音频特征）
        if audio_features is not None and 'features' in pred_data:
            metrics['sync_score'] = self.compute_sync_score(
                audio_features, 
                pred_data['features']
            )
        
        return metrics
