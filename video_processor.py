import os
import numpy as np
import cv2
import dlib
import torch
import tempfile
from tqdm import tqdm
from skimage.transform import resize

class VideoProcessorV2:
    """增强版视频处理器，用于面部和唇形特征提取"""
    def __init__(self, image_size=128, face_detector_path=None, max_frames=500):
        self.image_size = image_size  # 处理后的图像大小
        self.max_frames = max_frames  # 最大处理帧数
        self.face_detector = None
        self.landmark_predictor = None
        
        # 加载人脸检测和特征点预测模型
        if face_detector_path:
            self._load_face_models(face_detector_path)
        else:
            print("警告: 未提供人脸检测器路径，将使用OpenCV的默认检测器")
            self._load_default_face_detector()
        
        # 唇部区域索引（68点模型）
        self.lip_indices = list(range(48, 68))  # 唇部特征点索引
        self.mouth_roi_expand = 1.2  # 唇部区域扩展系数
    
    def _load_face_models(self, landmark_path):
        """加载dlib的人脸检测和特征点模型"""
        try:
            # 人脸检测器
            self.face_detector = dlib.get_frontal_face_detector()
            
            # 特征点预测器
            self.landmark_predictor = dlib.shape_predictor(landmark_path)
            print("成功加载人脸检测和特征点模型")
        except Exception as e:
            print(f"加载人脸模型失败: {str(e)}")
            print("将使用OpenCV的默认检测器")
            self._load_default_face_detector()
    
    def _load_default_face_detector(self,):
        """加载OpenCV的默认人脸检测器"""
        try:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("成功加载OpenCV人脸检测器")
        except Exception as e:
            print(f"加载OpenCV人脸检测器失败: {str(e)}")
            self.face_detector = None
    
    def load_video(self, video_path, max_frames=None):
        """
        加载视频并提取帧
        
        参数:
            video_path: 视频文件路径
            max_frames: 最大提取帧数，None则使用默认值
            
        返回:
            frames: 视频帧列表，形状为 (T, H, W, C)
            fps: 视频帧率
        """
        max_frames = max_frames or self.max_frames
        
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算需要提取的帧数
            frame_interval = 1
            if total_frames > max_frames:
                frame_interval = max(1, total_frames // max_frames)
            
            # 提取帧
            frames = []
            frame_count = 0
            
            with tqdm(total=min(total_frames, max_frames), desc="加载视频帧") as pbar:
                while cap.isOpened() and len(frames) < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 每隔frame_interval提取一帧
                    if frame_count % frame_interval == 0:
                        # 转换为RGB格式
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                        pbar.update(1)
                    
                    frame_count += 1
            
            cap.release()
            return np.array(frames), fps
        except Exception as e:
            print(f"加载视频失败: {str(e)}")
            return None, None
    
    def save_video(self, frames, output_path, fps=30, audio_path=None):
        """
        保存视频帧为视频文件
        
        参数:
            frames: 视频帧列表，形状为 (T, H, W, C)
            output_path: 输出视频路径
            fps: 帧率
            audio_path: 可选音频路径，用于合并音频
            
        返回:
            是否保存成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 如果是RGB格式，转换为BGR
            if frames.shape[-1] == 3:
                frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
            else:
                frames_bgr = frames
            
            # 获取帧大小
            height, width = frames_bgr[0].shape[:2]
            
            # 定义编码器并创建VideoWriter对象
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 写入帧
            for frame in frames_bgr:
                out.write(frame)
            
            out.release()
            
            # 如果提供了音频，合并音频和视频
            if audio_path and os.path.exists(audio_path):
                try:
                    from moviepy.editor import VideoFileClip, AudioFileClip
                    
                    # 加载视频和音频
                    video_clip = VideoFileClip(output_path)
                    audio_clip = AudioFileClip(audio_path)
                    
                    # 调整音频长度以匹配视频
                    if audio_clip.duration > video_clip.duration:
                        audio_clip = audio_clip.subclip(0, video_clip.duration)
                    elif audio_clip.duration < video_clip.duration:
                        # 循环音频以匹配视频长度
                        audio_clip = audio_clip.loop(duration=video_clip.duration)
                    
                    # 合并音频并保存
                    final_clip = video_clip.set_audio(audio_clip)
                    final_clip.write_videofile(
                        output_path, 
                        codec="libx264", 
                        audio_codec="aac",
                        overwrite_output=True
                    )
                    
                    # 清理
                    video_clip.close()
                    audio_clip.close()
                    final_clip.close()
                except Exception as e:
                    print(f"合并音频失败: {str(e)}")
            
            return True
        except Exception as e:
            print(f"保存视频失败: {str(e)}")
            return False
    
    def detect_face(self, frame):
        """
        检测图像中的人脸
        
        参数:
            frame: 输入图像，形状为 (H, W, C)
            
        返回:
            人脸边界框 (x1, y1, x2, y2) 或 None
        """
        if self.face_detector is None:
            return None
            
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        try:
            # 使用dlib检测器
            if hasattr(self.face_detector, 'run'):
                dets = self.face_detector(gray, 1)
                if len(dets) > 0:
                    # 取最大的人脸
                    largest_det = max(dets, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))
                    return (largest_det.left(), largest_det.top(), largest_det.right(), largest_det.bottom())
            
            # 使用OpenCV的Haar检测器
            else:
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    # 取最大的人脸
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    return (x, y, x + w, y + h)
        
        except Exception as e:
            print(f"人脸检测失败: {str(e)}")
        
        return None
    
    def detect_landmarks(self, frame, bbox=None):
        """
        检测面部特征点
        
        参数:
            frame: 输入图像，形状为 (H, W, C)
            bbox: 人脸边界框，None则自动检测
            
        返回:
            68个特征点的坐标，形状为 (68, 2) 或 None
        """
        if self.landmark_predictor is None:
            return None
            
        # 如果没有边界框，先检测人脸
        if bbox is None:
            bbox = self.detect_face(frame)
            if bbox is None:
                return None
        
        # 转换为dlib的矩形格式
        x1, y1, x2, y2 = bbox
        dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        try:
            # 预测特征点
            shape = self.landmark_predictor(gray, dlib_rect)
            
            # 提取坐标
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            return landmarks
        except Exception as e:
            print(f"特征点检测失败: {str(e)}")
            return None
    
    def extract_face_region(self, frames):
        """
        从视频帧序列中提取人脸区域和特征点
        
        参数:
            frames: 视频帧列表，形状为 (T, H, W, C)
            
        返回:
            face_frames: 裁剪并对齐的人脸区域，形状为 (T, image_size, image_size, 3)
            landmarks: 面部特征点，形状为 (T, 136) 即 (T, 68*2)
        """
        if frames is None or len(frames) == 0:
            return None, None
        
        face_frames = []
        all_landmarks = []
        
        # 跟踪前一帧的人脸位置，用于平滑跟踪
        prev_bbox = None
        
        with tqdm(total=len(frames), desc="提取人脸区域") as pbar:
            for frame in frames:
                # 检测人脸
                bbox = self.detect_face(frame)
                
                # 如果没检测到人脸，尝试使用前一帧的位置
                if bbox is None and prev_bbox is not None:
                    bbox = prev_bbox
                
                if bbox is None:
                    # 如果仍然没有检测到人脸，使用整帧
                    face_img = resize(frame, (self.image_size, self.image_size))
                    landmarks = np.zeros(136)  # 填充零特征点
                else:
                    # 更新前一帧边界框
                    prev_bbox = bbox
                    
                    # 裁剪人脸区域（适当扩展）
                    x1, y1, x2, y2 = bbox
                    h, w = y2 - y1, x2 - x1
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # 扩展边界框
                    expand = max(w, h) * 0.5 * self.mouth_roi_expand
                    x1 = max(0, int(center_x - expand))
                    y1 = max(0, int(center_y - expand))
                    x2 = min(frame.shape[1], int(center_x + expand))
                    y2 = min(frame.shape[0], int(center_y + expand))
                    
                    # 裁剪并调整大小
                    face_roi = frame[y1:y2, x1:x2]
                    face_img = resize(face_roi, (self.image_size, self.image_size))
                    
                    # 检测特征点
                    landmarks = self.detect_landmarks(frame, bbox)
                    if landmarks is not None:
                        # 将特征点坐标归一化到[0, 1]
                        h_orig, w_orig = frame.shape[:2]
                        landmarks[:, 0] /= w_orig  # x坐标归一化
                        landmarks[:, 1] /= h_orig  # y坐标归一化
                        landmarks = landmarks.flatten()  # 展平为136维向量
                    else:
                        landmarks = np.zeros(136)
                
                # 添加到结果列表
                face_frames.append(face_img)
                all_landmarks.append(landmarks)
                
                pbar.update(1)
        
        return np.array(face_frames), np.array(all_landmarks)
    
    def extract_lip_region(self, frames, landmarks=None):
        """
        从视频帧中提取唇部区域
        
        参数:
            frames: 视频帧列表，形状为 (T, H, W, C)
            landmarks: 面部特征点，None则自动检测
            
        返回:
            lip_frames: 唇部区域图像，形状为 (T, lip_size, lip_size, 3)
        """
        lip_size = self.image_size // 2  # 唇部区域大小为面部的一半
        
        if frames is None or len(frames) == 0:
            return None
        
        lip_frames = []
        
        # 如果没有提供特征点，自动检测
        if landmarks is None:
            _, landmarks = self.extract_face_region(frames)
            if landmarks is None:
                return None
        
        with tqdm(total=len(frames), desc="提取唇部区域") as pbar:
            for i, frame in enumerate(frames):
                # 获取当前帧的特征点
                frame_landmarks = landmarks[i].reshape(-1, 2)
                h, w = frame.shape[:2]
                
                # 将归一化的特征点坐标转换回像素坐标
                lip_landmarks = frame_landmarks[self.lip_indices]
                lip_landmarks[:, 0] *= w  # x坐标
                lip_landmarks[:, 1] *= h  # y坐标
                
                # 计算唇部边界框
                min_x = np.min(lip_landmarks[:, 0])
                max_x = np.max(lip_landmarks[:, 0])
                min_y = np.min(lip_landmarks[:, 1])
                max_y = np.max(lip_landmarks[:, 1])
                
                # 扩展边界框
                width = max_x - min_x
                height = max_y - min_y
                center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
                expand = max(width, height) * 0.5 * self.mouth_roi_expand
                
                # 计算扩展后的边界框
                x1 = max(0, int(center_x - expand))
                y1 = max(0, int(center_y - expand))
                x2 = min(w, int(center_x + expand))
                y2 = min(h, int(center_y + expand))
                
                # 裁剪唇部区域并调整大小
                if x2 > x1 and y2 > y1:
                    lip_roi = frame[y1:y2, x1:x2]
                    lip_img = resize(lip_roi, (lip_size, lip_size))
                else:
                    # 如果无法提取唇部区域，使用零填充
                    lip_img = np.zeros((lip_size, lip_size, 3))
                
                lip_frames.append(lip_img)
                pbar.update(1)
        
        return np.array(lip_frames)
    
    def preprocess_frames(self, frames, normalize=True):
        """
        预处理视频帧，转换为模型输入格式
        
        参数:
            frames: 视频帧列表，形状为 (T, H, W, C)
            normalize: 是否归一化到[-1, 1]
            
        返回:
            处理后的张量，形状为 (T, C, H, W)
        """
        if frames is None:
            return None
            
        # 转换为float32
        frames = frames.astype(np.float32)
        
        # 如果需要，归一化到[-1, 1]
        if normalize:
            frames = (frames / 255.0) * 2 - 1
        
        # 调整通道顺序 (T, H, W, C) -> (T, C, H, W)
        if frames.ndim == 4:
            frames = frames.transpose(0, 3, 1, 2)
        
        # 转换为PyTorch张量
        return torch.from_numpy(frames)
    
    def postprocess_frames(self, tensor, denormalize=True):
        """
        将模型输出的张量转换为可显示的视频帧
        
        参数:
            tensor: 模型输出张量，形状为 (T, C, H, W) 或 (B, T, C, H, W)
            denormalize: 是否从[-1, 1]反归一化到[0, 255]
            
        返回:
            视频帧列表，形状为 (T, H, W, C)
        """
        # 如果有批次维度，取第一个样本
        if tensor.ndim == 5:
            tensor = tensor[0]
        
        # 转换为numpy数组
        frames = tensor.cpu().detach().numpy()
        
        # 调整通道顺序 (T, C, H, W) -> (T, H, W, C)
        frames = frames.transpose(0, 2, 3, 1)
        
        # 如果需要，反归一化到[0, 255]
        if denormalize:
            frames = ((frames + 1) / 2) * 255
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        
        return frames
