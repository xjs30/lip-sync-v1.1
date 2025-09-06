import os
import numpy as np
import librosa
import soundfile as sf
import torch
import tempfile
from scipy.signal import resample

class AudioProcessorV2:
    """增强版音频处理器，支持多语言音频特征提取"""
    def __init__(self, sample_rate=16000, n_mfcc=40, max_duration=10):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_duration = max_duration  # 最大处理时长（秒）
        self.max_length = int(sample_rate * max_duration)  # 最大样本数
        
        # 预加重系数
        self.preemphasis = 0.97
        
        # 梅尔频谱参数
        self.n_fft = 512
        self.hop_length = int(sample_rate * 0.01)  # 10ms
        self.win_length = int(sample_rate * 0.025)  # 25ms
        self.n_mels = 80
        
        # 静音检测阈值
        self.silence_threshold = 0.01
        
    def load_audio(self, audio_path, resample=True):
        """
        加载音频文件
        
        参数:
            audio_path: 音频文件路径
            resample: 是否重采样到目标采样率
            
        返回:
            audio: 音频波形数据
            sr: 采样率
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=None)
            
            # 如果需要，重采样
            if resample and sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            
            # 确保音频是单声道
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            
            # 截断过长的音频
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            
            return audio, sr
        except Exception as e:
            print(f"加载音频失败: {str(e)}")
            return None, None
    
    def save_audio(self, audio, output_path, sr=None):
        """保存音频文件"""
        try:
            sr = sr or self.sample_rate
            sf.write(output_path, audio, sr)
            return True
        except Exception as e:
            print(f"保存音频失败: {str(e)}")
            return False
    
    def preprocess_waveform(self, audio):
        """预处理音频波形"""
        # 预加重
        audio = np.append(audio[0], audio[1:] - self.preemphasis * audio[:-1])
        
        # 去除静音
        audio, _ = self._remove_silence(audio)
        
        # 归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def _remove_silence(self, audio, frame_length=2048, hop_length=512):
        """去除音频中的静音部分"""
        # 计算能量
        energy = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        ).squeeze()
        
        # 找到非静音帧
        non_silence_indices = np.where(energy > self.silence_threshold)[0]
        if len(non_silence_indices) == 0:
            return audio, (0, len(audio))
        
        # 计算非静音区域
        start_idx = max(0, non_silence_indices[0] * hop_length - frame_length // 2)
        end_idx = min(len(audio), (non_silence_indices[-1] + 1) * hop_length)
        
        return audio[start_idx:end_idx], (start_idx, end_idx)
    
    def extract_mfcc(self, audio, sr=None):
        """提取MFCC特征"""
        sr = sr or self.sample_rate
        
        # 计算MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=50,  # 语音通常在50Hz以上
            fmax=sr // 2
        )
        
        # 转置为 (时间步, 特征维度)
        mfcc = mfcc.T
        
        # 特征标准化（均值为0，方差为1）
        mean = np.mean(mfcc, axis=0, keepdims=True)
        std = np.std(mfcc, axis=0, keepdims=True) + 1e-8
        mfcc = (mfcc - mean) / std
        
        return mfcc
    
    def extract_mel_spectrogram(self, audio, sr=None):
        """提取梅尔频谱特征"""
        sr = sr or self.sample_rate
        
        # 计算梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=50,
            fmax=sr // 2
        )
        
        # 转换为对数刻度
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 转置为 (时间步, 特征维度)
        mel_spec = mel_spec.T
        
        # 标准化
        mean = np.mean(mel_spec, axis=0, keepdims=True)
        std = np.std(mel_spec, axis=0, keepdims=True) + 1e-8
        mel_spec = (mel_spec - mean) / std
        
        return mel_spec
    
    def extract_pitch_features(self, audio, sr=None):
        """提取基频(pitch)特征"""
        sr = sr or self.sample_rate
        
        # 计算基频
        pitch, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=self.hop_length
        )
        
        # 处理NaN值（静音区域）
        if pitch is not None:
            pitch = np.nan_to_num(pitch)
            # 标准化
            mean = np.mean(pitch)
            std = np.std(pitch) + 1e-8
            pitch = (pitch - mean) / std
            # 扩展为二维特征
            return pitch.reshape(-1, 1)
        else:
            # 如果无法计算pitch，返回零向量
            length = int(np.ceil(len(audio) / self.hop_length))
            return np.zeros((length, 1))
    
    def extract_features_from_waveform(self, audio, sr=None, include_pitch=True):
        """从音频波形中提取综合特征"""
        # 预处理波形
        audio = self.preprocess_waveform(audio)
        
        if len(audio) == 0:
            return np.zeros((0, self.n_mfcc + (1 if include_pitch else 0)))
        
        # 提取MFCC特征
        mfcc = self.extract_mfcc(audio, sr)
        
        # 提取梅尔频谱（作为辅助特征）
        mel_spec = self.extract_mel_spectrogram(audio, sr)
        
        # 确保MFCC和梅尔频谱长度一致
        min_length = min(len(mfcc), len(mel_spec))
        mfcc = mfcc[:min_length]
        mel_spec = mel_spec[:min_length]
        
        # 融合MFCC和梅尔频谱的统计特征
        mfcc_mean = np.mean(mfcc, axis=0, keepdims=True)
        mfcc_std = np.std(mfcc, axis=0, keepdims=True)
        mfcc_max = np.max(mfcc, axis=0, keepdims=True)
        
        # 扩展统计特征到所有时间步
        mfcc_stats = np.concatenate([
            np.repeat(mfcc_mean, min_length, axis=0),
            np.repeat(mfcc_std, min_length, axis=0),
            np.repeat(mfcc_max, min_length, axis=0)
        ], axis=1)
        
        # 主特征：MFCC + 梅尔频谱
        features = np.concatenate([mfcc, mel_spec], axis=1)
        
        # 可选：添加基频特征
        if include_pitch:
            pitch = self.extract_pitch_features(audio, sr)
            # 确保长度一致
            if len(pitch) > min_length:
                pitch = pitch[:min_length]
            elif len(pitch) < min_length:
                pitch = np.pad(pitch, ((0, min_length - len(pitch)), (0, 0)), mode='edge')
            features = np.concatenate([features, pitch], axis=1)
        
        return features
    
    def extract_features(self, audio_path, include_pitch=True):
        """从音频文件中提取综合特征"""
        # 加载音频
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            return None
        
        # 提取特征
        return self.extract_features_from_waveform(audio, sr, include_pitch)
    
    def align_audio_with_video(self, audio, audio_sr, video_fps, video_length):
        """
        将音频与视频对齐
        
        参数:
            audio: 音频波形
            audio_sr: 音频采样率
            video_fps: 视频帧率
            video_length: 视频帧数
            
        返回:
            对齐后的音频特征
        """
        # 计算目标音频长度（与视频匹配）
        target_duration = video_length / video_fps
        target_samples = int(audio_sr * target_duration)
        
        # 调整音频长度
        if len(audio) > target_samples:
            # 截断
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            # 填充静音
            pad_length = target_samples - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        
        # 提取特征
        features = self.extract_features_from_waveform(audio, audio_sr)
        
        # 确保特征长度与视频帧数一致
        if len(features) != video_length:
            features = self._resample_features(features, video_length)
        
        return features
    
    def _resample_features(self, features, target_length):
        """重采样特征以匹配目标长度"""
        # 应用线性插值重采样
        return resample(features, target_length)
    
    def features_to_tensor(self, features):
        """将特征转换为PyTorch张量"""
        if features is None:
            return None
        return torch.FloatTensor(features)
    
    def text_to_speech(self, text, lang='en', output_path=None):
        """
        文本转语音（TTS）
        
        参数:
            text: 输入文本
            lang: 语言代码
            output_path: 输出音频路径，None则返回临时文件
            
        返回:
            音频文件路径
        """
        try:
            from gtts import gTTS
            from pydub import AudioSegment
            
            # 创建TTS对象
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # 保存到临时文件
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.close()
                output_path = temp_file.name
            
            tts.save(output_path)
            
            # 如果需要，转换为WAV格式
            if output_path.endswith('.wav'):
                wav_path = output_path
                mp3_path = output_path.replace('.wav', '.mp3')
                AudioSegment.from_mp3(mp3_path).export(wav_path, format='wav')
                os.remove(mp3_path)
                output_path = wav_path
            
            return output_path
        except Exception as e:
            print(f"TTS失败: {str(e)}")
            return None
