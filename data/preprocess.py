import os
import argparse
import numpy as np
import yaml
import shutil
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from audio_processor import AudioProcessorV2
from video_processor import VideoProcessorV2
from utils.helpers import ensure_dir, setup_logging

def process_single_video(video_path, args, config, logger):
    """处理单个视频文件及其对应的音频和文本"""
    try:
        # 解析文件名和路径
        video_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 确定分割类型（train/val/test）
        split = os.path.basename(os.path.dirname(video_dir))
        
        # 对应的音频文件路径
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_path = None
        for ext in audio_extensions:
            candidate = os.path.join(args.raw_data_dir, split, 'audios', f"{base_name}{ext}")
            if os.path.exists(candidate):
                audio_path = candidate
                break
        
        if not audio_path:
            logger.warning(f"未找到视频 {base_name} 对应的音频文件，跳过")
            return False
        
        # 对应的文本文件路径
        text_path = os.path.join(args.raw_data_dir, split, 'texts', f"{base_name}.txt")
        has_text = os.path.exists(text_path)
        
        # 创建输出目录
        out_video_dir = os.path.join(args.output_dir, split, 'videos')
        out_audio_dir = os.path.join(args.output_dir, split, 'audios')
        out_text_dir = os.path.join(args.output_dir, split, 'texts')
        
        ensure_dir(out_video_dir)
        ensure_dir(out_audio_dir)
        ensure_dir(out_text_dir)
        
        # 输出文件路径
        out_video_path = os.path.join(out_video_dir, f"{base_name}.mp4")
        out_audio_path = os.path.join(out_audio_dir, f"{base_name}.wav")
        out_text_path = os.path.join(out_text_dir, f"{base_name}.txt")
        
        # 如果已处理，跳过
        if os.path.exists(out_video_path) and os.path.exists(out_audio_path) and \
           (not has_text or os.path.exists(out_text_path)):
            logger.debug(f"文件 {base_name} 已处理，跳过")
            return True
        
        # 初始化处理器
        video_processor = VideoProcessorV2(
            image_size=config['model']['image_size'],
            face_detector_path=config.get('model', {}).get('face_detector_path')
        )
        
        audio_processor = AudioProcessorV2(
            sample_rate=config['data']['audio_sample_rate'],
            n_mfcc=config['data']['n_mfcc']
        )
        
        # 1. 处理视频
        logger.debug(f"处理视频 {base_name}")
        frames, fps = video_processor.load_video(video_path)
        
        if frames is None or len(frames) == 0:
            logger.warning(f"无法加载视频 {base_name}，跳过")
            return False
        
        # 提取人脸区域
        face_frames, landmarks = video_processor.extract_face_region(frames)
        
        if face_frames is None or len(face_frames) == 0:
            logger.warning(f"视频 {base_name} 中未检测到人脸，跳过")
            return False
        
        # 保存处理后的视频
        success = video_processor.save_video(face_frames, out_video_path, fps)
        if not success:
            logger.warning(f"保存视频 {base_name} 失败，跳过")
            return False
        
        # 保存特征点
        np.save(os.path.join(out_video_dir, f"{base_name}_landmarks.npy"), landmarks)
        
        # 2. 处理音频
        logger.debug(f"处理音频 {base_name}")
        audio, sr = audio_processor.load_audio(audio_path)
        
        if audio is None:
            logger.warning(f"无法加载音频 {base_name}，跳过")
            return False
        
        # 对齐音频与视频长度
        audio_features = audio_processor.align_audio_with_video(
            audio, sr, fps, len(face_frames)
        )
        
        # 保存处理后的音频
        success = audio_processor.save_audio(audio, out_audio_path, sr)
        if not success:
            logger.warning(f"保存音频 {base_name} 失败，跳过")
            return False
        
        # 保存音频特征
        np.save(os.path.join(out_audio_dir, f"{base_name}_features.npy"), audio_features)
        
        # 3. 处理文本（如果存在）
        if has_text:
            logger.debug(f"处理文本 {base_name}")
            try:
                # 复制文本文件
                shutil.copy2(text_path, out_text_path)
            except Exception as e:
                logger.warning(f"处理文本 {base_name} 失败: {str(e)}")
        
        logger.debug(f"成功处理 {base_name}")
        return True
        
    except Exception as e:
        logger.error(f"处理 {video_path} 时出错: {str(e)}")
        return False

def main():
    """主函数：预处理唇同步数据集"""
    parser = argparse.ArgumentParser(description="唇同步数据集预处理")
    parser.add_argument("--raw_data_dir", type=str, required=True, 
                      help="原始数据目录，结构应为 raw_data_dir/{train,val,test}/{videos,audios,texts}")
    parser.add_argument("--output_dir", type=str, default="processed_dataset", 
                      help="预处理后的数据输出目录")
    parser.add_argument("--config", type=str, default="model_config.yaml", 
                      help="模型配置文件路径")
    parser.add_argument("--split", type=str, default="all", 
                      choices=["all", "train", "val", "test"], 
                      help="要处理的数据集分割")
    parser.add_argument("--num_workers", type=int, default=4, 
                      help="并行处理的进程数")
    parser.add_argument("--overwrite", action="store_true", 
                      help="是否覆盖已处理的数据")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化日志
    logger = setup_logging(log_dir=os.path.join(args.output_dir, "logs"), log_name="preprocess.log")
    logger.info(f"开始预处理数据集，原始数据目录: {args.raw_data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 如果需要覆盖，清空输出目录
    if args.overwrite and os.path.exists(args.output_dir):
        logger.warning(f"将覆盖现有输出目录: {args.output_dir}")
        shutil.rmtree(args.output_dir, ignore_errors=True)
    
    # 确定要处理的分割
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    # 收集所有视频文件路径
    video_paths = []
    for split in splits:
        video_dir = os.path.join(args.raw_data_dir, split, "videos")
        if not os.path.exists(video_dir):
            logger.warning(f"视频目录 {video_dir} 不存在，跳过该分割")
            continue
        
        # 获取所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        for file in os.listdir(video_dir):
            if any(file.endswith(ext) for ext in video_extensions):
                video_paths.append(os.path.join(video_dir, file))
    
    if not video_paths:
        logger.error("未找到任何视频文件，退出")
        return
    
    logger.info(f"共找到 {len(video_paths)} 个视频文件需要处理")
    
    # 并行处理视频
    process_func = partial(process_single_video, 
                          args=args, 
                          config=config, 
                          logger=logger)
    
    # 使用进程池并行处理
    with mp.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, video_paths),
            total=len(video_paths),
            desc="预处理进度"
        ))
    
    # 统计结果
    success_count = sum(1 for res in results if res)
    logger.info(f"预处理完成，成功处理 {success_count}/{len(video_paths)} 个文件")
    
    # 生成数据集元数据
    metadata = {
        "total_files": len(video_paths),
        "success_files": success_count,
        "splits": {}
    }
    
    for split in splits:
        video_dir = os.path.join(args.output_dir, split, "videos")
        if os.path.exists(video_dir):
            video_count = len([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
            metadata["splits"][split] = video_count
    
    # 保存元数据
    metadata_path = os.path.join(args.output_dir, "dataset_metadata.yaml")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"数据集元数据已保存到 {metadata_path}")
    logger.info("预处理完成")

if __name__ == "__main__":
    main()
