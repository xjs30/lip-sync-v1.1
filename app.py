import argparse
import os
import yaml
import torch
import gradio as gr
import numpy as np
import tempfile
from datetime import datetime
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import uvicorn
import cv2

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 导入内部模块
from latent_diffusion import LatentDiffusionModelV2
from audio_processor import AudioProcessorV2
from video_processor import VideoProcessorV2
from language_processor import LanguageProcessor
from trainer import TrainerV2
from utils.helpers import ensure_dir, setup_logging

# 全局配置和模型
config = None
model = None
audio_processor = None
video_processor = None
language_processor = None
device = None
logger = None

def load_resources(config_path="model_config.yaml"):
    """加载配置和模型资源"""
    global config, model, audio_processor, video_processor, language_processor, device, logger
    
    # 加载配置
    config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化日志
    log_dir = ensure_dir(os.path.join(config['training']['log_dir'], 'inference'))
    logger = setup_logging(log_dir=log_dir, log_name="inference.log")
    logger.info(f"使用设备: {device}")
    
    # 初始化处理器
    audio_processor = AudioProcessorV2(
        sample_rate=config['data']['audio_sample_rate'],
        n_mfcc=config['data']['n_mfcc']
    )
    
    video_processor = VideoProcessorV2(
        image_size=config['model']['image_size'],
        face_detector_path=config['model']['face_detector_path']
    )
    
    language_processor = LanguageProcessor()
    
    # 初始化并加载模型
    model = LatentDiffusionModelV2(config).to(device)
    
    # 加载预训练权重
    if os.path.exists(config['model']['pretrained_path']):
        try:
            checkpoint = torch.load(config['model']['pretrained_path'], map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            logger.info(f"成功加载预训练模型: {config['model']['pretrained_path']}")
        except Exception as e:
            logger.error(f"加载预训练模型失败: {str(e)}")
            raise RuntimeError(f"加载预训练模型失败: {str(e)}")
    else:
        logger.warning(f"未找到预训练模型: {config['model']['pretrained_path']}，使用随机初始化模型")

def text_to_speech(text, lang='auto'):
    """文本转语音（使用TTS引擎）"""
    try:
        from gtts import gTTS
        from io import BytesIO
        from pydub import AudioSegment
        
        if lang == 'auto':
            lang = language_processor.detect_language(text)
            logger.info(f"自动检测语言: {lang}")
        
        # 生成语音
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_io = BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        
        # 转换为wav格式
        wav_io = BytesIO()
        AudioSegment.from_mp3(audio_io).export(wav_io, format="wav")
        wav_io.seek(0)
        
        # 保存为临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(wav_io.read())
            temp_path = f.name
        
        logger.info(f"文本转语音成功，临时文件: {temp_path}")
        return temp_path
    except ImportError as e:
        logger.error(f"TTS依赖未安装: {str(e)}")
        raise gr.Error("文本转语音功能依赖未安装，请安装gTTS和pydub")
    except Exception as e:
        logger.error(f"TTS转换失败: {str(e)}")
        raise gr.Error(f"文本转语音失败: {str(e)}")

def generate_lip_sync(video_path, audio_path, text_input, lang, progress=gr.Progress()):
    """生成唇形同步视频（支持音频/文本输入）"""
    global model, audio_processor, video_processor, language_processor, device, config, logger
    
    try:
        # 验证模型是否加载
        if model is None:
            raise gr.Error("模型未加载，请检查配置和预训练文件")
        
        # 处理文本输入（如果提供）
        temp_audio = None
        if text_input and (not audio_path or audio_path is None):
            progress(0.1, desc="正在将文本转换为语音...")
            audio_path = text_to_speech(text_input, lang)
            temp_audio = audio_path  # 标记为临时文件，后续需要清理
        
        # 验证输入
        if not video_path or not audio_path:
            raise gr.Error("请同时提供视频和音频（或文本）输入")
        
        logger.info(f"开始生成唇同步视频，视频: {video_path}, 音频: {audio_path}")
        
        # 处理输入
        progress(0.2, desc="正在处理视频和音频...")
        video_frames, fps = video_processor.load_video(video_path)
        if video_frames is None or len(video_frames) == 0:
            raise gr.Error("无法加载视频文件，请检查文件格式")
        
        audio, sr = audio_processor.load_audio(audio_path)
        if audio is None:
            raise gr.Error("无法加载音频文件，请检查文件格式")
        
        # 提取音频特征
        audio_features = audio_processor.extract_features_from_waveform(audio, sr)
        
        # 提取面部特征和裁剪的面部图像
        progress(0.3, desc="正在提取面部特征...")
        face_frames, face_landmarks = video_processor.extract_face_region(video_frames)
        
        # 如果没有检测到人脸，使用原始帧
        if face_frames is None:
            logger.warning("未检测到人脸，使用原始视频帧")
            face_frames = video_frames
            face_landmarks = np.zeros((len(video_frames), 136))  # 68点×2坐标
        
        # 处理文本特征（如果存在）
        phoneme_features = None
        if text_input:
            progress(0.35, desc="正在处理文本发音...")
            phonemes = language_processor.text_to_phonemes(text_input, lang)
            phoneme_features = language_processor.phonemes_to_features(phonemes, max_length=len(audio_features))
        
        # 调整长度匹配
        min_length = min(len(face_frames), len(audio_features))
        if min_length == 0:
            raise gr.Error("视频或音频长度为零，无法处理")
        
        logger.info(f"处理序列长度: {min_length} 帧")
        
        face_frames = face_frames[:min_length]
        face_landmarks = face_landmarks[:min_length]
        audio_features = audio_features[:min_length]
        
        # 如果文本特征存在，调整长度
        if phoneme_features is not None:
            phoneme_features = phoneme_features[:min_length]
        
        # 预处理视频帧（转换为张量）
        face_tensor = video_processor.preprocess_frames(face_frames)
        
        # 转换为张量
        progress(0.4, desc="正在准备模型输入...")
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(device)
        face_tensor = face_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)
        landmarks_tensor = torch.FloatTensor(face_landmarks).unsqueeze(0).to(device)
        
        phoneme_tensor = None
        if phoneme_features is not None:
            phoneme_tensor = torch.LongTensor(phoneme_features).unsqueeze(0).to(device)
        
        # 生成唇形同步视频
        progress(0.5, desc="正在生成唇同步视频...")
        with torch.no_grad():
            generated_frames = model.sample(
                audio_tensor, 
                landmarks_tensor,
                phonemes=phoneme_tensor,
                num_steps=config['inference']['num_steps'],
                guidance_scale=config['inference']['guidance_scale']
            )
        
        # 后处理
        progress(0.8, desc="正在处理输出结果...")
        generated_frames = generated_frames.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        generated_frames = (generated_frames * 0.5 + 0.5) * 255  # 反归一化
        generated_frames = generated_frames.astype('uint8')
        
        # 保存输出视频
        output_dir = ensure_dir("outputs")
        output_path = os.path.join(
            output_dir, 
            f"lip_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        
        # 保存视频（包含原始音频）
        success = video_processor.save_video(
            generated_frames, 
            output_path, 
            fps=fps,
            audio_path=audio_path
        )
        
        if not success:
            raise gr.Error("保存输出视频失败")
        
        # 清理临时文件
        if temp_audio and os.path.exists(temp_audio):
            os.unlink(temp_audio)
        
        logger.info(f"唇同步视频生成成功: {output_path}")
        progress(1.0, desc="完成")
        return output_path
        
    except Exception as e:
        logger.error(f"生成唇同步视频失败: {str(e)}", exc_info=True)
        raise gr.Error(f"生成失败: {str(e)}")

def create_web_interface():
    """创建Gradio Web界面（支持多语言）"""
    with gr.Blocks(title="唇同步大模型V1.1", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 唇同步大模型V1.1")
        gr.Markdown("基于潜在扩散模型的高精度唇形同步系统，支持多语言输入和实时生成")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="输入视频（含人脸）", height=300)
                
                with gr.Accordion("音频输入", open=True):
                    audio_input = gr.Audio(label="输入音频", type="filepath")
                
                with gr.Accordion("文本输入（可选）", open=False):
                    text_input = gr.Textbox(
                        label="输入文本", 
                        placeholder="输入文字生成语音驱动（支持多语言）",
                        lines=3
                    )
                    lang = gr.Dropdown(
                        choices=["auto", "en", "zh", "ja", "fr", "de", "es"],
                        label="语言",
                        value="auto",
                        info="选择语言或自动检测"
                    )
                
                generate_btn = gr.Button("生成唇同步视频", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_video = gr.Video(label="输出唇同步视频", height=300)
                status_msg = gr.Textbox(label="状态信息", interactive=False)
        
        # 设置生成按钮的点击事件
        generate_btn.click(
            fn=generate_lip_sync,
            inputs=[video_input, audio_input, text_input, lang],
            outputs=output_video,
            concurrency_limit=2  # 限制并发数，避免资源耗尽
        )
        
        # 示例
        gr.Examples(
            examples=[
                ["examples/speaker.mp4", "examples/voice_en.wav", "", "auto"],
                ["examples/presenter.mp4", "", "你好，这是中文唇同步测试。", "zh"],
                ["examples/host.mp4", "", "Bonjour, ceci est un test de syncronisation labiale en français.", "fr"]
            ],
            inputs=[video_input, audio_input, text_input, lang],
            outputs=output_video,
            fn=generate_lip_sync,
            cache_examples=False
        )
        
        # 页面加载时显示状态信息
        def on_load():
            return f"模型已加载，使用设备: {device}"
        
        demo.load(on_load, None, status_msg)
    
    return demo

def create_api_service():
    """创建FastAPI服务"""
    app = FastAPI(title="唇同步大模型API V1.1", description="提供唇形同步视频生成的API服务")
    
    @app.get("/health")
    async def health_check():
        """健康检查接口"""
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(device),
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/generate", response_class=FileResponse)
    async def api_generate(
        video_file: UploadFile = File(...),
        audio_file: UploadFile = None,
        text_input: str = Form(None),
        lang: str = Form("auto")
    ):
        """生成唇同步视频API"""
        try:
            # 保存上传的视频文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf:
                content = await video_file.read()
                vf.write(content)
                video_path = vf.name
            
            # 保存上传的音频文件（如果提供）
            audio_path = None
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as af:
                    content = await audio_file.read()
                    af.write(content)
                    audio_path = af.name
            
            # 生成唇同步视频
            output_path = generate_lip_sync(video_path, audio_path, text_input, lang)
            
            # 清理临时文件
            if os.path.exists(video_path):
                os.unlink(video_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"API生成失败: {str(e)}", exc_info=True)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={"error": f"生成失败: {str(e)}"}
            )
    
    return app

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="唇同步大模型V1.1")
    parser.add_argument("--mode", type=str, default="web", choices=["web", "api", "train"], 
                      help="运行模式: web(网页界面), api(接口服务), train(模型训练)")
    parser.add_argument("--config", type=str, default="model_config.yaml", 
                      help="配置文件路径")
    parser.add_argument("--port", type=int, default=7860, 
                      help="服务端口")
    parser.add_argument("--device", type=str, default=None, 
                      help="指定运行设备，如cuda或cpu")
    
    args = parser.parse_args()
    
    # 加载资源
    load_resources(args.config)
    
    # 如果指定了设备，覆盖自动检测结果
    global device
    if args.device is not None:
        device = args.device
        logger.info(f"使用指定设备: {device}")
    
    if args.mode == "web":
        # 启动Web界面
        logger.info(f"启动Web界面，端口: {args.port}")
        demo = create_web_interface()
        demo.launch(
            server_port=args.port, 
            share=False,
            server_name="0.0.0.0"  # 允许外部访问
        )
        
    elif args.mode == "api":
        # 启动API服务
        logger.info(f"启动API服务，端口: {args.port}")
        app = create_api_service()
        uvicorn.run(
            app, 
            host="0.0.0.0",  # 允许外部访问
            port=args.port,
            workers=1  # 单工作进程，避免多进程模型加载问题
        )
        
    elif args.mode == "train":
        # 启动模型训练
        logger.info("启动模型训练")
        trainer = TrainerV2(args.config)
        trainer.train()

if __name__ == "__main__":
    main()
