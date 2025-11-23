import pyaudio
import wave
import numpy as np
import os
from datetime import datetime

def record_audio(filename, duration=2, sample_rate=16000, channels=1, chunk=1024):
    """
    录制音频并保存为WAV文件
    
    参数:
    filename: 保存的文件名
    duration: 录音时长(秒)
    sample_rate: 采样率(Hz)
    channels: 声道数(1=单声道, 2=立体声)
    chunk: 每次读取的采样点数
    """
    
    # 初始化PyAudio
    p = pyaudio.PyAudio()
    
    # 打开音频流
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    print(f"开始录制 {duration} 秒音频...")
    print("录音中...")
    
    frames = []
    
    # 计算需要读取的块数
    total_chunks = int(sample_rate / chunk * duration)
    
    # 录制音频
    for i in range(total_chunks):
        data = stream.read(chunk)
        frames.append(data)
        # 显示进度
        if i % (total_chunks // 10) == 0:
            progress = (i / total_chunks) * 100
            print(f"进度: {progress:.0f}%")
    
    print("录音结束!")
    
    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 保存为WAV文件
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"音频已保存为: {filename}")
    
    # 验证文件信息
    verify_audio_file(filename, duration, sample_rate)

def verify_audio_file(filename, expected_duration, expected_sample_rate):
    """
    验证音频文件的参数是否符合预期
    """
    try:
        with wave.open(filename, 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            actual_duration = frames / sample_rate
            
            print(f"\n文件验证:")
            print(f"  采样率: {sample_rate} Hz (预期: {expected_sample_rate} Hz)")
            print(f"  时长: {actual_duration:.2f} 秒 (预期: {expected_duration} 秒)")
            print(f"  声道数: {wf.getnchannels()}")
            print(f"  采样宽度: {wf.getsampwidth()} 字节")
            
            if abs(actual_duration - expected_duration) > 0.1:
                print("警告: 实际时长与预期不符!")
            if sample_rate != expected_sample_rate:
                print("警告: 采样率与预期不符!")
                
    except Exception as e:
        print(f"验证文件时出错: {e}")

def batch_record_audio(output_dir, num_recordings=10, duration=2, sample_rate=16000):
    """
    批量录制多段音频
    
    参数:
    output_dir: 输出目录
    num_recordings: 录制数量
    duration: 每段音频时长(秒)
    sample_rate: 采样率
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")
    
    print(f"开始批量录制 {num_recordings} 段音频，每段 {duration} 秒")
    print("=" * 50)
    
    for i in range(num_recordings):
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"recording_{i+1:02d}_{timestamp}.wav")
        
        print(f"\n录制第 {i+1}/{num_recordings} 段音频...")
        record_audio(filename, duration, sample_rate)
        
        # 可选：在录制之间添加间隔
        if i < num_recordings - 1:
            input("按回车键开始录制下一段...")
    
    print(f"\n所有 {num_recordings} 段音频录制完成!")

def main():
    """
    主函数 - 提供用户交互界面
    """
    print("音频录制程序")
    print("=" * 30)
    
    while True:
        print("\n请选择录制模式:")
        print("1. 单次录制")
        print("2. 批量录制")
        print("3. 退出")
        
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == '1':
            # 单次录制
            filename = input("请输入保存的文件名 (例如: my_recording.wav): ").strip()
            if not filename.endswith('.wav'):
                filename += '.wav'
            
            duration = float(input("请输入录制时长 (秒): "))
            sample_rate = int(input("请输入采样率 (推荐 16000): "))
            
            record_audio(filename, duration, sample_rate)
            
        elif choice == '2':
            # 批量录制
            output_dir = input("请输入输出目录: ").strip()
            num_recordings = int(input("请输入录制数量: "))
            duration = float(input("请输入每段时长 (秒): "))
            sample_rate = int(input("请输入采样率 (推荐 16000): "))
            
            batch_record_audio(output_dir, num_recordings, duration, sample_rate)
            
        elif choice == '3':
            print("程序退出!")
            break
            
        else:
            print("无效选择，请重新输入!")

if __name__ == "__main__":
    # 检查依赖
    try:
        import pyaudio
    except ImportError:
        print("错误: 未找到 pyaudio 库")
        print("请安装: pip install pyaudio")
        exit(1)
    
    # 运行主程序
    main()