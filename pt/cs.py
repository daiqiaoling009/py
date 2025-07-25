import sys
import torch
import platform
from transformers import __version__ as transformers_version

def print_experiment_environment():
    print("========================================")
    print("🚀 实验环境信息：")
    print("----------------------------------------")

    # 操作系统信息
    print(f"操作系统: {platform.system()} {platform.release()} ({platform.version()})")

    # Python 版本
    print(f"Python 版本: {sys.version.split('|')[0].strip()}")

    # PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")

    # 是否支持CUDA
    if torch.cuda.is_available():
        print(f"CUDA 可用: 是 (版本: {torch.version.cuda})")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA 可用: 否")

    # Transformers 库版本
    print(f"Transformers 版本: {transformers_version}")

    # 当前脚本所在目录（可选）
    import os
    print(f"当前工作目录: {os.getcwd()}")

    print("----------------------------------------")
    print("✅ 环境信息输出完毕")
    print("========================================")


# 调用函数输出环境信息
print_experiment_environment()