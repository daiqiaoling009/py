# # run_config_test.py
#
# import config
#
# def main():
#     args = config.parse_args()
#     print("Current Configurations:")
#     for key, value in vars(args).items():
#         print(f"{key}: {value}")
#
# if __name__ == "__main__":
#     main()

# launch_pretraining.py

# # run_config_test.py
#
# import config
#
# def main():
#     args = config.parse_args()
#     print("Current Configurations:")
#     for key, value in vars(args).items():
#         print(f"{key}: {value}")
#
# if __name__ == "__main__":
#     main()

##################################################################
#
# import os
# import sys
# import argparse
#
# # 设置 protocol buffers 实现为 python 版本
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
#
# # 添加当前目录到 PYTHONPATH，确保可以导入 config、model 等模块
# sys.path.append(os.getcwd())
#
# from run_pretraining import main
# from config import parse_args
#
#
# def set_up_directories(args):
#     """创建必要的目录"""
#     os.makedirs(args.train_tfrecord_dir, exist_ok=True)
#     os.makedirs(args.eval_tfrecord_dir, exist_ok=True)
#     os.makedirs(args.output_dir, exist_ok=True)
#     os.makedirs(args.cache_dir, exist_ok=True)
#
#
# def check_files(args):
#     """检查必要文件是否存在"""
#     # 更新此部分以确保使用正确的目录
#     train_files = [f for f in os.listdir(args.train_tfrecord_dir) if f.endswith('.tfrecord')]
#     eval_files = [f for f in os.listdir(args.eval_tfrecord_dir) if f.endswith('.tfrecord')]
#
#     if not train_files:
#         print(f"⚠️ Warning: No training .tfrecord files found in {args.train_tfrecord_dir}")
#     else:
#         print(f"✅ Found training .tfrecord files in {args.train_tfrecord_dir}")
#
#     if not eval_files:
#         print(f"⚠️ Warning: No evaluation .tfrecord files found in {args.eval_tfrecord_dir}")
#     else:
#         print(f"✅ Found evaluation .tfrecord files in {args.eval_tfrecord_dir}")
#
#     print("✅ All required checks passed (even without files).")
#
#
# def run_training():
#     args = parse_args()
#
#     # 设置本地模型路径
#     args.model_name_or_path = 'E:\\作业报告ppt\\大三下\\大数据分析和内存计算\\实验\\bert-pretraining-main\\pt\\data\\cache'
#
#     # 修改以下两个参数，确保它们指向包含.tfrecord文件的目录
#     args.train_tfrecord_dir = 'E:\\作业报告ppt\\大三下\\大数据分析和内存计算\\实验\\bert-pretraining-main\\pt\\tfrecord'
#     args.eval_tfrecord_dir = 'E:\\作业报告ppt\\大三下\\大数据分析和内存计算\\实验\\bert-pretraining-main\\pt\\tfrecord'
#
#     # 打印当前配置
#     print("\n" + "=" * 40)
#     print("🚀 Starting BERT Pretraining with the following configuration:")
#     for key, value in vars(args).items():
#         print(f"{key}: {value}")
#     print("=" * 40 + "\n")
#
#     # 设置目录
#     set_up_directories(args)
#
#     # 检查文件
#     check_files(args)
#
#     # 开始训练
#     best_perf = main(args)
#
#     # 合并最佳模型（可选）
#     if 'best_model_path' in best_perf:
#         print(f"🎉 Best model saved at: {best_perf['best_model_path']}")
#     else:
#         print("⚠️ No best model path found, skipping model merging.")
#
#
# if __name__ == "__main__":
#     run_training()



import os
import sys
import argparse

# 设置 protocol buffers 实现为 python 版本
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 添加当前目录到 PYTHONPATH，确保可以导入 config、model 等模块
sys.path.append(os.getcwd())

from run_pretraining import main
from config import parse_args


def set_up_directories(args):
    """创建必要的目录"""
    os.makedirs(args.train_tfrecord_dir, exist_ok=True)
    os.makedirs(args.eval_tfrecord_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)


def check_files(args):
    """检查必要文件是否存在"""
    # 更新此部分以确保使用正确的目录
    train_files = [f for f in os.listdir(args.train_tfrecord_dir) if f.endswith('.tfrecord')]
    eval_files = [f for f in os.listdir(args.eval_tfrecord_dir) if f.endswith('.tfrecord')]

    if not train_files:
        print(f"⚠️ Warning: No training .tfrecord files found in {args.train_tfrecord_dir}")
    else:
        print(f"✅ Found training .tfrecord files in {args.train_tfrecord_dir}")

    if not eval_files:
        print(f"⚠️ Warning: No evaluation .tfrecord files found in {args.eval_tfrecord_dir}")
    else:
        print(f"✅ Found evaluation .tfrecord files in {args.eval_tfrecord_dir}")

    print("✅ All required checks passed (even without files).")


def run_training():
    args = parse_args()

    # 设置本地模型路径
    args.model_name_or_path = 'E:\\作业报告ppt\\大三下\\大数据分析和内存计算\\实验\\bert-pretraining-main\\pt\\data\\cache'

    # 修改以下两个参数，确保它们指向包含.tfrecord文件的目录
    args.train_tfrecord_dir = 'E:\\作业报告ppt\\大三下\\大数据分析和内存计算\\实验\\bert-pretraining-main\\pt\\tfrecord'
    args.eval_tfrecord_dir = 'E:\\作业报告ppt\\大三下\\大数据分析和内存计算\\实验\\bert-pretraining-main\\pt\\tfrecord'

    # 打印当前配置
    print("\n" + "=" * 40)
    print("🚀 Starting BERT Pretraining with the following configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 40 + "\n")

    # 设置目录
    set_up_directories(args)

    # 检查文件
    check_files(args)

    # 开始训练
    best_perf = main(args)

    # 合并最佳模型（可选）
    if 'best_model_path' in best_perf:
        print(f"🎉 Best model saved at: {best_perf['best_model_path']}")
    else:
        print("⚠️ No best model path found, skipping model merging.")


if __name__ == "__main__":
    run_training()