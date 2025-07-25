import os
import subprocess
import argparse

def generate_index_for_tfrecord(tfrecord_path, output_index_path, tools_script_path):
    """
    为单个 .tfrecord 文件生成 .index 文件
    """
    if not os.path.exists(tfrecord_path):
        print(f"Error: TFRecord file {tfrecord_path} does not exist.")
        return

    if not os.path.exists(tools_script_path):
        print(f"Error: Script {tools_script_path} does not exist.")
        return

    cmd = ["python", tools_script_path, tfrecord_path, output_index_path]
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully generated index for {tfrecord_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate index for {tfrecord_path}: {e}")

def generate_indices_for_directory(tfrecord_dir, tools_script_path):
    """
    为整个目录中的所有 .tfrecord 文件生成 .index 文件
    """
    print(f"Processing directory: {tfrecord_dir}")  # 打印处理的目录
    if not os.path.exists(tools_script_path):
        print(f"Error: Script {tools_script_path} does not exist.")
        return

    for root, _, files in os.walk(tfrecord_dir):
        print(f"Scanning directory: {root}")  # 打印扫描的子目录
        for file in files:
            if file.endswith(".tfrecord"):
                tfrecord_path = os.path.join(root, file)
                index_path = tfrecord_path.replace(".tfrecord", ".index")
                print(f"Found TFRecord file: {tfrecord_path}")  # 打印找到的文件
                if os.path.exists(index_path):
                    print(f"Skipping {tfrecord_path}, index already exists.")
                    continue
                generate_index_for_tfrecord(tfrecord_path, index_path, tools_script_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .index files for all .tfrecord files.")

    parser.add_argument(
        "--tfrecord_file",
        type=str,
        default=None,
        help="Path to a single .tfrecord file to index. If not set, indexes entire directory.",
    )
    parser.add_argument(
        "--tfrecord_dir",
        type=str,
        default=r"E:\作业报告ppt\大三下\大数据分析和内存计算\实验\bert-pretraining-main\pt\tfrecord",
        help="Directory containing .tfrecord files (used if --tfrecord_file is not provided).",
    )
    parser.add_argument(
        "--tools_script_path",
        type=str,
        default=r'E:\作业报告ppt\大三下\大数据分析和内存计算\实验\bert-pretraining-main\pt\tfrecord\tools\tfrecord2idx.py',
        help="Path to the tfrecord2idx.py script."
    )

    args = parser.parse_args()

    if args.tfrecord_file:
        index_path = args.tfrecord_file.replace(".tfrecord", ".index")
        generate_index_for_tfrecord(args.tfrecord_file, index_path, args.tools_script_path)
    else:
        generate_indices_for_directory(args.tfrecord_dir, args.tools_script_path)

