# 逐个文件修改
# from tokenization import FullTokenizer
# import csv
# import tensorflow as tf
#
# # 加载分词器
# vocab_file = "vocab.txt"
# tokenizer = FullTokenizer(vocab_file=vocab_file)
#
# # 定义最大序列长度
# max_seq_length = 128
#
#
# def convert_single_example(sentence_a, sentence_b, label):
#     tokens_a = tokenizer.tokenize(sentence_a)
#     tokens_b = tokenizer.tokenize(sentence_b)
#
#     # 构建 [CLS] + A + [SEP] + B + [SEP]
#     tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#     # 构建 attention mask
#     input_mask = [1] * len(input_ids)
#
#     # 构建 segment_ids: 0 for sentence A, 1 for sentence B
#     segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
#
#     # Zero-pad up to max_seq_length
#     while len(input_ids) < max_seq_length:
#         input_ids.append(0)
#         input_mask.append(0)
#         segment_ids.append(0)
#
#     return input_ids, input_mask, segment_ids, label
#
#
# def create_int_feature(values):
#     """创建 int64 类型的 Feature"""
#     if not isinstance(values, list):
#         values = list(values)
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
#
#
# def write_to_tfrecord(examples, output_file):
#     """将 examples 写入 TFRecord 文件"""
#     with tf.io.TFRecordWriter(output_file) as writer:
#         for (input_ids, input_mask, segment_ids, label) in examples:
#             features = {
#                 "input_ids": create_int_feature(input_ids),
#                 "attention_mask": create_int_feature(input_mask),
#                 "token_type_ids": create_int_feature(segment_ids),
#                 "label": create_int_feature([int(label)]),  # 包装成列表以支持 scalar
#             }
#
#             tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#             writer.write(tf_example.SerializeToString())
#
#     print(f"✅ 成功写入 {output_file}")
#
#
# def process_tsv(input_file, output_file):
#     examples = []
#
#     with open(input_file, mode='r', encoding='utf-8') as infile:
#         reader = csv.reader(infile, delimiter='\t')
#         header = next(reader)  # Skip header
#
#         for row in reader:
#             if len(row) < 4:
#                 print("⚠️ 跳过无效行:", row)
#                 continue
#
#             index, sentence_a, sentence_b, label = row
#
#             try:
#                 label = int(label)
#             except ValueError:
#                 print(f"❌ 标签无法转换为整数: {label}")
#                 continue
#
#             input_ids, input_mask, segment_ids, label = convert_single_example(sentence_a, sentence_b, label)
#             examples.append((input_ids, input_mask, segment_ids, label))
#
#     write_to_tfrecord(examples, output_file)
#
#
# if __name__ == "__main__":
#     input_tsv = "eval00.tsv"
#     output_tfrecord = "eval00.tfrecord"
#     process_tsv(input_tsv, output_tfrecord)

from tokenization import FullTokenizer
import csv
import tensorflow as tf
import os

# 加载分词器
vocab_file = "vocab.txt"
tokenizer = FullTokenizer(vocab_file=vocab_file)

# 定义最大序列长度
max_seq_length = 128


def convert_single_example(sentence_a, sentence_b, label=None):
    tokens_a = tokenizer.tokenize(sentence_a)
    tokens_b = tokenizer.tokenize(sentence_b)

    # 构建 [CLS] + A + [SEP] + B + [SEP]
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 构建 attention mask
    input_mask = [1] * len(input_ids)

    # 构建 segment_ids: 0 for sentence A, 1 for sentence B
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

    # Zero-pad up to max_seq_length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    if label is None:
        label = 0  # 默认值，用于 test 文件

    return input_ids, input_mask, segment_ids, label


def create_int_feature(values):
    """创建 int64 类型的 Feature"""
    if not isinstance(values, list):
        values = list(values)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def write_to_tfrecord(examples, output_file):
    """将 examples 写入 TFRecord 文件"""
    with tf.io.TFRecordWriter(output_file) as writer:
        for (input_ids, input_mask, segment_ids, label) in examples:
            features = {
                "input_ids": create_int_feature(input_ids),
                "attention_mask": create_int_feature(input_mask),
                "token_type_ids": create_int_feature(segment_ids),
                "label": create_int_feature([int(label)]),  # 包装成列表以支持 scalar
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

    print(f"✅ 成功写入 {output_file}")


def process_tsv(input_file, output_file, has_label=True):
    examples = []

    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t')
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) < 3:
                print("⚠️ 跳过无效行:", row)
                continue

            index = row[0]
            sentence_a = row[1]
            sentence_b = row[2]

            label = None
            if has_label:
                try:
                    label = int(row[3])
                except (ValueError, IndexError):
                    print(f"❌ 标签解析失败: {row}")
                    continue

            input_ids, input_mask, segment_ids, label = convert_single_example(
                sentence_a, sentence_b, label
            )
            examples.append((input_ids, input_mask, segment_ids, label))

    write_to_tfrecord(examples, output_file)


if __name__ == "__main__":
    data_dir = r'E:\作业报告ppt\大三下\大数据分析和内存计算\实验\glue\1data'

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".tsv"):
                input_path = os.path.join(root, file)
                output_name = file.replace(".tsv", ".tfrecord")
                output_path = os.path.join(root, output_name)

                # 判断是否是 test 文件（没有 label）
                is_test = "test" in file.lower()
                print(f"Processing {'(test)' if is_test else '(train/dev)'}: {file}")
                process_tsv(input_path, output_path, has_label=not is_test)