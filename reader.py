"""Reader utils."""

import functools
import gzip
import io
import os
import struct
import typing
import math

import numpy as np

from tfrecord import example_pb2
from tfrecord import iterator_utils


def tfrecord_iterator(
    data_path: str,
    index_path: typing.Optional[str] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None,
    compression_type: typing.Optional[str] = None,
    batch_size: typing.Union[int, None] = None
) -> typing.Iterable[memoryview]:
    if compression_type == "gzip":
        file = gzip.open(data_path, "rb")
    elif compression_type is None:
        file = io.open(data_path, "rb")
    else:
        raise ValueError("compression_type should be either 'gzip' or None")
    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)

    def read_records(start_offset=None, end_offset=None):
        nonlocal length_bytes, crc_bytes, datum_bytes

        if start_offset is not None:
            file.seek(start_offset)
        if end_offset is None:
            end_offset = os.path.getsize(data_path)
        while file.tell() < end_offset:
            if file.readinto(length_bytes) != 8:
                raise RuntimeError("Failed to read the record size.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the start token.")
            length, = struct.unpack("<Q", length_bytes)
            if length > len(datum_bytes):
                datum_bytes = datum_bytes.zfill(int(length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:length]
            if file.readinto(datum_bytes_view) != length:
                raise RuntimeError("Failed to read the record.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the end token.")
            yield datum_bytes_view

    if index_path is None:
        yield from read_records()
    else:
        index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
        if shard is None:
            offset = np.random.choice(index)
            yield from read_records(offset)
            yield from read_records(0, offset)
        else:
            num_records = len(index)
            shard_idx, shard_count = shard
            if batch_size is None:
                start_index = (num_records * shard_idx) // shard_count
                end_index = (num_records * (shard_idx + 1)) // shard_count
            else:
                num_shards = math.ceil(num_records / batch_size)
                start_index = (num_shards * shard_idx) // shard_count * batch_size
                end_index = (num_shards * (shard_idx + 1)) // shard_count * batch_size
            start_byte = index[start_index]
            end_byte = index[end_index] if end_index < num_records else None
            yield from read_records(start_byte, end_byte)

    file.close()


def process_feature(feature: example_pb2.Feature,
                    typename: str,
                    typename_mapping: dict,
                    key: str):
    # NOTE: We assume that each key in the example has only one field
    # (either "bytes_list", "float_list", or "int64_list")!
    field = feature.ListFields()[0]
    inferred_typename, value = field[0].name, field[1].value

    if typename is not None:
        tf_typename = typename_mapping[typename]
        if tf_typename != inferred_typename:
            reversed_mapping = {v: k for k, v in typename_mapping.items()}
            raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                        f"(should be '{reversed_mapping[inferred_typename]}').")

    if inferred_typename == "bytes_list":
        value = np.frombuffer(value[0], dtype=np.uint8)
    elif inferred_typename == "float_list":
        value = np.array(value, dtype=np.float32)
    elif inferred_typename == "int64_list":
        value = np.array(value, dtype=np.int64)
    return value


# def extract_feature_dict(features, description, typename_mapping):
#
#     if isinstance(features, example_pb2.FeatureLists):
#         features = features.feature_list
#
#         def get_value(typename, typename_mapping, key):
#             feature = features[key].feature
#             fn = functools.partial(process_feature, typename=typename,
#                                    typename_mapping=typename_mapping, key=key)
#             return list(map(fn, feature))
#     elif isinstance(features, example_pb2.Features):
#         features = features.feature
#
#         def get_value(typename, typename_mapping, key):
#             return process_feature(features[key], typename,
#                                    typename_mapping, key)
#     else:
#         raise TypeError(f"Incompatible type: features should be either of type "
#                         f"example_pb2.Features or example_pb2.FeatureLists and "
#                         f"not {type(features)}")
#
#     all_keys = list(features.keys())
#
#     if description is None or len(description) == 0:
#         description = dict.fromkeys(all_keys, None)
#     elif isinstance(description, list):
#         description = dict.fromkeys(description, None)
#
#     processed_features = {}
#     for key, typename in description.items():
#         if key not in all_keys:
#             raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")
#
#         processed_features[key] = get_value(typename, typename_mapping, key)
#     if 'input_mask' in processed_features:
#         processed_features['attention_mask'] = processed_features['input_mask']
#
#     return processed_features
def extract_feature_dict(features, description, typename_mapping):
    if isinstance(features, example_pb2.FeatureLists):
        features = features.feature_list

        def get_value(typename, key):
            feature_list = features[key].feature
            return [process_feature(f, typename, typename_mapping, key) for f in feature_list]

    elif isinstance(features, example_pb2.Features):
        features = features.feature

        def get_value(typename, key):
            return process_feature(features[key], typename, typename_mapping, key)

    else:
        raise TypeError(f"Unsupported type: {type(features)}")

    all_keys = list(features.keys())

    if description is None or len(description) == 0:
        description = dict.fromkeys(all_keys, None)
    elif isinstance(description, list):
        description = dict.fromkeys(description, None)

    processed_features = {}
    for key, typename in description.items():
        if key not in all_keys:
            raise KeyError(f"Key '{key}' does not exist (available keys: {all_keys})")
        processed_features[key] = get_value(typename, key)

    # 兼容 BERT 输入格式
    if 'input_mask' in processed_features:
        processed_features['attention_mask'] = processed_features.pop('input_mask')

    return processed_features


def example_loader(
    data_path: str,
    index_path: typing.Union[str, None],
    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None,
    compression_type: typing.Optional[str] = None,
    batch_size: typing.Union[int, None] = None
) -> typing.Iterable[typing.Dict[str, np.ndarray]]:

    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    record_iterator = tfrecord_iterator(
        data_path=data_path,
        index_path=index_path,
        shard=shard,
        compression_type=compression_type,
        batch_size=batch_size
    )

    for record in record_iterator:
        example = example_pb2.Example()
        example.ParseFromString(record)

        yield extract_feature_dict(example.features, description, typename_mapping)


def sequence_loader(
    data_path: str,
    index_path: typing.Union[str, None],
    context_description: typing.Union[
        typing.List[str], typing.Dict[str, str], None
    ] = None,
    features_description: typing.Union[
        typing.List[str], typing.Dict[str, str], None
    ] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None,
    compression_type: typing.Optional[str] = None,
    batch_size: typing.Union[int, None] = None
) -> typing.Iterable[
    typing.Tuple[
        typing.Dict[str, np.ndarray], typing.Dict[str, typing.List[np.ndarray]]
    ]
]:
    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    record_iterator = tfrecord_iterator(
        data_path=data_path,
        index_path=index_path,
        shard=shard,
        compression_type=compression_type,
        batch_size=batch_size
    )

    for record in record_iterator:
        example = example_pb2.SequenceExample()
        example.ParseFromString(record)

        context = extract_feature_dict(example.context, context_description, typename_mapping)
        features = extract_feature_dict(example.feature_lists, features_description, typename_mapping)

        yield context, features


def tfrecord_loader(
    data_path: str,
    index_path: typing.Union[str, None],
    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None,
    sequence_description: typing.Union[
        typing.List[str], typing.Dict[str, str], None
    ] = None,
    compression_type: typing.Optional[str] = None,
    batch_size: typing.Union[int, None] = None
) -> typing.Iterable[
    typing.Union[
        typing.Dict[str, np.ndarray],
        typing.Tuple[
            typing.Dict[str, np.ndarray], typing.Dict[str, typing.List[np.ndarray]]
        ],
    ]
]:

    if sequence_description is not None:
        return sequence_loader(
            data_path=data_path,
            index_path=index_path,
            context_description=description,
            features_description=sequence_description,
            shard=shard,
            compression_type=compression_type,
            batch_size=batch_size
        )
    return example_loader(
        data_path=data_path,
        index_path=index_path,
        description=description,
        shard=shard,
        compression_type=compression_type,
        batch_size=batch_size
    )


def multi_tfrecord_loader(data_pattern: str,
                          index_pattern: typing.Union[str, None],
                          splits: typing.Dict[str, float],
                          description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                          shard: typing.Optional[typing.Tuple[int, int]] = None,
                          sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                          compression_type: typing.Optional[str] = None,
                          infinite: bool = True,
                          batch_size: typing.Union[int, None] = None
                          ) -> typing.Iterable[typing.Union[typing.Dict[str, np.ndarray],
                                                            typing.Tuple[typing.Dict[str, np.ndarray],
                                                                         typing.Dict[str, typing.List[np.ndarray]]]]]:
    loaders = [functools.partial(tfrecord_loader, data_path=data_pattern.format(split),
                                 index_path=index_pattern.format(split) \
                                     if index_pattern is not None else None,
                                 description=description,
                                 shard=shard,
                                 sequence_description=sequence_description,
                                 compression_type=compression_type,
                                 batch_size=batch_size
                                 )
               for split in splits.keys()]
    return iterator_utils.sample_iterators(loaders, list(splits.values()), infinite=infinite)
