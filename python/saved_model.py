# -*- coding: utf-8 -*-

import os

import time
import varint

from proto.data_pb2 import Data, Tensor, DataType
from proto.index_pb2 import Index, Field


def float_tensor(shape, float_val):
    return Tensor(dtype=DataType.Value("DT_FLOAT"),
                  tensor_shape=shape,
                  float_val=float_val)


class SavedModel(object):
    def __init__(self):
        self.data = Data()
        self.index = Index()

    def add_named_tensor(self, key, tensor):
        self.data.data[key].CopyFrom(tensor)

    def add_field_index(self, field):
        self.index.embeddings[field.name].CopyFrom(field)

    @staticmethod
    def header(model_name, version):
        model_name = model_name.encode("utf-8")
        version = varint.encode(version).ljust(10, b'\x00')
        return model_name + version

    @staticmethod
    def footer(header_size, data_size):
        magic = b'go_serving'
        data_offset = header_size
        index_offset = data_offset + data_size
        return varint.encode(data_offset).ljust(10, b'\x00') + \
               varint.encode(index_offset).ljust(10, b'\x00') + magic

    def export(self, model_name, version):
        assert (isinstance(version, int))
        file = "%d.pb" % version
        try:
            with open(file, "wb") as f:
                # header
                header_size = f.write(self.header(model_name, version))
                # data
                data_size = f.write(self.data.SerializeToString())
                # index
                index_size = f.write(self.index.SerializeToString())
                # footer
                footer_size = f.write(self.footer(header_size, data_size))
                assert (footer_size == 30)
            print("[%s] save %s ok (%d, %d, %d, %d)" % (
                model_name, file, header_size, data_size, index_size, footer_size))
        except Exception as e:
            print("[%s] save %s fails: %s" % (model_name, file, e))
            os.remove(f.name)

        def load(self, file):
            # todo:
            pass


# m = SavedModel()
# m.add_named_tensor("F1", float_tensor([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]))
# m.add_named_tensor("F2", float_tensor([2, 4], [3, 4, 5, 6, 0, 2, 4, 1]))
# m.add_field_index(Field(name="F1", dim=3, records={"125": 0, "124": 1, "123": 2}))
# m.add_field_index(Field(name="F2", dim=4, records={"0": 0, "1": 1}))
# m.export(model_name="wide_deep", version=int(time.time()))
