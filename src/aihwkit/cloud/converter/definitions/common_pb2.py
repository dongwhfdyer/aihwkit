# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: common.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from . import onnx_common_pb2 as onnx__common__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0c\x63ommon.proto\x12\x05\x61ihwx\x1a\x11onnx_common.proto\"F\n\x0ePrimitiveProto\x12\n\n\x02id\x18\x01 \x02(\t\x12(\n\targuments\x18\x02 \x03(\x0b\x32\x15.aihwx.AttributeProto\"F\n\x0eTransformProto\x12\n\n\x02id\x18\x01 \x02(\t\x12(\n\targuments\x18\x02 \x03(\x0b\x32\x15.aihwx.AttributeProto\"O\n\x17\x41\x63tivationFunctionProto\x12\n\n\x02id\x18\x01 \x02(\t\x12(\n\targuments\x18\x02 \x03(\x0b\x32\x15.aihwx.AttributeProto\"m\n\nLayerProto\x12\n\n\x02id\x18\x01 \x02(\t\x12(\n\targuments\x18\x02 \x03(\x0b\x32\x15.aihwx.AttributeProto\x12)\n\nstate_dict\x18\x03 \x03(\x0b\x32\x15.aihwx.AttributeProto\"\x86\x01\n\x19LayerOrActivationFunction\x12=\n\x13\x61\x63tivation_function\x18\x01 \x01(\x0b\x32\x1e.aihwx.ActivationFunctionProtoH\x00\x12\"\n\x05layer\x18\x02 \x01(\x0b\x32\x11.aihwx.LayerProtoH\x00\x42\x06\n\x04item\"F\n\x0eOptimizerProto\x12\n\n\x02id\x18\x01 \x02(\t\x12(\n\targuments\x18\x02 \x03(\x0b\x32\x15.aihwx.AttributeProto\"F\n\x0eSchedulerProto\x12\n\n\x02id\x18\x01 \x02(\t\x12(\n\targuments\x18\x02 \x03(\x0b\x32\x15.aihwx.AttributeProto\"I\n\x11LossFunctionProto\x12\n\n\x02id\x18\x01 \x02(\t\x12(\n\targuments\x18\x02 \x03(\x0b\x32\x15.aihwx.AttributeProto\";\n\x07Network\x12\x30\n\x06layers\x18\x01 \x03(\x0b\x32 .aihwx.LayerOrActivationFunction\"(\n\x07Version\x12\x0e\n\x06schema\x18\x01 \x02(\x03\x12\r\n\x05opset\x18\x02 \x02(\x03')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'common_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _PRIMITIVEPROTO._serialized_start = 42
    _PRIMITIVEPROTO._serialized_end = 112
    _TRANSFORMPROTO._serialized_start = 114
    _TRANSFORMPROTO._serialized_end = 184
    _ACTIVATIONFUNCTIONPROTO._serialized_start = 186
    _ACTIVATIONFUNCTIONPROTO._serialized_end = 265
    _LAYERPROTO._serialized_start = 267
    _LAYERPROTO._serialized_end = 376
    _LAYERORACTIVATIONFUNCTION._serialized_start = 379
    _LAYERORACTIVATIONFUNCTION._serialized_end = 513
    _OPTIMIZERPROTO._serialized_start = 515
    _OPTIMIZERPROTO._serialized_end = 585
    _SCHEDULERPROTO._serialized_start = 587
    _SCHEDULERPROTO._serialized_end = 657
    _LOSSFUNCTIONPROTO._serialized_start = 659
    _LOSSFUNCTIONPROTO._serialized_end = 732
    _NETWORK._serialized_start = 734
    _NETWORK._serialized_end = 793
    _VERSION._serialized_start = 795
    _VERSION._serialized_end = 835
# @@protoc_insertion_point(module_scope)
