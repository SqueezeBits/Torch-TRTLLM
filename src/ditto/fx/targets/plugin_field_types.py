import numpy as np
import tensorrt as trt

PLUGIN_FIELD_TYPES = {
    np.int8: trt.PluginFieldType.INT8,
    np.int16: trt.PluginFieldType.INT16,
    np.int32: trt.PluginFieldType.INT32,
    np.float16: trt.PluginFieldType.FLOAT16,
    np.float32: trt.PluginFieldType.FLOAT32,
    np.float64: trt.PluginFieldType.FLOAT64,
}
