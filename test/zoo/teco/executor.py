import numpy as np
import ctypes
import torch
import paddle
from prototxt_parser.prototxt_parser_main import parse
import os


def read_prototxt(file) -> dict:
    filtered_lines = []
    with open(file, "r") as f:
        for line in f:
            index = line.find("#")
            if index != -1:
                line = line[0:index] + "\n"
            filtered_lines.append(line)
    content = "".join(filtered_lines)
    return parse(content)


def check_framework(framework):
    if framework not in ["torch", "paddle"]:
        print("not supported framework {}".format(framework))
        return False
    return True


def save_tensor(fp, data, dtype=None, framework="torch"):
    if not check_framework(framework):
        return None
    if dtype:
        if framework == "torch":
            data = data.type(to_torch_dtype(dtype))
            if dtype == "DTYPE_BFLOAT16":
                data = data.type(torch.float32)
        elif framework == "paddle":
            data = paddle.cast(data, dtype=to_paddle_dtype(dtype))
            if dtype == "DTYPE_BFLOAT16":
                data = paddle.cast(data, dtype=paddle.float32)
    data.detach().cpu().numpy().tofile(fp)

def save_tensor_bf16(fp, tensor, framework="torch"):
    if framework=="torch":
        tensor_cpu = tensor.cpu().contiguous()
        data_ptr = tensor_cpu.data_ptr()
        num_bytes = tensor_cpu.numel() * 2
        buffer = (ctypes.c_byte * num_bytes).from_address(data_ptr)
        np_array = np.frombuffer(buffer, dtype=np.uint16).copy()
        np_array.tofile(fp)
    if framework=="paddle":
        tensor_cpu = tensor.cpu()
        tensor_cpu = tensor_cpu.clone()
        tensor_desc = tensor_cpu.value().get_tensor()
        data_ptr = tensor_desc._ptr()
        num_bytes = tensor_desc._numel()  * 2
    buffer = (ctypes.c_byte * num_bytes).from_address(data_ptr)
    np_array = np.frombuffer(buffer, dtype=np.uint16).copy()
    np_array.tofile(fp)

def load_bfloat16_direct(filename, params, framework="torch"):
    np_shape = get_shape(params["layout"], params["shape"]["dims"])
    with open(filename, "rb") as f:
        raw_data = f.read()
    # 创建NumPy数组作为中介（避免警告）
    np_arr = np.frombuffer(raw_data, dtype=np.int16)
    # 转换为PyTorch张量
    tensor = torch.tensor(np_arr, dtype=torch.int16).view(torch.bfloat16)
    tensor = tensor.to(torch.float64)
    if framework == "paddle":
        np_array_mid = tensor.detach().cpu().numpy()
        tensor = paddle.to_tensor(np_array_mid)
        tensor = paddle.cast(tensor, 'float64')
    if np_shape is not None:
        tensor = tensor.reshape(np_shape)
    return tensor

def to_np_dtype(dtype):
    if dtype == "DTYPE_HALF":
        return np.float16
    elif dtype == "DTYPE_FLOAT":
        return np.float32
    elif dtype == "DTYPE_DOUBLE":
        return np.float64
    elif dtype == "DTYPE_INT8":
        return np.int8
    elif dtype == "DTYPE_INT16":
        return np.int16
    elif dtype == "DTYPE_INT32":
        return np.int32
    elif dtype == "DTYPE_INT64":
        return np.int64
    elif dtype == "DTYPE_UINT8":
        return np.uint8
    elif dtype == "DTYPE_UINT16":
        return np.uint16
    elif dtype == "DTYPE_UINT32":
        return np.uint32
    elif dtype == "DTYPE_UINT64":
        return np.uint64
    elif dtype == "DTYPE_BOOL":
        return np.bool_
    elif dtype == "DTYPE_BFLOAT16":
        return np.float32
    elif dtype == "DTYPE_COMPLEX_FLOAT":
        return np.complex64
    else:
        print("not supported dtype {}".format(dtype))
        return None


def to_torch_dtype(dtype):
    if dtype == "DTYPE_HALF":
        return torch.float16
    elif dtype == "DTYPE_FLOAT":
        return torch.float32
    elif dtype == "DTYPE_DOUBLE":
        return torch.float64
    elif dtype == "DTYPE_INT8":
        return torch.int8
    elif dtype == "DTYPE_INT16":
        return torch.int16
    elif dtype == "DTYPE_INT32":
        return torch.int32
    elif dtype == "DTYPE_INT64":
        return torch.int64
    elif dtype == "DTYPE_UINT8":
        return torch.uint8
    elif dtype == "DTYPE_BOOL":
        return torch.bool
    elif dtype == "DTYPE_BFLOAT16":
        return torch.bfloat16
    elif dtype == "DTYPE_COMPLEX_FLOAT":
        return torch.complex64
    else:
        print("not supported dtype {}".format(dtype))
        return None


def to_paddle_dtype(dtype):
    if dtype == "DTYPE_HALF":
        return paddle.float16
    elif dtype == "DTYPE_FLOAT":
        return paddle.float32
    elif dtype == "DTYPE_DOUBLE":
        return paddle.float64
    elif dtype == "DTYPE_INT8":
        return paddle.int8
    elif dtype == "DTYPE_INT16":
        return paddle.int16
    elif dtype == "DTYPE_INT32":
        return paddle.int32
    elif dtype == "DTYPE_INT64":
        return paddle.int64
    elif dtype == "DTYPE_UINT8":
        return paddle.uint8
    elif dtype == "DTYPE_BOOL":
        return paddle.bool
    elif dtype == "DTYPE_BFLOAT16":
        return paddle.bfloat16
    else:
        print("not supported dtype {}".format(dtype))
        return None


def get_shape(layout, dims):
    if type(dims) != list:
        dims = [int(dims)]
    else:
        dims = [int(i) for i in dims]
    np_shape = tuple(dims)
    return np_shape


def get_stride(strides, type_size):
    if type(strides) != list:
        strides = [strides]
    strides = [int(i) * type_size for i in strides]
    np_stride = tuple(strides)
    return np_stride


def filedata_to_tensor(filename, dtype, layout, shape, framework, device):
    if not check_framework(framework):
        return None
    np_dtype = to_np_dtype(dtype)
    np_shape = get_shape(layout, shape["dims"])
    if np_dtype is None or np_shape is None:
        return None

    if "dim_stride" not in shape:
        data = np.fromfile(filename, dtype=np_dtype).reshape(np_shape)
    else:
        data = np.fromfile(filename, dtype=np_dtype)
        np_stride = get_stride(shape["dim_stride"], data.itemsize)
        data = np.lib.stride_tricks.as_strided(data, np_shape, np_stride)
        data = np.ascontiguousarray(data)
    high_precision = os.environ.get('USE_HIGH_CUDA_PRECISION', '0')
    if framework == "torch":
        data = torch.tensor(data)
        if device == "cpu":
            if data.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
                data = data.type(torch.float64)
            elif data.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                data = data.type(torch.int64)
        elif device == "cuda":
            if data.dtype in [torch.float16, torch.bfloat16] and high_precision == '1':
                data = data.type(torch.float32)
            data = data.cuda()

    elif framework == "paddle":
        data = paddle.to_tensor(data)
        if device == "cpu":
            paddle.set_device("cpu")
            data = data.detach().cpu()
            if data.dtype in [paddle.float16, paddle.float32, paddle.float64, paddle.bfloat16]:
                data = paddle.cast(data, dtype=paddle.float64)
            elif data.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64]:
                data = paddle.cast(data, dtype=paddle.int64)
        elif device == "cuda":
            if data.dtype in [paddle.float16, paddle.bfloat16] and high_precision == '1':
                data = paddle.cast(data, dtype=paddle.float32)
            paddle.set_device("gpu")

    return data


def to_tensor(filename, params, framework="torch", device=""):
    if not check_framework(framework):
        return None
    return filedata_to_tensor(
        filename, params["dtype"], params["layout"], params["shape"], framework, device
    )


def tensor_to_NCHW(tensor, layout, framework="torch"):
    if not check_framework(framework):
        return None
    if framework == "torch":
        if layout == "LAYOUT_NCHW":
            return tensor
        elif layout == "LAYOUT_NHWC":
            return tensor.permute(0, 3, 1, 2).contiguous()
        elif layout == "LAYOUT_HWCN":
            return tensor.permute(3, 2, 0, 1).contiguous()
        elif layout == "LAYOUT_NWHC":
            return tensor.permute(0, 3, 2, 1).contiguous()
        elif layout == "LAYOUT_CHWN":
            return tensor.permute(3, 0, 1, 2).contiguous()
        else:
            print("not supported layout {}".format(layout))
            return None
    elif framework == "paddle":
        if layout == "LAYOUT_NCHW":
            return tensor
        elif layout == "LAYOUT_NHWC":
            return tensor.transpose([0, 3, 1, 2])
        elif layout == "LAYOUT_HWCN":
            return tensor.transpose([3, 2, 0, 1])
        elif layout == "LAYOUT_NWHC":
            return tensor.transpose([0, 3, 2, 1])
        elif layout == "LAYOUT_CHWN":
            return tensor.transpose([3, 0, 1, 2])
        else:
            print("not supported layout {}".format(layout))
            return None


def tensor_to_NCDHW(tensor, layout, framework="torch"):
    if not check_framework(framework):
        return None
    if framework == "torch":
        if layout == "LAYOUT_NCDHW":
            return tensor
        elif layout == "LAYOUT_NDHWC":
            return tensor.permute(0, 4, 1, 2, 3).contiguous()
        elif layout == "LAYOUT_CDHWN":
            return tensor.permute(4, 0, 1, 2, 3).contiguous()
        else:
            print("not supported layout {}".format(layout))
            return None
    elif framework == "paddle":
        if layout == "LAYOUT_NCDHW":
            return tensor
        elif layout == "LAYOUT_NDHWC":
            return tensor.transpose([0, 4, 1, 2, 3])
        elif layout == "LAYOUT_CDHWN":
            return tensor.transpose([4, 0, 1, 2, 3])
        else:
            print("not supported layout {}".format(layout))
            return None


def tensor_from_NCHW(tensor, layout, framework="torch"):
    if not check_framework(framework):
        return None
    if framework == "torch":
        if layout == "LAYOUT_NCHW":
            return tensor
        elif layout == "LAYOUT_NHWC":
            return tensor.permute(0, 2, 3, 1).contiguous()
        elif layout == "LAYOUT_HWCN":
            return tensor.permute(2, 3, 1, 0).contiguous()
        elif layout == "LAYOUT_NWHC":
            return tensor.permute(0, 3, 2, 1).contiguous()
        elif layout == "LAYOUT_CHWN":
            return tensor.permute(1, 2, 3, 0).contiguous()
        else:
            print("not supported layout {}".format(layout))
            return None
    elif framework == "paddle":
        if layout == "LAYOUT_NCHW":
            return tensor
        elif layout == "LAYOUT_NHWC":
            return tensor.transpose([0, 2, 3, 1])
        elif layout == "LAYOUT_HWCN":
            return tensor.transpose([2, 3, 1, 0])
        elif layout == "LAYOUT_NWHC":
            return tensor.transpose([0, 3, 2, 1])
        elif layout == "LAYOUT_CHWN":
            return tensor.transpose([1, 2, 3, 0])
        else:
            print("not supported layout {}".format(layout))
            return None


def tensor_from_NCDHW(tensor, layout, framework="torch"):
    if not check_framework(framework):
        return None
    if framework == "torch":
        if layout == "LAYOUT_NCDHW":
            return tensor
        if layout == "LAYOUT_NDHWC":
            return tensor.permute(0, 2, 3, 4, 1).contiguous()
        if layout == "LAYOUT_CDHWN":
            return tensor.permute(1, 2, 3, 4, 0).contiguous()
        else:
            print("not supported layout {}".format(layout))
            return None
    elif framework == "paddle":
        if layout == "LAYOUT_NCDHW":
            return tensor
        if layout == "LAYOUT_NDHWC":
            return tensor.permute(0, 2, 3, 4, 1)
        if layout == "LAYOUT_CDHWN":
            return tensor.permute(1, 2, 3, 4, 0)
        else:
            print("not supported layout {}".format(layout))
            return None


def transform(tensor, src_layout, dst_layout, framework="torch"):
    if framework == "torch":
        if src_layout == "LAYOUT_NCHW":
            if dst_layout == "LAYOUT_NCHW":
                return tensor
            elif dst_layout == "LAYOUT_NHWC":
                return tensor.permute(0, 2, 3, 1).contiguous()
            elif dst_layout == "LAYOUT_CHWN":
                return tensor.permute(1, 2, 3, 0).contiguous()
            elif dst_layout == "LAYOUT_NWHC":
                return tensor.permute(0, 3, 2, 1).contiguous()
        elif src_layout == "LAYOUT_NHWC":
            if dst_layout == "LAYOUT_NCHW":
                return tensor.permute(0, 3, 1, 2).contiguous()
            elif dst_layout == "LAYOUT_NHWC":
                return tensor
            elif dst_layout == "LAYOUT_CHWN":
                return tensor.permute(3, 1, 2, 0).contiguous()
            elif dst_layout == "LAYOUT_NWHC":
                return tensor.permute(0, 2, 1, 3).contiguous()
        elif src_layout == "LAYOUT_CHWN":
            if dst_layout == "LAYOUT_NCHW":
                return tensor.permute(3, 0, 1, 2).contiguous()
            elif dst_layout == "LAYOUT_NHWC":
                return tensor.permute(3, 1, 2, 0).contiguous()
            elif dst_layout == "LAYOUT_CHWN":
                return tensor
            elif dst_layout == "LAYOUT_NWHC":
                return tensor.permute(3, 2, 1, 0).contiguous()
        elif src_layout == "LAYOUT_NWHC":
            if dst_layout == "LAYOUT_NCHW":
                return tensor.permute(0, 3, 2, 1).contiguous()
            elif dst_layout == "LAYOUT_NHWC":
                return tensor.permute(0, 2, 1, 3).contiguous()
            elif dst_layout == "LAYOUT_CHWN":
                return tensor.permute(3, 2, 1, 0).contiguous()
            elif dst_layout == "LAYOUT_NWHC":
                return tensor
    elif framework == "paddle":
        if src_layout == "LAYOUT_NCHW":
            if dst_layout == "LAYOUT_NCHW":
                return tensor
            elif dst_layout == "LAYOUT_NHWC":
                return tensor.transpose([0, 2, 3, 1])
            elif dst_layout == "LAYOUT_CHWN":
                return tensor.transpose([1, 2, 3, 0])
            elif dst_layout == "LAYOUT_NWHC":
                return tensor.transpose([0, 3, 2, 1])
        elif src_layout == "LAYOUT_NHWC":
            if dst_layout == "LAYOUT_NCHW":
                return tensor.transpose([0, 3, 1, 2])
            elif dst_layout == "LAYOUT_NHWC":
                return tensor
            elif dst_layout == "LAYOUT_CHWN":
                return tensor.transpose([3, 1, 2, 0])
            elif dst_layout == "LAYOUT_NWHC":
                return tensor.transpose([0, 2, 1, 3])
        elif src_layout == "LAYOUT_CHWN":
            if dst_layout == "LAYOUT_NCHW":
                return tensor.transpose([3, 0, 1, 2])
            elif dst_layout == "LAYOUT_NHWC":
                return tensor.transpose([3, 1, 2, 0])
            elif dst_layout == "LAYOUT_CHWN":
                return tensor
            elif dst_layout == "LAYOUT_NWHC":
                return tensor.transpose([3, 2, 1, 0])
        elif src_layout == "LAYOUT_NWHC":
            if dst_layout == "LAYOUT_NCHW":
                return tensor.transpose([0, 3, 2, 1])
            elif dst_layout == "LAYOUT_NHWC":
                return tensor.transpose([0, 2, 1, 3])
            elif dst_layout == "LAYOUT_CHWN":
                return tensor.transpose([3, 2, 1, 0])
            elif dst_layout == "LAYOUT_NWHC":
                return tensor


def is_device_available(device, framework="torch"):
    used_device = ""
    if device == "cuda":
        if not check_framework(framework):
            return False, used_device
        if framework == "torch":
            if torch.cuda.is_available():
                used_device = torch.device("cuda:0")
            else:
                print("not found cuda device")
                return False, used_device
        elif framework == "paddle":
            if not paddle.fluid.is_compiled_with_cuda():
                print("not found cuda device")
                return False, used_device
    return True, used_device


def get_NCHW(shape, layout):
    if type(shape) != list:
        shape = [shape]
    shape = [int(i) for i in shape]
    if layout == "LAYOUT_CHWN":
        C, H, W, N = shape
    elif layout == "LAYOUT_NHWC":
        N, H, W, C = shape
    elif layout == "LAYOUT_NCHW":
        N, C, H, W = shape
    elif layout == "LAYOUT_NWHC":
        N, W, H, C = shape
    return N, C, H, W

def get_NCDHW(shape, layout):
    if type(shape) != list:
        shape = [shape]
    shape = [int(i) for i in shape]
    if layout == "LAYOUT_NCDHW":
        N, C, D, H, W = shape
    if layout == "LAYOUT_NDHWC":
        N, D, H, W, C = shape
    if layout == "LAYOUT_CDHWN":
        C, D, H, W, N = shape
    return N, C, D, H, W

def get_perf_json_path(param_path, op_name):
    import os

    is_get_cuda = os.getenv("DNN_ENABEL_CUDA_PERF")
    warm_repeats = os.getenv("DNN_WARM_REPEAT")
    cuda_json_path = param_path[: -len(".prototxt")] + "_cuda.json"
    return warm_repeats, is_get_cuda, cuda_json_path
