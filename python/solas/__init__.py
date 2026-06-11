print("loading solas")
#从底层 ctypes 封装层拿到 C API 函数。
from solas.libsolas import solas_last_error, solas_version
from solas.tensor import Tensor


#给用户提供一个更干净的接口。
def version() -> str:
    return solas_version()

def last_error() -> str:
    return solas_last_error()