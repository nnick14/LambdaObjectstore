"""
Before running, make sure the proxy is running and that the shared go library has been built via
the command: `go build -o ecClient.so -buildmode=c-shared go_client.go`
"""
from __future__ import annotations

from ctypes import CDLL, c_char_p, c_void_p, cdll, string_at
from typing import TypeVar

NumpyDtype = TypeVar("NumpyDtype")

NEGATIVE_ASCII_VALUE = 45

# # Run: `go build -o ecClient.so -buildmode=c-shared go_client.go`
def load_go_lib(library_path: str) -> CDLL:
    """Load the Go library that was exported to a .so file."""
    return cdll.LoadLibrary(library_path)


def get_array_from_cache(
    go_library: CDLL, cache_key: str, length: int
) -> bytes:
    """
    Example:
        go_library = load_go_lib(args.go_lib_path)
        cache_key = "test_" + str(random.randint(0, 50000))
    """
    # Need to make sure to free any pointers
    go_library.free.argtypes = [c_void_p]
    go_library.getFromCache.argtypes = [c_char_p]
    go_library.getFromCache.restype = c_void_p

    clientResultPtr = go_library.getFromCache(cache_key.encode("utf-8"))
    clientResultStr = string_at(clientResultPtr, length)

    go_library.free(clientResultPtr)
    if clientResultStr[0] == NEGATIVE_ASCII_VALUE:
        raise KeyError("Key is not in cache")
    return clientResultStr


def set_array_in_cache(go_library: CDLL, cache_key: str, input: bytes):
    """
    Example:
        go_library = load_go_lib(args.go_lib_path)
        cache_key = "test_" + str(random.randint(0, 50000))
        input_arr = np.random.randn(2, 2)
    """
    # Need to make sure to free any pointers
    go_library.free.argtypes = [c_void_p]
    go_library.setInCache.argtypes = [c_char_p, c_char_p]

    go_library.setInCache(cache_key.encode("utf-8"), input, len(input))

GO_LIB = None
