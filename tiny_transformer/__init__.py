import importlib.metadata

try:
    __version__ = importlib.metadata.version("tiny-transformer-from-scratch")
except importlib.metadata.PackageNotFoundError:
    pass
