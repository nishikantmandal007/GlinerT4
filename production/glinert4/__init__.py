from .detector import Detector, DetectorConfig
from .memory import MemoryConfig
from .runtime import GLINER2_VERSION

__all__ = ["Detector", "DetectorConfig", "MemoryConfig", "install", "uninstall", "GLINER2_VERSION"]


def install(kernels=True):
    from . import count, overlap, runtime

    runtime.check_gliner2()
    runtime.install_word_cache()
    count.install()
    overlap.install()
    if kernels:
        from . import attention, layernorm

        attention.install()
        layernorm.install()


def uninstall():
    from . import attention, count, layernorm, overlap, runtime

    attention.uninstall()
    layernorm.uninstall()
    count.uninstall()
    overlap.uninstall()
    runtime.uninstall_word_cache()
