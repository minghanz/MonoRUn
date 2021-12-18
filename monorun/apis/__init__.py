from .inference import init_detector, inference_detector
from .test import single_gpu_test, single_gpu_eval

__all__ = ['init_detector', 'inference_detector', 'single_gpu_test', 'single_gpu_eval']
