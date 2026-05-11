from .methods.gaf_transformation import GAF
from .methods.mtf_transformation import MTF
from .methods.stft_transformation import STFTSpectrogram
from .types import ImageTransformationType


TRANSFORMATION_MAPPING = {
    ImageTransformationType.MTF: MTF,
    ImageTransformationType.GAF: GAF,
    ImageTransformationType.STFT: STFTSpectrogram,
}