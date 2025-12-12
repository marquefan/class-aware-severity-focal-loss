from .config import CAFLConfig
from .losses.cafl import CAFLoss
from .weighting.effective_number import effective_number_weights
from .weighting.severity import SeverityMap
from .weighting.similarity import ClassEmbeddingSimilarity
from .adapters.torchvision_retinanet import (
    RetinanetCAFLClassificationHead,
    swap_in_cafl_head,
)
