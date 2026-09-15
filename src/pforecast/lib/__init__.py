"""물리 컴포넌트 라이브러리."""

from .abatement import Stack, ToolGroupSource, WetScrubber
from .flow import Duct, Fan, GasComponent, Mixer
from .gas import FLUE_GAS, GasMedium, gas_port

#: YAML 모델 로더가 참조하는 컴포넌트 레지스트리.
REGISTRY = {
    "Duct": Duct,
    "Fan": Fan,
    "Mixer": Mixer,
    "ToolGroupSource": ToolGroupSource,
    "WetScrubber": WetScrubber,
    "Stack": Stack,
}

__all__ = [
    "Duct", "Fan", "Mixer", "GasComponent", "Stack", "ToolGroupSource",
    "WetScrubber", "GasMedium", "FLUE_GAS", "gas_port", "REGISTRY",
]
