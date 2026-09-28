"""물리 컴포넌트 라이브러리."""

from .abatement import Stack, ToolGroupSource, WetScrubber
from .flow import Duct, Fan, GasComponent, Mixer
from .gas import FLUE_GAS, GasMedium, gas_port
from .generic import (MEDIA, PORT_KINDS, EquationComponent, domain_functions,
                      load_component_spec)

#: YAML 모델 로더가 참조하는 컴포넌트 레지스트리.
REGISTRY = {
    "Duct": Duct,
    "Fan": Fan,
    "Mixer": Mixer,
    "ToolGroupSource": ToolGroupSource,
    "WetScrubber": WetScrubber,
    "Stack": Stack,
    # 선언형: YAML 에 방정식을 직접 적는다
    "equation": EquationComponent,
    "Equation": EquationComponent,
}

__all__ = [
    "Duct", "Fan", "Mixer", "GasComponent", "Stack", "ToolGroupSource",
    "WetScrubber", "GasMedium", "FLUE_GAS", "gas_port", "REGISTRY",
    "EquationComponent", "load_component_spec", "PORT_KINDS", "MEDIA",
    "domain_functions",
]
