"""데이터 주도 분석 사다리: 프로파일링 -> 대리모델/XAI -> 차원해석 -> 보존식 탐지."""

from .dimensional import Balance, PiGroup, find_balances, null_space_int, pi_groups
from .pipeline import AnalysisConfig, AnalysisResult, Rung, run_analysis
from .profile import ColumnProfile, DatasetProfile, Envelope, profile_dataset, steady_state_mask
from .surrogate import (Importance, Improvement, LagInfo, SurrogateResult, find_lags,
                        fit_surrogate, partial_dependence, permutation_importance,
                        search_improvement, time_blocked_folds)

__all__ = [
    "run_analysis", "AnalysisConfig", "AnalysisResult", "Rung",
    "profile_dataset", "DatasetProfile", "ColumnProfile", "Envelope", "steady_state_mask",
    "find_balances", "Balance", "pi_groups", "PiGroup", "null_space_int",
    "fit_surrogate", "SurrogateResult", "Importance", "LagInfo", "find_lags",
    "permutation_importance", "partial_dependence", "time_blocked_folds",
    "search_improvement", "Improvement",
]
