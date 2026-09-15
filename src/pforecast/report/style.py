"""리포트 차트 공통 스타일.

색상은 고정 순서로만 배정한다(순환 금지). 3개 슬롯까지만 쓰고, 그 이상이 필요하면
차트를 나눈다. 축은 하나만 쓴다 (이중 y축 금지 - 단위가 다르면 차트를 나눈다).
"""

from __future__ import annotations

#: 범주형 색상 슬롯 (고정 순서). 라이트/다크 각각 검증된 값.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70"]

STATUS = {
    "good": "#1baf7a",
    "warning": "#eda100",
    "critical": "#e34948",
}

SURFACE_LIGHT = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

#: 한글이 되는 폰트를 우선순위대로 찾는다. 사내 Windows PC 는 'Malgun Gothic' 이 잡힌다.
KOREAN_FONTS = [
    "Malgun Gothic", "NanumGothic", "NanumBarunGothic", "Apple SD Gothic Neo",
    "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR", "WenQuanYi Zen Hei",
    "Source Han Sans KR",
]


def pick_font() -> str | None:
    try:
        import matplotlib.font_manager as fm
    except ImportError:
        return None
    have = {f.name for f in fm.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in have:
            return name
    return None


def setup_style():
    """matplotlib 전역 스타일. 격자/축은 뒤로 물리고 데이터가 앞에 오게 한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = pick_font()
    rc = {
        "figure.facecolor": SURFACE_LIGHT,
        "axes.facecolor": SURFACE_LIGHT,
        "savefig.facecolor": SURFACE_LIGHT,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.0,
        "lines.solid_capstyle": "round",
        "figure.dpi": 130,
        "font.size": 10,
    }
    if font:
        rc["font.family"] = font
        rc["axes.unicode_minus"] = False
    plt.rcParams.update(rc)
    return plt
