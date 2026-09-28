"""자체 완결형 HTML 리포트 생성기.

matplotlib 그림을 base64 로 박아 넣어 파일 하나로 끝낸다. 사내망 어디서 열어도
외부 리소스를 받아오지 않는다. matplotlib 이 없으면 표만 있는 리포트로 떨어진다.

차트 규칙(범주형 색 고정 순서, 단일 축, 범례 상시, 표 동반)은 style.py 참고.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .style import INK_MUTED, INK_SECONDARY, SERIES_LIGHT, STATUS, setup_style

_CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;
--ink2:#52514e;--ink3:#8a8985;--line:#e6e5e1;--accent:#2a78d6;
--good:#1baf7a;--warn:#eda100;--bad:#e34948;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--surface:#1a1a19;--panel:#222221;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#8a8985;
--line:#383835;--accent:#3987e5;--good:#199e70;--warn:#c98500;--bad:#e66767;}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:14px/1.6 "Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}
h1{font-size:24px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:17px;margin:40px 0 12px;padding-top:20px;border-top:1px solid var(--line)}
h2:first-of-type{border-top:none;padding-top:0}
h3{font-size:14px;margin:22px 0 8px;color:var(--ink2)}
.meta{color:var(--ink3);font-size:12.5px}
p{color:var(--ink2);margin:10px 0}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}
.tile{flex:1 1 170px;min-width:150px;background:var(--panel);border:1px solid var(--line);
border-radius:10px;padding:14px 16px}
.tile .k{font-size:11.5px;color:var(--ink3);text-transform:none;letter-spacing:.02em}
.tile .v{font-size:25px;font-weight:650;margin-top:3px;letter-spacing:-.02em;
font-variant-numeric:tabular-nums}
.tile .u{font-size:12px;color:var(--ink3);font-weight:400;margin-left:3px}
.tile .n{font-size:11.5px;color:var(--ink3);margin-top:4px}
.tile.good .v{color:var(--good)} .tile.warn .v{color:var(--warn)} .tile.bad .v{color:var(--bad)}
figure{margin:18px 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
border-radius:10px;background:#fcfcfb}
figcaption{color:var(--ink3);font-size:12.5px;margin-top:8px}
.tablewrap{overflow-x:auto;margin:14px 0;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--panel);color:var(--ink3);font-weight:600;text-align:right;
position:sticky;top:0}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
td{font-variant-numeric:tabular-nums;color:var(--ink2)}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:8px;padding:12px 16px;margin:16px 0;font-size:13px;color:var(--ink2)}
.note.warn{border-left-color:var(--warn)} .note.bad{border-left-color:var(--bad)}
.note b{color:var(--ink)}
ul{color:var(--ink2);padding-left:20px} li{margin:4px 0}
code{background:var(--panel);border:1px solid var(--line);border-radius:4px;
padding:1px 5px;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace}
footer{margin-top:60px;padding-top:16px;border-top:1px solid var(--line);
color:var(--ink3);font-size:12px}
"""


def fig_to_img(fig, alt: str = "") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'


def df_to_table(df: pd.DataFrame, float_fmt: str = "{:,.4g}", max_rows: int = 60) -> str:
    d = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
                cells.append(f"<td>{float_fmt.format(v)}</td>")
            elif isinstance(v, (bool, np.bool_)):
                cells.append(f"<td>{'O' if v else 'X'}</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    more = ""
    if len(df) > max_rows:
        more = f'<p class="meta">... 전체 {len(df)}행 중 {max_rows}행 표시</p>'
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>{more}')


@dataclass
class Report:
    """섹션을 차곡차곡 쌓아 HTML 한 장으로 뽑는다."""

    title: str
    subtitle: str = ""
    parts: list[str] = field(default_factory=list)

    def h2(self, text: str) -> "Report":
        self.parts.append(f"<h2>{html.escape(text)}</h2>")
        return self

    def h3(self, text: str) -> "Report":
        self.parts.append(f"<h3>{html.escape(text)}</h3>")
        return self

    def text(self, body: str) -> "Report":
        self.parts.append(f"<p>{html.escape(body)}</p>")
        return self

    def html(self, raw: str) -> "Report":
        self.parts.append(raw)
        return self

    def note(self, body: str, kind: str = "") -> "Report":
        cls = f"note {kind}".strip()
        self.parts.append(f'<div class="{cls}">{body}</div>')
        return self

    @staticmethod
    def _emphasis(text: str) -> str:
        """``**강조**`` 를 ``<b>`` 로. 판정 문장이 마크다운으로 오므로 전부 변환한다."""
        import re
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    def bullets(self, items: list[str], markdown: bool = False) -> "Report":
        li = "".join(f"<li>{self._emphasis(i) if markdown else i}</li>" for i in items)
        self.parts.append(f"<ul>{li}</ul>")
        return self

    def tiles(self, items: list[dict[str, Any]]) -> "Report":
        """KPI 타일. ``{label, value, unit, note, status}``."""
        cells = []
        for t in items:
            st = t.get("status", "")
            note = f'<div class="n">{html.escape(str(t["note"]))}</div>' if t.get("note") else ""
            unit = f'<span class="u">{html.escape(str(t.get("unit","")))}</span>' if t.get("unit") else ""
            cells.append(
                f'<div class="tile {st}"><div class="k">{html.escape(str(t["label"]))}</div>'
                f'<div class="v">{html.escape(str(t["value"]))}{unit}</div>{note}</div>'
            )
        self.parts.append(f'<div class="tiles">{"".join(cells)}</div>')
        return self

    def figure(self, fig, caption: str = "") -> "Report":
        cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
        self.parts.append(f"<figure>{fig_to_img(fig, caption)}{cap}</figure>")
        return self

    def table(self, df: pd.DataFrame, caption: str = "", **kw) -> "Report":
        cap = f'<p class="meta">{html.escape(caption)}</p>' if caption else ""
        self.parts.append(df_to_table(df, **kw) + cap)
        return self

    def render(self, path: str | Path) -> Path:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{html.escape(self.title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header><h1>{html.escape(self.title)}</h1>
<div class="meta">{html.escape(self.subtitle)} · 생성 {now} · pforecast</div></header>
{''.join(self.parts)}
<footer>물리 지배방정식 기반 예측 (pforecast). 이 문서는 외부 리소스를 참조하지 않는
단일 HTML 파일입니다.</footer>
</div></body></html>"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(doc, encoding="utf-8")
        return p


# --- 자주 쓰는 그림 ---------------------------------------------------------

def timeseries_fig(index, series: dict[str, np.ndarray], ylabel: str,
                   title: str = "", spans: list[tuple] | None = None, figsize=(10, 3.4)):
    """시계열 비교. 계열은 최대 3개 (그 이상이면 차트를 나눈다)."""
    plt = setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    for (name, y), c in zip(series.items(), SERIES_LIGHT):
        ax.plot(index, y, color=c, label=name, linewidth=1.6)
    for span in spans or []:
        lo, hi, label, color = span
        ax.axvspan(lo, hi, color=color, alpha=0.09, lw=0)
        # 축 분수 좌표를 써야 데이터 범위와 무관하게 상단에 붙는다
        ax.text(lo, 0.98, f" {label}", va="top", ha="left", fontsize=9,
                color=INK_MUTED, transform=ax.get_xaxis_transform())
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) >= 2:
        ax.legend(loc="upper left", ncols=min(len(series), 3))
    fig.autofmt_xdate()
    return fig


def parity_fig(pairs: dict[str, tuple[np.ndarray, np.ndarray]], label: str,
               title: str = "", figsize=(4.6, 4.4)):
    """예측-실측 대응도. 1:1 선과 +-10% 띠를 같이 그린다."""
    plt = setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    allv = np.concatenate([np.concatenate(v) for v in pairs.values()])
    allv = allv[np.isfinite(allv)]
    lo, hi = float(np.min(allv)), float(np.max(allv))
    pad = 0.06 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                    color=INK_MUTED, alpha=0.10, lw=0, label="±10%")
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1.2, ls="--", zorder=2)
    for (name, (meas, pred)), c in zip(pairs.items(), SERIES_LIGHT):
        ax.scatter(meas, pred, s=9, color=c, alpha=0.55, lw=0, label=name, zorder=3)
    ax.set_xlabel(f"실측 {label}")
    ax.set_ylabel(f"예측 {label}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left")
    return fig


def grouped_bar_fig(categories, series: dict[str, np.ndarray], xlabel: str,
                    title: str = "", value_fmt: str = "{:.2f}", figsize=(7.6, None),
                    threshold: float | None = None, threshold_label: str = ""):
    """가로 막대 비교. 계열은 같은 단위여야 한다 (이중 축은 쓰지 않는다).

    값은 막대 끝에 직접 적는다 - 범례만 있고 숫자가 없으면 눈으로 재야 한다.
    ``threshold`` 를 주면 판정선을 세로 점선으로 긋는다.
    """
    plt = setup_style()
    n_cat, n_ser = len(categories), len(series)
    h = figsize[1] or max(2.3, 0.42 * n_cat * n_ser + 1.4)
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    y = np.arange(n_cat)
    bar_h = 0.62 / n_ser                       # 얇게: 막대 사이에 지면이 보이도록
    vmax = max(float(np.nanmax(v)) for v in series.values())
    if threshold is not None:
        vmax = max(vmax, threshold)
    for i, ((name, vals), c) in enumerate(zip(series.items(), SERIES_LIGHT)):
        off = (i - (n_ser - 1) / 2) * bar_h
        ax.barh(y - off, vals, height=bar_h * 0.86, color=c, label=name, zorder=3)
        for yy, v in zip(y - off, vals):
            if np.isfinite(v):
                ax.text(v + vmax * 0.015, yy, value_fmt.format(v), va="center", ha="left",
                        fontsize=8.5, color=INK_SECONDARY, zorder=4)
    if threshold is not None:
        ax.axvline(threshold, color=STATUS["critical"], lw=1.6, ls=(0, (4, 3)), zorder=2)
        ax.text(threshold, 1.005, f" {threshold_label}", ha="left", va="bottom",
                fontsize=9, color=STATUS["critical"], transform=ax.get_xaxis_transform())
    ax.set_yticks(y, [str(c) for c in categories])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, vmax * 1.25)
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, pad=24 if threshold is not None else 12)
    if n_ser >= 2:
        # 막대가 화면을 가득 채우므로 범례는 축 위로 뺀다
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.12), mode="expand",
                  ncols=min(n_ser, 3), borderaxespad=0)
    return fig


def sweep_fig(x, series: dict[str, np.ndarray], xlabel: str, ylabel: str,
              limit: float | None = None, limit_label: str = "관리기준",
              train_range: tuple | None = None, title: str = "", figsize=(7.2, 4.0)):
    """스윕 곡선. 관리기준선과 학습구간을 함께 표시한다."""
    plt = setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    for (name, y), c in zip(series.items(), SERIES_LIGHT):
        ax.plot(x, y, color=c, label=name, marker="o", markersize=3.6,
                markeredgewidth=0, linewidth=2.0, zorder=3)
    if train_range is not None:
        ax.axvspan(train_range[0], train_range[1], color=SERIES_LIGHT[0], alpha=0.08,
                   lw=0, zorder=0)
        ax.text(float(np.mean(train_range)), 0.02, "보정에 쓴 구간", ha="center",
                va="bottom", fontsize=9, color=INK_MUTED,
                transform=ax.get_xaxis_transform())
    if limit is not None:
        ax.axhline(limit, color=STATUS["critical"], lw=1.6, ls=(0, (4, 3)), zorder=2)
        ax.text(0.995, limit, f"{limit_label} {limit:g} ", va="bottom", ha="right",
                fontsize=9, color=STATUS["critical"],
                transform=ax.get_yaxis_transform())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=14)
    if len(series) >= 2:
        # 상단은 관리기준선 라벨이 쓰므로 범례는 아래 왼쪽에 둔다
        ax.legend(loc="lower right", ncols=min(len(series), 3))
    return fig
