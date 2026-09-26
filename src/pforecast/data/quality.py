"""데이터 품질 진단 — 모델에 넣기 전에 계측이 무슨 짓을 했는지 본다.

수집기(:mod:`.ingest`)가 **파일 형식**을 고친다면, 여기서는 **값**을 본다.

탐지 항목
---------
* **고착** — 평소 흔들리던 값이 한 값에 붙어 있다 (통신 모듈 이상, 마지막 값 유지).
* **스파이크** — 짧고 큰 튐 (열전대 단선 999.9, 시료 라인 응축).
* **정기 교정** — 매일 같은 시각에 되풀이되는 튐. 분석계의 영점/스팬 가스 점검이다.
  한두 점이 아니라 매일 들어가므로, 걸러내지 않으면 대리모델이 '새벽 3시 효과'를 배운다.
* **계단형 갱신** — 한 시간에 한 번만 바뀌는 값 (MES 집계). 5분 해상도 정보가 없다.
* **정지(0) 구간** — 값이 정확히 0 에 머문다. 고착이 아니라 설비가 선 것이다. 지우지 않는다.
* **장기 결측** — 센서 정비·고장으로 하루 이상 비어 있다.

원칙
----
* 기본 조치는 '그 점을 결측으로' 다. 값을 지어내서 채우지 않는다. 보간은 모델이 아닌
  가정을 데이터에 끼워 넣는다.
* **계단 변화(레벨 시프트)는 여기서 판정하지 않는다.** 증설·설정 변경도 원시 신호에
  계단을 만든다. 센서 문제와 공정 변화를 가르는 것은 물리 모델 잔차다
  (:mod:`pforecast.calib.diagnostics`).
* 스파이크 임계는 보수적으로 잡는다 (국소 중앙값 대비 잡음의 8배). 실제 공정 이벤트
  (펌프 정지로 NOx 가 오르는 것)를 이상치로 지우면, 모델이 배워야 할 것을 지우는 셈이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Issue:
    column: str
    kind: str              # frozen | spike | calibration | step_signal | zero_hold | long_missing
    label: str             # 화면에 쓰는 짧은 이름
    message: str
    n_points: int = 0      # 결측 처리한 점 수 (0 이면 정보성)
    start: str | None = None
    end: str | None = None

    @property
    def excluded(self) -> bool:
        return self.n_points > 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class QualityReport:
    n_rows: int
    interval_s: float
    issues: list[Issue] = field(default_factory=list)
    masks: dict[str, np.ndarray] = field(default_factory=dict)   # True = 제외

    def excluded(self, column: str | None = None) -> int:
        if column is not None:
            m = self.masks.get(column)
            return int(m.sum()) if m is not None else 0
        return int(sum(int(m.sum()) for m in self.masks.values()))

    def by_column(self) -> dict[str, list[Issue]]:
        out: dict[str, list[Issue]] = {}
        for i in self.issues:
            out.setdefault(i.column, []).append(i)
        return out

    def lines(self) -> list[str]:
        if not self.issues:
            return ["값 수준의 이상은 찾지 못했습니다."]
        out = []
        n_ex = self.excluded()
        cols = sorted({i.column for i in self.issues if i.excluded})
        if n_ex:
            out.append(f"{len(cols)}개 컬럼에서 {n_ex:,}점을 결측으로 처리했습니다 "
                       "(값을 지어내 채우지 않습니다).")
        for kind in ("calibration", "frozen", "spike", "zero_hold", "long_missing"):
            for i in self.issues:
                if i.kind == kind:
                    out.append(f"[{i.label}] {i.column}: {i.message}")
        # 계단형 신호는 정보성이라 종류별로 한 줄에 모은다
        groups: dict[str, list[str]] = {}
        for i in self.issues:
            if i.kind == "step_signal":
                groups.setdefault(i.label, []).append(i.column)
        why = {"1시간 갱신": "한 시간에 한 번만 값이 바뀜 — 시간 내 변동은 이 컬럼으로 설명 불가",
               "설정값": "계단형 설정값/상태값 — 고착·스파이크 검사 생략",
               "상수": "한 번도 바뀌지 않음 — 영향 식별 불가"}
        for label, cols in groups.items():
            out.append(f"[{label}] {', '.join(cols)}: {why.get(label, '')}")
        return out

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "컬럼": i.column, "항목": i.label, "제외 점수": i.n_points,
            "시작": i.start, "끝": i.end, "내용": i.message} for i in self.issues])

    def to_dict(self) -> dict:
        return {"n_rows": self.n_rows, "interval_s": self.interval_s,
                "excluded": self.excluded(),
                "excluded_by_column": {c: int(m.sum()) for c, m in self.masks.items() if m.any()},
                "issues": [i.to_dict() for i in self.issues], "lines": self.lines()}


# --------------------------------------------------------------------------

def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """같은 값이 이어지는 구간 [시작, 끝) 목록 (값 배열은 결측 없이 압축된 것)."""
    if len(values) == 0:
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(values)]
    return list(zip(starts.tolist(), ends.tolist()))


def _bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _ts(idx, i) -> str | None:
    if isinstance(idx, pd.DatetimeIndex) and 0 <= i < len(idx):
        return str(idx[i])
    return None


def _interval(df: pd.DataFrame) -> float:
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 2:
        d = np.diff(df.index.to_numpy().astype("datetime64[s]").astype("int64"))
        d = d[d > 0]
        if len(d):
            return float(np.median(d))
    return float("nan")


def _noise_scale(x: np.ndarray) -> float:
    """고주파 잡음의 강건 추정. 1차 차분의 MAD 를 쓴다 (느린 추세에 휘둘리지 않는다).

    하한은 **분해능**(0 이 아닌 가장 작은 차분)이다. 0.01 단위로 반올림된 매끄러운
    신호는 차분이 전부 0 또는 ±0.01 이라 MAD 가 0 에 붙고, 그러면 0.01 만 움직여도
    스파이크가 된다 (예제 인버터 주파수에서 실제로 4건 오탐이 났다).
    """
    d = np.diff(x[np.isfinite(x)])
    d = d[d != 0]
    if len(d) < 10:
        return 0.0
    resolution = float(np.min(np.abs(d)))
    return max(float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)), resolution)


def assess(df: pd.DataFrame, columns: list[str] | None = None, *,
           spike_k: float = 8.0, spike_window: int = 7, spike_max_run: int = 6,
           min_days_recurring: int = 10, recurring_frac: float = 0.3) -> QualityReport:
    """컬럼별 값 이상을 찾는다. 규칙적인 시간 격자를 가정한다 (:func:`read_table` 결과)."""
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    interval = _interval(df)
    rep = QualityReport(n_rows=len(df), interval_s=interval)
    idx = df.index
    per_hour = 3600.0 / interval if np.isfinite(interval) and interval > 0 else np.nan
    per_day = 86400.0 / interval if np.isfinite(interval) and interval > 0 else np.nan

    alive = df[cols].notna().sum(axis=1).to_numpy() > max(1, 0.3 * len(cols))
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x)
        if finite.sum() < 20:
            continue
        mask = np.zeros(len(x), dtype=bool)
        pos = np.flatnonzero(finite)
        xv = x[pos]
        same = float(np.mean(xv[1:] == xv[:-1])) if len(xv) > 1 else 0.0

        # ---- 계단형 신호 (설정값, 상태값, 저빈도 갱신) --------------------------
        if same >= 0.8:
            changes = pos[1:][xv[1:] != xv[:-1]]
            msg = None
            if len(changes) == 0:
                rep.issues.append(Issue(
                    c, "step_signal", "상수",
                    f"값이 {xv[0]:g} 에서 한 번도 바뀌지 않습니다 — 죽은 입력이거나 고정값입니다. "
                    "어떤 영향도 식별할 수 없습니다."))
                continue
            run_len = np.diff(np.r_[0, np.flatnonzero(xv[1:] != xv[:-1]) + 1, len(xv)])
            if isinstance(idx, pd.DatetimeIndex) and len(changes) >= 24 \
                    and np.isfinite(per_hour) and per_hour > 1 \
                    and np.median(run_len) <= 1.5 * per_hour:
                on_hour = np.mean((idx[changes].minute == 0) & (idx[changes].second == 0))
                if on_hour > 0.9:
                    msg = ("한 시간에 한 번만 값이 바뀝니다 (집계값). "
                           f"{int(per_hour)}행 중 11행은 새 정보가 없습니다 — 시간 내 변동은 "
                           "이 컬럼으로 설명할 수 없습니다.")
                    rep.issues.append(Issue(c, "step_signal", "1시간 갱신", msg))
            if msg is None:
                n_change = int(np.sum(xv[1:] != xv[:-1]))
                rep.issues.append(Issue(
                    c, "step_signal", "설정값",
                    f"값이 {n_change:,}번만 바뀌는 계단형 신호입니다 (설정값/상태값으로 보임). "
                    "고착·스파이크 검사를 하지 않습니다."))
            continue

        # ---- 고착 / 정지(0) ----------------------------------------------------
        p = max(same, 1e-3)
        expect = np.log(max(len(xv), 2)) / max(-np.log(p), 1e-6) + 1.0
        min_run = int(max(12, np.ceil(3.0 * expect)))
        for a, b in _runs(xv):
            if b - a < min_run:
                continue
            i0, i1 = pos[a], pos[b - 1]
            hours = (i1 - i0 + 1) / per_hour if np.isfinite(per_hour) else float("nan")
            if xv[a] == 0.0:
                rep.issues.append(Issue(
                    c, "zero_hold", "정지(0)",
                    f"값이 정확히 0 으로 {hours:.1f}시간 유지 — 설비 정지로 보고 **지우지 "
                    "않습니다** (고착이면 0 이 아닌 마지막 값에 붙는다).",
                    start=_ts(idx, i0), end=_ts(idx, i1)))
                continue
            span = pos[a + 1:b]
            mask[span] = True
            rep.issues.append(Issue(
                c, "frozen", "고착",
                f"{xv[a]:g} 에 {hours:.1f}시간 붙어 있음 (평소엔 "
                + ("거의 매 점 바뀌는 값" if same < 0.05 else f"{1/(1-same):.1f}점마다 바뀌는 값")
                + "). 첫 점만 남기고 결측 처리.",
                n_points=len(span), start=_ts(idx, i0), end=_ts(idx, i1)))

        # ---- 스파이크 (Hampel) -------------------------------------------------
        sigma = _noise_scale(x)
        q75, q25 = np.nanpercentile(x, [75, 25])
        sigma = max(sigma, 1e-3 * (q75 - q25), 1e-12)

        def hampel(excluded: np.ndarray) -> np.ndarray:
            s = pd.Series(np.where(excluded, np.nan, x))
            med = s.rolling(spike_window, center=True, min_periods=3).median().to_numpy()
            dev = np.abs(s.to_numpy() - med)
            flag = np.isfinite(dev) & (dev > spike_k * sigma)
            out = np.zeros(len(x), dtype=bool)
            for a, b in _bool_runs(flag):
                if b - a <= spike_max_run:
                    out[a:b] = True
            return out

        spikes = hampel(mask)

        # ---- 정기 교정: 매일 같은 시각에 되풀이되는 스파이크 ------------------------
        cal = np.zeros(len(x), dtype=bool)
        if spikes.sum() >= min_days_recurring and isinstance(idx, pd.DatetimeIndex) \
                and np.isfinite(per_day):
            slot = ((idx.hour * 3600 + idx.minute * 60 + idx.second) // interval).to_numpy()
            # 날짜는 정수(일 번호)로 비교한다. Timestamp 집합에 np.isin 을 쓰면 조용히 전부
            # 불일치가 난다 (datetime64 와 Timestamp 객체의 비교).
            day = (idx.normalize().to_numpy().astype("datetime64[D]").astype("int64"))
            n_days = int(len(np.unique(day[finite])))
            hit = pd.DataFrame({"slot": slot[spikes], "day": day[spikes]}).drop_duplicates()
            per_slot = hit.groupby("slot")["day"].nunique()
            need = max(min_days_recurring, recurring_frac * n_days)
            rec = sorted(int(s_) for s_ in per_slot[per_slot >= need].index)
            if rec:
                # 이웃 슬롯(회복 구간)까지 한 창으로 묶는다
                windows, cur = [], [rec[0], rec[0]]
                for s_ in rec[1:]:
                    if s_ <= cur[1] + 2:
                        cur[1] = s_
                    else:
                        windows.append(cur)
                        cur = [s_, s_]
                windows.append(cur)
                for w0, w1 in windows:
                    in_win = (slot >= w0) & (slot <= w1)
                    days_hit = np.unique(day[spikes & in_win])
                    on = in_win & np.isin(day, days_hit)
                    cal |= on & finite
                    t0 = pd.Timestamp(0) + pd.Timedelta(seconds=w0 * interval)
                    t1 = pd.Timestamp(0) + pd.Timedelta(seconds=(w1 + 1) * interval)
                    vals = x[on & finite]
                    lo, hi = (np.percentile(vals, [10, 90]) if len(vals) else (np.nan, np.nan))
                    rep.issues.append(Issue(
                        c, "calibration", "정기 교정",
                        f"매일 {t0:%H:%M}~{t1:%H:%M} 에 같은 모양의 튐이 {len(days_hit)}일 "
                        f"반복됩니다 (값 {lo:.4g} ~ {hi:.4g}). 분석계 영점/스팬 점검으로 "
                        "보입니다 — 그 창 전체를 결측 처리.",
                        n_points=int(on.sum())))
        if cal.any():
            # 교정 창을 빼고 다시 본다. 창 옆 행이 비어 있으면 이동 중앙값이 교정값에
            # 끌려가 멀쩡한 이웃 점이 스파이크로 잡힌다 (실제로 03:25 에서 5건 났다).
            spikes = hampel(mask | cal) & ~cal
        if spikes.any():
            runs = _bool_runs(spikes)
            ex = x[spikes]
            rep.issues.append(Issue(
                c, "spike", "스파이크",
                f"{len(runs)}건 ({int(spikes.sum())}점) — 국소 중앙값에서 잡음의 "
                f"{spike_k:g}배 이상 벗어난 짧은 튐. 예: {', '.join(f'{v:g}' for v in ex[:3])}",
                n_points=int(spikes.sum()),
                start=_ts(idx, runs[0][0]), end=_ts(idx, runs[-1][1] - 1)))
        mask |= spikes | cal

        # ---- 장기 결측 (다른 컬럼은 살아 있는데 이것만 빈 구간) --------------------
        if np.isfinite(per_day):
            # 전 태그가 빈 행(통신 두절, 누락 시각)은 판단에서 빼고 이어 붙인다. 안 그러면
            # 한 번 죽은 센서가 통신 두절마다 끊겨 수십 건으로 보고된다.
            alive_idx = np.flatnonzero(alive)
            for a, b in _bool_runs(~finite[alive_idx]):
                i0, i1 = alive_idx[a], alive_idx[b - 1]
                if i1 - i0 + 1 < per_day:
                    continue
                tail = not finite[i1 + 1:].any()
                rep.issues.append(Issue(
                    c, "long_missing", "센서 정지" if tail else "장기 결측",
                    (f"{_ts(idx, i0)} 이후 값이 없습니다 — 고장/철거 추정"
                     if tail else f"{(i1 - i0 + 1) / per_day:.1f}일 동안 이 컬럼만 비어 있습니다 "
                     "— 센서 정비/고장 추정"),
                    start=_ts(idx, i0), end=_ts(idx, i1)))

        if mask.any():
            rep.masks[c] = mask
    return rep


def apply(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """진단에서 제외하기로 한 점을 결측으로 바꾼 사본."""
    out = df.copy()
    for c, m in report.masks.items():
        if c in out.columns and len(m) == len(out):
            col = np.array(out[c].to_numpy(dtype=float), copy=True)
            col[m] = np.nan
            out[c] = col
    return out
