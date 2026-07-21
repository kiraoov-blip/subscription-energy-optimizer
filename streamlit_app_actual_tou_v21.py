from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from calendar import monthrange
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ortools.sat.python import cp_model

APP_VERSION = "2026-07-21-actual-tou-v21.0"
BASE_DIR = Path(__file__).resolve().parent

DATA_FILES = {
    "summary": "summary.json",
    "stats": "analysis_stats.json",
    "customers": "matched_customer_yoy_enriched.csv",
    "monthly": "matched_monthly.csv.gz",
    "profiles": "matched_profiles.csv.gz",
    "overall_monthly": "overall_monthly.csv",
    "overall_profiles": "overall_profiles.csv",
    "monthly_change": "monthly_change.csv",
    "cluster8": "cluster_summary_8.csv",
    "cluster_transition8": "cluster_transition_8.csv",
    "tariff_summary": "tariff_summary.csv",
    "tariff_transition": "tariff_transition.csv",
    "monthly_recommendation": "monthly_recommendation.csv",
    "profile_summary": "profile_summary.csv",
    "quality": "matched_quality.csv",
    "daily": "matched_daily.csv.gz",
}

FEATURE_BASE = [
    "연간사용량_kWh", "최대시간사용량_kWh", "주말주중비", "경부하비중",
    "중간부하비중", "최대부하비중", "월변동계수", "부하율",
    "하계민감도", "동계민감도",
]

PLAN_DEFAULTS = {
    "기본형": {"fee": 84_900.0, "included": 450.0},
    "프리미엄형": {"fee": 249_000.0, "included": 1_000.0},
}

# 청구금액 구성의 기본값. 사이드바에서 변경 가능함.
# 일반 주택용(저압)과 제주 TOU에는 연료비조정액·기후환경요금·부가세·전력기금을 별도 반영함.
# 구독 기본형·프리미엄형의 표시가격과 초과단가는 이들 항목을 모두 포함한 최종 소비자가격으로 해석함.
FUEL_ADJUSTMENT_RATE = 5.0      # 원/kWh
CLIMATE_ENV_RATE = 9.0          # 원/kWh
VAT_RATE = 0.10                 # 10%
POWER_FUND_RATE = 0.027         # 2.7%
TOU_CONTRACT_KW = 3.0             # 제주 TOU 계약전력 가정
SUPERUSER_RATE = 736.2             # 원/kWh

SEASON_MONTHS = {
    "봄가을": [3, 4, 5, 9, 10],
    "여름": [6, 7, 8],
    "겨울": [1, 2, 11, 12],
}


def fmt_won(v: float) -> str:
    return f"{float(v):,.0f}원"


def fmt_kwh(v: float) -> str:
    return f"{float(v):,.1f}kWh"


@st.cache_data(show_spinner=False)
def load_data():
    missing = [v for v in DATA_FILES.values() if not (BASE_DIR / v).exists()]
    if missing:
        raise FileNotFoundError("필요한 데이터 파일이 없습니다: " + ", ".join(missing))
    with open(BASE_DIR / DATA_FILES["summary"], encoding="utf-8") as f:
        summary = json.load(f)
    with open(BASE_DIR / DATA_FILES["stats"], encoding="utf-8") as f:
        stats = json.load(f)
    customers = pd.read_csv(BASE_DIR / DATA_FILES["customers"])
    monthly = pd.read_csv(BASE_DIR / DATA_FILES["monthly"], compression="gzip")
    profiles = pd.read_csv(BASE_DIR / DATA_FILES["profiles"], compression="gzip")
    overall_monthly = pd.read_csv(BASE_DIR / DATA_FILES["overall_monthly"])
    overall_profiles = pd.read_csv(BASE_DIR / DATA_FILES["overall_profiles"])
    monthly_change = pd.read_csv(BASE_DIR / DATA_FILES["monthly_change"])
    cluster8 = pd.read_csv(BASE_DIR / DATA_FILES["cluster8"])
    cluster_transition8 = pd.read_csv(BASE_DIR / DATA_FILES["cluster_transition8"])
    tariff_summary = pd.read_csv(BASE_DIR / DATA_FILES["tariff_summary"])
    tariff_transition = pd.read_csv(BASE_DIR / DATA_FILES["tariff_transition"])
    monthly_recommendation = pd.read_csv(BASE_DIR / DATA_FILES["monthly_recommendation"])
    profile_summary = pd.read_csv(BASE_DIR / DATA_FILES["profile_summary"])
    quality = pd.read_csv(BASE_DIR / DATA_FILES["quality"])
    daily = pd.read_csv(BASE_DIR / DATA_FILES["daily"], compression="gzip", parse_dates=["날짜"])
    return {
        "summary": summary,
        "stats": stats,
        "customers": customers,
        "monthly": monthly,
        "profiles": profiles,
        "overall_monthly": overall_monthly,
        "overall_profiles": overall_profiles,
        "monthly_change": monthly_change,
        "cluster8": cluster8,
        "cluster_transition8": cluster_transition8,
        "tariff_summary": tariff_summary,
        "tariff_transition": tariff_transition,
        "monthly_recommendation": monthly_recommendation,
        "profile_summary": profile_summary,
        "quality": quality,
        "daily": daily,
    }


def robust_scale(frame: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    x = frame.astype(float).replace([np.inf, -np.inf], np.nan).copy()
    params = {}
    for col in x.columns:
        s = x[col]
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi).fillna(s.median())
        if col == "연간사용량_kWh":
            s = np.log1p(s.clip(lower=0))
        med = float(s.median())
        scale = float(s.quantile(0.75) - s.quantile(0.25))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(s.std(ddof=0)) or 1.0
        x[col] = (s - med) / scale
        params[col] = (med, scale)
    return x.to_numpy(float), params


def kmeans_numpy(x: np.ndarray, n_clusters: int, seed: int = 42, max_iter: int = 150):
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[int(rng.integers(0, n))].copy()]
    for _ in range(1, n_clusters):
        d2 = np.min(np.stack([np.sum((x - c) ** 2, axis=1) for c in centers], axis=1), axis=1)
        total = float(d2.sum())
        idx = int(rng.choice(n, p=d2 / total)) if total > 0 else int(rng.integers(0, n))
        centers.append(x[idx].copy())
    centers = np.vstack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist = np.stack([np.sum((x - c) ** 2, axis=1) for c in centers], axis=1)
        new_labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for k in range(n_clusters):
            members = x[new_labels == k]
            if len(members):
                new_centers[k] = members.mean(axis=0)
            else:
                new_centers[k] = x[int(np.argmax(np.min(dist, axis=1)))]
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers, atol=1e-7):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers
    return labels, centers


def cluster_names(means: pd.DataFrame) -> Dict[int, str]:
    cols = FEATURE_BASE
    z = means[cols].copy()
    for c in cols:
        sd = float(z[c].std(ddof=0)) or 1.0
        z[c] = (z[c] - float(z[c].mean())) / sd
    traits = {
        "연간사용량_kWh": ("고사용", "저사용"),
        "최대시간사용량_kWh": ("고피크", "저피크"),
        "주말주중비": ("주말집중", "주중집중"),
        "경부하비중": ("야간집중", "주간집중"),
        "중간부하비중": ("중간부하집중", "중간부하낮음"),
        "최대부하비중": ("최대부하집중", "비피크중심"),
        "월변동계수": ("변동", "규칙"),
        "부하율": ("평탄", "첨두"),
        "하계민감도": ("하계민감", "하계둔감"),
        "동계민감도": ("동계민감", "동계둔감"),
    }
    order = means["연간사용량_kWh"].sort_values().index.tolist()
    out = {}
    for rank, idx in enumerate(order, 1):
        row = z.loc[idx]
        ranked = sorted(cols, key=lambda c: abs(float(row[c])), reverse=True)
        selected = []
        for c in ranked:
            v = float(row[c])
            if abs(v) < 0.35:
                continue
            selected.append(traits[c][0] if v >= 0 else traits[c][1])
            if len(selected) == 2:
                break
        if not selected:
            selected = ["표준"]
        out[int(idx)] = f"군집 {rank} · {'·'.join(selected)}형"
    return out


@st.cache_data(show_spinner=False)
def joint_dynamic_clusters(customers: pd.DataFrame, n_clusters: int):
    rows = []
    for year in (2024, 2025):
        part = pd.DataFrame({"고객ID": customers["고객ID"], "연도": year})
        for f in FEATURE_BASE:
            part[f] = customers[f"{year}_{f}"].astype(float)
        rows.append(part)
    stacked = pd.concat(rows, ignore_index=True)
    x, _ = robust_scale(stacked[FEATURE_BASE])
    score = np.max(np.abs(x), axis=1)
    fit = x[score <= np.quantile(score, 0.98)]
    _, centers = kmeans_numpy(fit, n_clusters, seed=42)
    dist = np.stack([np.sum((x - c) ** 2, axis=1) for c in centers], axis=1)
    stacked["cluster_id"] = np.argmin(dist, axis=1)
    means = stacked.groupby("cluster_id")[FEATURE_BASE].mean()
    names = cluster_names(means)
    stacked["군집"] = stacked["cluster_id"].map(names)
    summary = stacked.groupby(["연도", "군집"], as_index=False).agg(
        고객수=("고객ID", "size"),
        연간사용량_kWh=("연간사용량_kWh", "mean"),
        최대시간사용량_kWh=("최대시간사용량_kWh", "mean"),
        주말주중비=("주말주중비", "mean"),
        경부하비중=("경부하비중", "mean"),
        중간부하비중=("중간부하비중", "mean"),
        최대부하비중=("최대부하비중", "mean"),
        월변동계수=("월변동계수", "mean"),
        부하율=("부하율", "mean"),
        하계민감도=("하계민감도", "mean"),
        동계민감도=("동계민감도", "mean"),
    )
    summary["비중"] = summary["고객수"] / len(customers)
    wide = stacked.pivot(index="고객ID", columns="연도", values="군집").reset_index()
    wide.columns = ["고객ID", "2024군집", "2025군집"]
    wide["군집유지여부"] = np.where(wide["2024군집"] == wide["2025군집"], "유지", "이동")
    transition = wide.groupby(["2024군집", "2025군집"], as_index=False).size().rename(columns={"size": "고객수"})
    transition["2024군집내비중"] = transition["고객수"] / transition.groupby("2024군집")["고객수"].transform("sum")
    return stacked, summary, wide, transition


def enrich_scores(customers: pd.DataFrame, cluster_wide: pd.DataFrame) -> pd.DataFrame:
    out = customers.merge(cluster_wide, on="고객ID", how="left", suffixes=("", "_동적"))
    usage_change = (out["연간사용량증감률"].abs() / 0.50).clip(0, 1)
    tou_change = (
        out["경부하비중_증감"].abs() + out["중간부하비중_증감"].abs() + out["최대부하비중_증감"].abs()
    ).div(0.30).clip(0, 1)
    weekend_change = (out["주말주중비_증감"].abs() / 0.50).clip(0, 1)
    load_change = (out["부하율_증감"].abs() / 0.10).clip(0, 1)
    out["패턴안정성점수"] = 100 * (1 - (0.45 * usage_change + 0.25 * tou_change + 0.15 * weekend_change + 0.15 * load_change))
    out["패턴안정성점수"] = out["패턴안정성점수"].clip(0, 100)
    def pct(s): return s.rank(pct=True).fillna(0.5)
    out["수요관리우선점수"] = 100 * (
        0.30 * pct(out["2025_최대시간사용량_kWh"]) +
        0.25 * pct(out["2025_최대부하비중"]) +
        0.20 * pct(out["2025_연간사용량_kWh"]) +
        0.15 * pct(out[["2025_하계민감도", "2025_동계민감도"]].max(axis=1)) +
        0.10 * (1 - pct(out["20일예측_MAPE"]))
    )
    out["구조변화신호"] = np.select(
        [
            out["연간사용량증감률"] >= 0.20,
            out["연간사용량증감률"] <= -0.20,
            out["군집유지여부_동적"].eq("이동") & out["추천요금제유지여부"].eq("변경"),
            out["군집유지여부_동적"].eq("이동"),
            out["추천요금제유지여부"].eq("변경"),
        ],
        ["사용량 20% 이상 증가", "사용량 20% 이상 감소", "군집·요금제 동시변경", "사용패턴 군집 이동", "추천요금제 변경"],
        default="안정",
    )
    return out


def usage_pattern_label(score: float) -> str:
    score = float(score)
    if score >= 80:
        return "매우 일정"
    if score >= 65:
        return "대체로 일정"
    if score >= 45:
        return "변화 있음"
    return "변화 큼"


def peak_management_label(score: float) -> str:
    score = float(score)
    if score >= 75:
        return "매우 높음"
    if score >= 55:
        return "높음"
    if score >= 35:
        return "보통"
    return "낮음"


def profile_for_customer(profiles: pd.DataFrame, cid: str, year: int, season: str, daytype: str) -> pd.DataFrame:
    months = SEASON_MONTHS[season]
    p = profiles[(profiles["고객ID"] == cid) & (profiles["연도"] == year) & (profiles["월"].isin(months)) & (profiles["일유형"] == daytype)]
    return p.groupby("시간", as_index=False)["평균사용량_kWh"].mean()


def aggregate_portfolio_profile(profiles: pd.DataFrame, ids: List[str], year: int, season: str, daytype: str) -> np.ndarray:
    months = SEASON_MONTHS[season]
    p = profiles[(profiles["고객ID"].isin(ids)) & (profiles["연도"] == year) & (profiles["월"].isin(months)) & (profiles["일유형"] == daytype)]
    by_customer = p.groupby(["고객ID", "시간"], as_index=False)["평균사용량_kWh"].mean()
    agg = by_customer.groupby("시간")["평균사용량_kWh"].sum().reindex(range(1, 25), fill_value=0.0)
    return agg.to_numpy(float)


def optimize_transformer_profile(base: np.ndarray, limit_ratio: float, participation: float) -> Dict[str, object]:
    base = np.maximum(np.asarray(base, dtype=float), 0.0)
    n = len(base)
    peak_before = float(base.max()) if n else 0.0
    limit = peak_before * float(limit_ratio)
    overload_before_arr = np.maximum(base - limit, 0.0)
    overload_before = float(overload_before_arr.sum())
    hours_before = int((overload_before_arr > 1e-9).sum())
    if n == 0 or overload_before <= 1e-9 or participation <= 0:
        return {
            "after": base.copy(), "limit": limit, "peak_before": peak_before, "peak_after": peak_before,
            "shifted": 0.0, "reduced": 0.0, "overload_before": overload_before,
            "overload_after": overload_before, "hours_before": hours_before, "hours_after": hours_before,
            "shift_out": np.zeros(n), "shift_in": np.zeros(n), "reduction": np.zeros(n),
            "status": "제어 불필요" if overload_before <= 1e-9 else "직접제어 참여고객 없음",
        }
    scale = 1000
    base_i = np.rint(base * scale).astype(int)
    limit_i = int(round(limit * scale))
    max_shift_fraction = 0.14 * participation
    max_reduce_fraction = 0.06 * participation
    peak_window = set(range(16, 22))
    offpeak_window = set(list(range(22, 24)) + list(range(0, 8)))
    max_shift = np.zeros(n, dtype=int)
    max_reduce = np.zeros(n, dtype=int)
    headroom = np.maximum(limit_i - base_i, 0)
    for h in range(n):
        if base[h] > limit + 1e-9 or h in peak_window:
            factor = 1.0 if base[h] > limit + 1e-9 else 0.5
            max_shift[h] = int(round(base_i[h] * max_shift_fraction * factor))
            max_reduce[h] = int(round(base_i[h] * max_reduce_fraction * factor))
    model = cp_model.CpModel()
    so, rd, si, af, ov = [], [], [], [], []
    for h in range(n):
        a = model.NewIntVar(0, int(max_shift[h]), f"so{h}")
        b = model.NewIntVar(0, int(max_reduce[h]), f"rd{h}")
        c = model.NewIntVar(0, int(headroom[h]), f"si{h}")
        upper = max(int(base_i[h] + headroom[h]), int(base_i[h]), limit_i, 1)
        d = model.NewIntVar(0, upper, f"af{h}")
        e = model.NewIntVar(0, max(upper, int(base_i[h]), 1), f"ov{h}")
        model.Add(d == int(base_i[h]) - a - b + c)
        model.Add(e >= d - limit_i)
        so.append(a); rd.append(b); si.append(c); af.append(d); ov.append(e)
    model.Add(sum(so) == sum(si))
    objective = 100_000 * sum(ov) + 40 * sum(rd) + 8 * sum(so)
    objective += 4 * sum(si[h] for h in range(n) if h not in offpeak_window)
    model.Minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    solver.parameters.num_search_workers = 8
    code = solver.Solve(model)
    if code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"after": base.copy(), "limit": limit, "peak_before": peak_before, "peak_after": peak_before,
                "shifted": 0.0, "reduced": 0.0, "overload_before": overload_before,
                "overload_after": overload_before, "hours_before": hours_before, "hours_after": hours_before,
                "shift_out": np.zeros(n), "shift_in": np.zeros(n), "reduction": np.zeros(n), "status": "최적화 실패"}
    shift_out = np.array([solver.Value(v) for v in so]) / scale
    reduction = np.array([solver.Value(v) for v in rd]) / scale
    shift_in = np.array([solver.Value(v) for v in si]) / scale
    after = np.array([solver.Value(v) for v in af]) / scale
    oa = np.maximum(after - limit, 0.0)
    status = "운전한도 충족" if not np.any(oa > 1e-6) else ("과부하 완화·잔여 초과 존재" if oa.sum() < overload_before else "유연성 부족")
    return {
        "after": after, "limit": limit, "peak_before": peak_before, "peak_after": float(after.max()),
        "shifted": float(shift_out.sum()), "reduced": float(reduction.sum()),
        "overload_before": overload_before, "overload_after": float(oa.sum()),
        "hours_before": hours_before, "hours_after": int((oa > 1e-6).sum()),
        "shift_out": shift_out, "shift_in": shift_in, "reduction": reduction, "status": status,
    }


def zip_results(files: Dict[str, bytes]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return bio.getvalue()




ACTION_LIBRARY: List[Dict[str, object]] = [
    {"id":"standby","name":"취침·외출 시 대기전력 일괄 차단","ownership":"대기전력차단","low":0.12,"high":0.30,"daily_max":1,"discomfort":1,"reliability":0.82,"kind":"reduce"},
    {"id":"hvac_set","name":"냉난방 설정온도 1℃ 완화·외출 절전","ownership":"냉난방기","low":0.45,"high":1.20,"daily_max":1,"discomfort":4,"reliability":0.72,"kind":"reduce","seasons":["여름","겨울"]},
    {"id":"hvac_hour","name":"냉난방 운전시간 1시간 단축","ownership":"냉난방기","low":0.55,"high":1.35,"daily_max":2,"discomfort":7,"reliability":0.68,"kind":"reduce","seasons":["여름","겨울"]},
    {"id":"dryer","name":"건조기 1회 자연건조로 대체","ownership":"건조기","low":1.40,"high":2.80,"weekly_max":3,"discomfort":5,"reliability":0.88,"kind":"reduce"},
    {"id":"dishwasher","name":"식기세척기 절전모드·모아서 사용","ownership":"식기세척기","low":0.22,"high":0.55,"weekly_max":6,"discomfort":2,"reliability":0.76,"kind":"reduce"},
    {"id":"laundry","name":"세탁기 냉수·절전코스 사용","ownership":"세탁기","low":0.10,"high":0.28,"weekly_max":6,"discomfort":1,"reliability":0.74,"kind":"reduce"},
    {"id":"game_tv","name":"게임·TV 이용시간 2시간 단축","ownership":"게임TV","low":0.25,"high":0.70,"daily_max":1,"discomfort":4,"reliability":0.78,"kind":"reduce"},
    {"id":"aircare","name":"공기청정기·제습기 절전운전","ownership":"공기관리기기","low":0.16,"high":0.55,"daily_max":1,"discomfort":2,"reliability":0.67,"kind":"reduce"},
    {"id":"shift_laundry","name":"세탁·건조를 22시 이후로 이동","ownership":"세탁기","low":1.0,"high":2.2,"daily_max":1,"discomfort":2,"reliability":0.88,"kind":"shift"},
    {"id":"shift_dish","name":"식기세척기를 취침 후 예약운전","ownership":"식기세척기","low":0.5,"high":1.0,"daily_max":1,"discomfort":1,"reliability":0.90,"kind":"shift"},
]
CONTROL_MODES = {
    "편의 우선": {"target_factor":0.80,"delivery":0.92,"discomfort_weight":140},
    "균형": {"target_factor":1.00,"delivery":0.96,"discomfort_weight":80},
    "목표달성 우선": {"target_factor":1.15,"delivery":0.98,"discomfort_weight":40},
}


def season_for_month(month: int) -> str:
    if month in (6, 7, 8): return "여름"
    if month in (1, 2, 11, 12): return "겨울"
    return "봄가을"


def round_half_up(value: float) -> int:
    """양수의 4사5입 반올림."""
    return int(math.floor(float(value) + 0.5))


def truncate_won(value: float) -> int:
    """원 미만 절사."""
    return int(math.floor(max(float(value), 0.0) + 1e-9))


def truncate_10won(value: float) -> int:
    """청구금액의 10원 미만 절사."""
    return int(math.floor(max(float(value), 0.0) / 10.0 + 1e-12) * 10)


def billed_kwh(kwh: float) -> int:
    """전기요금 계산단위 1kWh에 맞춰 사용량을 반올림."""
    return max(round_half_up(max(float(kwh), 0.0)), 0)


def allocate_integer_kwh(total_kwh: int, shares: List[float]) -> List[int]:
    """시간대 비중을 정수 kWh로 배분하면서 합계가 정확히 일치하도록 함."""
    total=max(int(total_kwh),0)
    arr=np.asarray(shares,dtype=float)
    arr=np.where(np.isfinite(arr)&(arr>0),arr,0.0)
    if arr.sum()<=0:
        arr=np.ones(len(arr),dtype=float)/max(len(arr),1)
    else:
        arr=arr/arr.sum()
    raw=arr*total
    base=np.floor(raw).astype(int)
    remain=total-int(base.sum())
    if remain>0:
        order=np.argsort(-(raw-base))
        for idx in order[:remain]: base[idx]+=1
    return base.tolist()


def finalize_electric_bill(base_fee: float, energy_charge: float, kwh: float) -> float:
    """한전 계산순서에 따라 일반·TOU 최종 청구액을 계산함.

    기본요금·전력량요금·기후환경요금·연료비조정액은 원 미만 절사,
    부가가치세는 원 미만 반올림, 전력산업기반기금 및 최종 청구액은
    10원 미만 절사함.
    """
    usage=billed_kwh(kwh)
    basic=truncate_won(base_fee)
    energy=truncate_won(energy_charge)
    fuel=truncate_won(usage*float(FUEL_ADJUSTMENT_RATE))
    climate=truncate_won(usage*float(CLIMATE_ENV_RATE))
    electricity_charge=basic+energy+fuel+climate
    vat=round_half_up(electricity_charge*float(VAT_RATE))
    fund=truncate_10won(electricity_charge*float(POWER_FUND_RATE))
    return float(truncate_10won(electricity_charge+vat+fund))


def bill_component_breakdown(base_fee: float, energy_charge: float, kwh: float) -> Dict[str, float]:
    """화면 검증용 요금 구성요소."""
    usage=billed_kwh(kwh)
    basic=truncate_won(base_fee)
    energy=truncate_won(energy_charge)
    fuel=truncate_won(usage*float(FUEL_ADJUSTMENT_RATE))
    climate=truncate_won(usage*float(CLIMATE_ENV_RATE))
    electricity_charge=basic+energy+fuel+climate
    vat=round_half_up(electricity_charge*float(VAT_RATE))
    fund=truncate_10won(electricity_charge*float(POWER_FUND_RATE))
    total=truncate_10won(electricity_charge+vat+fund)
    return {
        "요금계산 사용량(kWh)":float(usage),
        "기본요금(원)":float(basic),
        "전력량요금(원)":float(energy),
        "연료비조정액(원)":float(fuel),
        "기후환경요금(원)":float(climate),
        "전기요금계(원)":float(electricity_charge),
        "부가가치세(원)":float(vat),
        "전력산업기반기금(원)":float(fund),
        "최종 청구액(원)":float(total),
    }


def residential_base_energy(kwh: float, month: int) -> Tuple[float,float,int]:
    """주택용전력 저압의 기본요금·전력량요금.

    저압 단가: 120.0 / 214.6 / 307.3원/kWh.
    하계(7~8월)는 300·450kWh, 기타계절은 200·400kWh 구간을 사용함.
    하계 및 동계(12~2월) 1,000kWh 초과분에는 736.2원/kWh를 적용함.
    """
    u=billed_kwh(kwh)
    summer=month in (7,8)
    if summer:
        t1,t2=300,450
    else:
        t1,t2=200,400
    basic=910 if u<=t1 else (1600 if u<=t2 else 7300)
    first=min(u,t1)
    second=min(max(u-t1,0),t2-t1)
    third=max(u-t2,0)
    super_month=month in (7,8,12,1,2)
    if super_month and u>1000:
        normal_third=max(1000-t2,0)
        excess=u-1000
        energy=first*120.0+second*214.6+normal_third*307.3+excess*SUPERUSER_RATE
    else:
        energy=first*120.0+second*214.6+third*307.3
    return float(basic),float(energy),u


def residential_bill(kwh: float, month: int) -> float:
    basic,energy,u=residential_base_energy(kwh,month)
    return finalize_electric_bill(basic,energy,u)


def tou_base_energy(kwh: float, month: int, off_share: float, mid_share: float, peak_share: float, contract_kw: float|None=None) -> Tuple[float,float,int,List[int]]:
    """제주 주택용 계시별 요금의 기본요금·전력량요금."""
    u=billed_kwh(kwh)
    ck=float(TOU_CONTRACT_KW if contract_kw is None else contract_kw)
    rates=(125.8,153.8,172.4) if month in (3,4,5,9,10) else (138.7,184.7,220.5)
    buckets=allocate_integer_kwh(u,[off_share,mid_share,peak_share])
    super_month=month in (6,7,8,11,12,1,2)
    if super_month and u>1000:
        first_buckets=allocate_integer_kwh(1000,[off_share,mid_share,peak_share])
        energy=sum(v*r for v,r in zip(first_buckets,rates))+(u-1000)*SUPERUSER_RATE
    else:
        energy=sum(v*r for v,r in zip(buckets,rates))
    return 4310.0*ck,float(energy),u,buckets


def tou_bill(kwh: float, month: int, off_share: float, mid_share: float, peak_share: float, contract_kw: float|None=None) -> float:
    basic,energy,u,_=tou_base_energy(kwh,month,off_share,mid_share,peak_share,contract_kw)
    return finalize_electric_bill(basic,energy,u)

def subscription_bill(kwh: float, fee: float, included: float, overage: float) -> float:
    """구독료와 초과단가를 모든 부가요금·세금이 포함된 최종 소비자가격으로 계산함.

    제공량 이내에서는 고객 청구액이 월 구독료로 고정되고, 제공량을 초과한 사용량에만
    최종 초과단가를 곱하여 더함. 따라서 연료비조정액·기후환경요금·부가세·전력기금을
    구독요금에 다시 가산하지 않음.
    """
    usage = max(float(kwh), 0.0)
    final_fee = max(float(fee), 0.0)
    excess = max(usage - max(float(included), 0.0), 0.0)
    final_overage = max(float(overage), 0.0)
    return float(round_half_up(final_fee + excess * final_overage))


def inverse_subscription_bill(target_bill: float, fee: float, included: float, overage: float) -> float:
    """최종 납부목표 아래에서 사용할 수 있는 최대 사용량을 역산함."""
    target = float(target_bill)
    final_fee = max(float(fee), 0.0)
    allowance = max(float(included), 0.0)
    final_overage = max(float(overage), 0.0)
    if target < final_fee:
        return 0.0
    if final_overage <= 1e-12:
        return allowance
    return float(allowance + (target - final_fee) / final_overage)

def fmt_won1(v: float) -> str:
    return f"{float(v):,.0f}원"


def fmt_pct(v: float) -> str:
    return f"{float(v)*100:.1f}%"


def is_money_column(name: str) -> bool:
    name = str(name)
    return any(token in name for token in ["(원)", "요금", "납부액", "절감액", "금액"])


def round_table(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    for c in out.select_dtypes(include=[np.number]).columns:
        out[c]=out[c].round(0 if is_money_column(c) else 1)
    return out


def dataframe_config(df: pd.DataFrame, percent_cols: List[str] | None=None) -> Dict[str, object]:
    percent_cols=set(percent_cols or [])
    cfg={}
    for c in df.columns:
        name=str(c)
        if c in percent_cols or "(%)" in name or "%p" in name:
            cfg[c]=st.column_config.NumberColumn(format="%.1f%%")
        elif is_money_column(name):
            cfg[c]=st.column_config.NumberColumn(format="₩%,.0f")
        elif "(명)" in name or "고객수" in name or "고객 수" in name:
            cfg[c]=st.column_config.NumberColumn(format="%,.0f")
        elif pd.api.types.is_numeric_dtype(df[c]):
            cfg[c]=st.column_config.NumberColumn(format="%,.1f")
        else:
            wide_tokens=("방식","행동","권고","판정","설명","추천요금제","현재요금제","최근 변화","군집 유형")
            cfg[c]=st.column_config.TextColumn(width="large" if any(t in name for t in wide_tokens) else "medium")
    return cfg


def display_full_text_table(df: pd.DataFrame) -> None:
    """고객별 진단·제어 탭의 긴 문구를 말줄임표 없이 줄바꿈해 표시함."""
    show=df.copy()
    for col in show.columns:
        name=str(col)
        if pd.api.types.is_numeric_dtype(show[col]):
            if "(%)" in name or "%p" in name:
                show[col]=show[col].map(lambda v: "" if pd.isna(v) else f"{float(v):,.1f}%")
            elif is_money_column(name):
                show[col]=show[col].map(lambda v: "" if pd.isna(v) else f"{float(v):,.0f}")
            elif "(명)" in name or "고객수" in name or "고객 수" in name or name in ("연도","월","일","시간"):
                show[col]=show[col].map(lambda v: "" if pd.isna(v) else f"{float(v):,.0f}")
            else:
                show[col]=show[col].map(lambda v: "" if pd.isna(v) else f"{float(v):,.1f}")
    html=show.to_html(index=False,escape=True,border=0)
    st.markdown(f'<div class="full-text-table-wrap">{html}</div>',unsafe_allow_html=True)


def forecast_month_longitudinal(customer_daily: pd.DataFrame, year: int, month: int, cutoff_day: int) -> Dict[str,float]:
    dm=customer_daily[(customer_daily["연도"]==year)&(customer_daily["월"]==month)].sort_values("일")
    observed=dm[dm["일"]<=cutoff_day]
    remaining=dm[dm["일"]>cutoff_day]
    actual=float(dm["일사용량_kWh"].sum())
    current=float(observed["일사용량_kWh"].sum())
    current_means=observed.groupby("일유형")["일사용량_kWh"].mean().to_dict()
    current_overall=float(observed["일사용량_kWh"].mean()) if len(observed) else 0.0
    prev=customer_daily[(customer_daily["연도"]==year-1)&(customer_daily["월"]==month)]
    prev_means=prev.groupby("일유형")["일사용량_kWh"].mean().to_dict()
    prev_overall=float(prev["일사용량_kWh"].mean()) if len(prev) else current_overall
    alpha=min(0.85,max(0.55,len(observed)/max(cutoff_day,1))) if len(prev) else 1.0
    pred_remaining=0.0
    for _,r in remaining.iterrows():
        dt=r["일유형"]
        cur=float(current_means.get(dt,current_overall))
        prv=float(prev_means.get(dt,prev_overall))
        pred_remaining+=alpha*cur+(1-alpha)*prv
    forecast=current+pred_remaining
    std_parts=[]
    if len(observed)>1: std_parts.append(float(observed["일사용량_kWh"].std(ddof=0)))
    if len(prev)>1: std_parts.append(float(prev["일사용량_kWh"].std(ddof=0)))
    daily_std=float(np.mean(std_parts)) if std_parts else 0.0
    uncertainty=1.28*daily_std*math.sqrt(max(len(remaining),1))
    return {"current":current,"forecast":forecast,"lower":max(current,forecast-uncertainty),"upper":forecast+uncertainty,
            "actual":actual,"remaining_days":len(remaining),"observed_days":len(observed),"days_in_month":len(dm)}


def alert_level(current: float, forecast: float, included: float) -> str:
    used=current/max(included,1e-9); projected=forecast/max(included,1e-9)
    if current>=included or used>=0.95 or projected>=1.25: return "긴급"
    if used>=0.85 or projected>=1.10: return "경고"
    if used>=0.70 or projected>1.00: return "주의"
    if used>=0.50 or projected>=0.90: return "관심"
    return "정상"


def monthly_bill_map(usage: float, month: int, monthly_row: pd.Series, basic_fee: float, basic_inc: float,
                     premium_fee: float, premium_inc: float, overage: float) -> Dict[str,float]:
    return {
        "일반 주택용(저압)": residential_bill(usage,month),
        "제주 TOU": tou_bill(usage,month,float(monthly_row["경부하비중"]),float(monthly_row["중간부하비중"]),float(monthly_row["최대부하비중"]),TOU_CONTRACT_KW),
        "구독 기본형": subscription_bill(usage,basic_fee,basic_inc,overage),
        "구독 프리미엄형": subscription_bill(usage,premium_fee,premium_inc,overage),
    }


def annual_bill_map(customer_monthly: pd.DataFrame, basic_fee: float, basic_inc: float, premium_fee: float,
                    premium_inc: float, overage: float) -> Dict[str,float]:
    totals={"일반 주택용(저압)":0.0,"제주 TOU":0.0,"구독 기본형":0.0,"구독 프리미엄형":0.0}
    for _,r in customer_monthly.sort_values("월").iterrows():
        b=monthly_bill_map(float(r["사용량_kWh"]),int(r["월"]),r,basic_fee,basic_inc,premium_fee,premium_inc,overage)
        for k,v in b.items(): totals[k]+=v
    return totals



PLAN_ORDER = ["일반 주택용(저압)", "제주 TOU", "구독 기본형", "구독 프리미엄형"]
PLAN_BILL_COLUMNS = {
    "일반 주택용(저압)": "일반주택용(원)",
    "제주 TOU": "제주TOU(원)",
    "구독 기본형": "기본형(원)",
    "구독 프리미엄형": "프리미엄형(원)",
}


@st.cache_data(show_spinner=False)
def dynamic_tariff_analysis(monthly: pd.DataFrame, basic_fee: float, basic_inc: float,
                            premium_fee: float, premium_inc: float, overage: float,
                            fuel_rate: float, climate_rate: float, vat_rate: float, fund_rate: float, tou_contract_kw: float) -> Dict[str, pd.DataFrame | float]:
    """현재 화면의 요금 설정으로 월별·연간 추천요금제를 모두 다시 계산합니다."""
    monthly_rows=[]
    for _,r in monthly.iterrows():
        bills=monthly_bill_map(float(r["사용량_kWh"]),int(r["월"]),r,basic_fee,basic_inc,premium_fee,premium_inc,overage)
        rec=min(PLAN_ORDER,key=lambda p:bills[p])
        monthly_rows.append({
            "고객ID":r["고객ID"],"연도":int(r["연도"]),"월":int(r["월"]),"사용량(kWh)":float(r["사용량_kWh"]),
            "일반주택용(원)":bills["일반 주택용(저압)"],"제주TOU(원)":bills["제주 TOU"],
            "기본형(원)":bills["구독 기본형"],"프리미엄형(원)":bills["구독 프리미엄형"],
            "월별추천요금제":rec,"월별최저요금(원)":bills[rec],
        })
    monthly_rec=pd.DataFrame(monthly_rows)
    bill_cols=list(PLAN_BILL_COLUMNS.values())
    annual=monthly_rec.groupby(["고객ID","연도"],as_index=False)[bill_cols].sum()
    annual["연간사용량(kWh)"]=monthly_rec.groupby(["고객ID","연도"])["사용량(kWh)"].sum().to_numpy()
    annual["월평균사용량(kWh)"]=annual["연간사용량(kWh)"]/12.0
    annual["연간추천요금제"]=annual[bill_cols].idxmin(axis=1).map({v:k for k,v in PLAN_BILL_COLUMNS.items()})
    annual["연간최저요금(원)"]=annual[bill_cols].min(axis=1)
    annual["연간TOU대비절감(원)"]=(annual["제주TOU(원)"]-annual["연간최저요금(원)"]).clip(lower=0)
    monthly_rec=monthly_rec.merge(annual[["고객ID","연도","연간추천요금제"]],on=["고객ID","연도"],how="left")
    monthly_rec["월·연간추천일치"]=np.where(monthly_rec["월별추천요금제"]==monthly_rec["연간추천요금제"],"일치","상이")

    summary_rows=[]
    for year,g in annual.groupby("연도"):
        for plan in PLAN_ORDER:
            col=PLAN_BILL_COLUMNS[plan]
            vals=g[col].astype(float)
            count=int((g["연간추천요금제"]==plan).sum())
            summary_rows.append({
                "연도":str(int(year)),"요금제":plan,
                "고객당평균연간요금(원)":float(vals.mean()),
                "고객당중앙연간요금(원)":float(vals.median()),
                "고객당평균월요금(원)":float((vals/12.0).mean()),
                "고객당중앙월요금(원)":float((vals/12.0).median()),
                "연간추천고객수":count,"연간추천비중(%)":count/max(len(g),1)*100,
            })
    annual_summary=pd.DataFrame(summary_rows)

    annual_wide=annual.pivot(index="고객ID",columns="연도",values="연간추천요금제")
    annual_stability=float((annual_wide.get(2024)==annual_wide.get(2025)).mean()) if {2024,2025}.issubset(annual_wide.columns) else float("nan")
    if {2024,2025}.issubset(annual_wide.columns):
        transition=(annual_wide.groupby([2024,2025]).size().rename("고객수").reset_index()
                    .rename(columns={2024:"2024 연간추천",2025:"2025 연간추천"}))
        denom=transition.groupby("2024 연간추천")["고객수"].transform("sum")
        transition["2024 추천군 내 비중(%)"]=transition["고객수"]/denom*100
    else:
        transition=pd.DataFrame(columns=["2024 연간추천","2025 연간추천","고객수","2024 추천군 내 비중(%)"])

    all_index=pd.MultiIndex.from_product([sorted(monthly_rec["연도"].unique()),range(1,13),PLAN_ORDER],names=["연도","월","요금제"])
    msum=(monthly_rec.groupby(["연도","월","월별추천요금제"]).size().reindex(all_index,fill_value=0)
          .rename("월별추천고객수").reset_index().rename(columns={"월별추천요금제":"요금제"}))
    msum["월별추천비중(%)"]=msum["월별추천고객수"]/monthly_rec["고객ID"].nunique()*100
    monthly_wide=monthly_rec.pivot(index=["고객ID","월"],columns="연도",values="월별추천요금제")
    monthly_stability=float((monthly_wide.get(2024)==monthly_wide.get(2025)).mean()) if {2024,2025}.issubset(monthly_wide.columns) else float("nan")
    return {
        "monthly_customer":monthly_rec,"monthly_summary":msum,
        "annual_customer":annual,"annual_summary":annual_summary,
        "annual_transition":transition,"annual_stability":annual_stability,
        "monthly_stability":monthly_stability,
    }


def bill_for_plan(plan: str, usage: float, month: int, monthly_row: pd.Series, basic_fee: float, basic_inc: float,
                  premium_fee: float, premium_inc: float, overage: float) -> float:
    bills = monthly_bill_map(usage, month, monthly_row, basic_fee, basic_inc, premium_fee, premium_inc, overage)
    return float(bills[plan])


def inverse_bill_for_plan(target_bill: float, plan: str, month: int, monthly_row: pd.Series, basic_fee: float,
                          basic_inc: float, premium_fee: float, premium_inc: float, overage: float) -> float:
    target_bill = max(float(target_bill), 0.0)
    # 연료비·기후환경요금과 세금·기금까지 포함한 최종 청구액을 기준으로 모든 요금제를 수치적으로 역산함.
    lo, hi = 0.0, 2_000.0
    while bill_for_plan(plan, hi, month, monthly_row, basic_fee, basic_inc, premium_fee, premium_inc, overage) < target_bill and hi < 50_000:
        hi *= 2
    for _ in range(70):
        mid = (lo + hi) / 2
        if bill_for_plan(plan, mid, month, monthly_row, basic_fee, basic_inc, premium_fee, premium_inc, overage) <= target_bill:
            lo = mid
        else:
            hi = mid
    return float(lo)


def tariff_comparison_table(bills: Dict[str, float], current_plan: str) -> pd.DataFrame:
    cheapest = min(bills, key=bills.get)
    current_bill = float(bills[current_plan])
    minimum_bill = float(bills[cheapest])
    rows=[]
    for plan, value in bills.items():
        rows.append({
            "요금제": plan,
            "월말 예상요금(원)": value,
            "현재요금제 대비 차이(원)": value-current_bill,
            "최저요금 대비 차이(원)": value-minimum_bill,
            "판정": "현재 적용" if plan==current_plan else ("추천" if plan==cheapest else "비교"),
        })
    return pd.DataFrame(rows).sort_values("월말 예상요금(원)")


def build_tariff_monitor(daily: pd.DataFrame, monthly: pd.DataFrame, customers: pd.DataFrame, cluster_col: str,
                         period: str, year: int, month: int, cutoff: int, current_plan: str,
                         basic_fee: float, basic_inc: float, premium_fee: float, premium_inc: float, overage: float) -> pd.DataFrame:
    cust_lookup=customers.set_index("고객ID")
    rows=[]
    if period=="연간 전체":
        mm=monthly[monthly["연도"]==year]
        for cid,g in mm.groupby("고객ID",sort=False):
            if cid not in cust_lookup.index: continue
            bills=annual_bill_map(g,basic_fee,basic_inc,premium_fee,premium_inc,overage)
            usage=float(g["사용량_kWh"].sum()); rec=min(bills,key=bills.get)
            rows.append({"고객ID":cid,"군집":cust_lookup.loc[cid,cluster_col],"연간사용량(kWh)":usage,"월평균사용량(kWh)":usage/12,
                         "일반주택용(원)":bills["일반 주택용(저압)"],"제주TOU(원)":bills["제주 TOU"],"기본형(원)":bills["구독 기본형"],
                         "프리미엄형(원)":bills["구독 프리미엄형"],"추천요금제":rec,"TOU대비절감(원)":max(bills["제주 TOU"]-bills[rec],0),
                         "기본형제공량사용률(%)":usage/max(basic_inc*12,1e-9)*100,"프리미엄형제공량사용률(%)":usage/max(premium_inc*12,1e-9)*100,
                         "2024→2025증감률(%)":float(cust_lookup.loc[cid,"연간사용량증감률"])*100,"패턴안정성점수":float(cust_lookup.loc[cid,"패턴안정성점수"]),
                         "수요관리우선점수":float(cust_lookup.loc[cid,"수요관리우선점수"])})
    else:
        dd=daily[(daily["연도"]==year)&(daily["월"]==month)]
        mm=monthly[(monthly["연도"]==year)&(monthly["월"]==month)].set_index("고객ID")
        daily_groups={cid:g for cid,g in daily.groupby("고객ID",sort=False)}
        for cid,g in dd.groupby("고객ID",sort=False):
            if cid not in cust_lookup.index or cid not in mm.index: continue
            f=forecast_month_longitudinal(daily_groups[cid],year,month,cutoff)
            bills=monthly_bill_map(f["forecast"],month,mm.loc[cid],basic_fee,basic_inc,premium_fee,premium_inc,overage)
            rec=min(bills,key=bills.get); inc=basic_inc if current_plan=="기본형" else premium_inc
            rows.append({"고객ID":cid,"군집":cust_lookup.loc[cid,cluster_col],"현재누적(kWh)":f["current"],"남은정액량(kWh)":max(inc-f["current"],0),
                         "월말예상(kWh)":f["forecast"],"예측하한(kWh)":f["lower"],"예측상한(kWh)":f["upper"],"실제월사용량(kWh)":f["actual"],
                         "일반주택용(원)":bills["일반 주택용(저압)"],"제주TOU(원)":bills["제주 TOU"],"기본형(원)":bills["구독 기본형"],
                         "프리미엄형(원)":bills["구독 프리미엄형"],"추천요금제":rec,"TOU대비절감(원)":max(bills["제주 TOU"]-bills[rec],0),
                         "기본형제공량사용률(%)":f["forecast"]/max(basic_inc,1e-9)*100,"프리미엄형제공량사용률(%)":f["forecast"]/max(premium_inc,1e-9)*100,
                         "알림단계":alert_level(f["current"],f["forecast"],inc),"예측오차(%)":abs(f["forecast"]-f["actual"])/max(f["actual"],1e-9)*100,
                         "패턴안정성점수":float(cust_lookup.loc[cid,"패턴안정성점수"]),"수요관리우선점수":float(cust_lookup.loc[cid,"수요관리우선점수"])})
    return pd.DataFrame(rows)


def optimize_actions(required_kwh: float, remaining_days: int, season: str, ownership: List[str], mode: str, direct: bool=False) -> Tuple[pd.DataFrame,float,float]:
    if remaining_days<=0:
        return pd.DataFrame(columns=["대안","유형","실행횟수","예상절감·이동량(kWh)","실효량(kWh)","불편점수"]),0.0,0.0
    reductions=[]; shifts=[]
    for a in ACTION_LIBRARY:
        if a["ownership"] not in ownership: continue
        if "seasons" in a and season not in a["seasons"]: continue
        mx=int(a.get("daily_max",0)*remaining_days or a.get("weekly_max",0)*math.ceil(remaining_days/7))
        if mx<=0: continue
        avg=(float(a["low"])+float(a["high"]))/2; delivery=CONTROL_MODES[mode]["delivery"] if direct else float(a["reliability"])
        (shifts if a["kind"]=="shift" else reductions).append((a,mx,avg,delivery))
    rows=[]; gross=0.0; effective=0.0
    if required_kwh>0 and reductions:
        model=cp_model.CpModel(); vars=[]; scale=1000
        for i,(a,mx,avg,delivery) in enumerate(reductions): vars.append(model.NewIntVar(0,mx,f"act{i}"))
        delivered=[int(round(avg*delivery*scale)) for a,mx,avg,delivery in reductions]
        model.Add(sum(v*d for v,d in zip(vars,delivered))>=int(math.ceil(required_kwh*CONTROL_MODES[mode]["target_factor"]*scale)))
        model.Minimize(sum(v*(int(reductions[i][0]["discomfort"])*CONTROL_MODES[mode]["discomfort_weight"]+5) for i,v in enumerate(vars)))
        solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=2.0;status=solver.Solve(model)
        counts=[solver.Value(v) for v in vars] if status in (cp_model.OPTIMAL,cp_model.FEASIBLE) else [mx for a,mx,avg,delivery in reductions]
        for count,(a,mx,avg,delivery) in zip(counts,reductions):
            if count<=0: continue
            g=count*avg;e=g*delivery;gross+=g;effective+=e
            rows.append({"대안":a["name"],"유형":"사용량감축","실행횟수":count,"예상절감·이동량(kWh)":g,"실효량(kWh)":e,"불편점수":count*int(a["discomfort"])})
    shift_ratio={"편의 우선":0.15,"균형":0.35,"목표달성 우선":0.55}[mode]*(1.0 if direct else 0.55)
    for a,mx,avg,delivery in shifts:
        count=int(round(mx*shift_ratio))
        if count<=0: continue
        g=count*avg;e=g*delivery
        rows.append({"대안":a["name"],"유형":"시간이동","실행횟수":count,"예상절감·이동량(kWh)":g,"실효량(kWh)":e,"불편점수":count*int(a["discomfort"])})
    return pd.DataFrame(rows),gross,effective


def controlled_profile(base: np.ndarray, action_plan: pd.DataFrame, remaining_days: int) -> np.ndarray:
    p=np.asarray(base,dtype=float).copy()
    if action_plan is None or action_plan.empty or remaining_days<=0:return p
    reduce_total=float(action_plan.loc[action_plan["유형"]=="사용량감축","실효량(kWh)"].sum())/remaining_days
    shift_total=float(action_plan.loc[action_plan["유형"]=="시간이동","실효량(kWh)"].sum())/remaining_days
    rh=np.arange(14,24);w=p[rh];w=w/w.sum() if w.sum()>0 else np.ones(len(w))/len(w)
    for h,ww in zip(rh,w):p[h]=max(p[h]-reduce_total*ww,base[h]*0.35)
    ph=np.arange(16,22);oh=np.array(list(range(22,24))+list(range(0,8)));pw=p[ph];pw=pw/pw.sum() if pw.sum()>0 else np.ones(len(pw))/len(pw)
    removed=0.0
    for h,ww in zip(ph,pw):
        x=min(shift_total*ww,p[h]*0.45);p[h]-=x;removed+=x
    cap=float(np.max(base));res=removed
    for h in oh[np.argsort(p[oh])]:
        add=min(max(cap-p[h],0),res);p[h]+=add;res-=add
        if res<=1e-9:break
    return np.maximum(p,0)


def cumulative_projection(dm: pd.DataFrame, cutoff: int, forecast_total: float, advisory_reduction: float, direct_reduction: float) -> pd.DataFrame:
    dm=dm.sort_values("일").copy();obs=dm[dm["일"]<=cutoff];rem=dm[dm["일"]>cutoff]
    current=float(obs["일사용량_kWh"].sum());base_remaining=max(forecast_total-current,0)
    pattern=rem["일사용량_kWh"].to_numpy(float)
    pattern=pattern/pattern.sum() if len(pattern) and pattern.sum()>0 else (np.ones(len(rem))/len(rem) if len(rem) else np.array([]))
    rows=[];actual_cum=obs["일사용량_kWh"].cumsum().to_numpy(float)
    for i,(_,r) in enumerate(obs.iterrows()):rows.append({"일":int(r["일"]),"실제누적":actual_cum[i],"미제어예상":np.nan,"행동권고예상":np.nan,"직접제어예상":np.nan})
    bc=ac=dc=current
    for i,(_,r) in enumerate(rem.iterrows()):
        bd=base_remaining*(pattern[i] if len(pattern) else 0);ad=max(bd-advisory_reduction/max(len(rem),1),0);dd=max(bd-direct_reduction/max(len(rem),1),0)
        bc+=bd;ac+=ad;dc+=dd;rows.append({"일":int(r["일"]),"실제누적":np.nan,"미제어예상":bc,"행동권고예상":ac,"직접제어예상":dc})
    return pd.DataFrame(rows)


st.set_page_config(page_title="제주 TOU 고객 요금 분석·추천 및 사용량 제어 시뮬레이터", page_icon="⚡", layout="wide")
st.markdown("""
<style>
div[data-testid="stMetric"] { min-width: 0 !important; }
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"] {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    overflow-wrap: anywhere !important;
    line-height: 1.15 !important;
}
div[data-testid="stMetricLabel"] p { font-size: 0.76rem !important; }
div[data-testid="stMetricValue"] { font-size: 1.28rem !important; }
div[data-testid="stMetricDelta"] { font-size: 0.74rem !important; }
[data-testid="stAlert"] p,
[data-testid="stMarkdownContainer"] p,
button[data-baseweb="tab"] p {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    overflow-wrap: anywhere !important;
}
button[data-baseweb="tab"] p { font-size: 0.82rem !important; line-height: 1.15 !important; }
div[data-baseweb="select"] span { font-size: 0.84rem !important; }
.full-text-table-wrap { width: 100%; overflow-x: auto; margin: 0.25rem 0 0.75rem 0; }
.full-text-table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.full-text-table-wrap th,
.full-text-table-wrap td {
    border: 1px solid rgba(128,128,128,0.28);
    padding: 0.38rem 0.48rem;
    text-align: left;
    vertical-align: top;
    white-space: normal !important;
    word-break: keep-all;
    overflow-wrap: anywhere;
}
.full-text-table-wrap th { font-size: 0.76rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)
st.title("제주 TOU 고객 요금 분석·추천 및 사용량 제어 시뮬레이터")
st.caption(f"앱 버전 {APP_VERSION} · 2024~2025년 공통 핵심고객 712명 · 일반 저압·제주 TOU 공식단가 및 부가요금·세금 반영, 구독료는 모두 포함된 최종가격 · 요금은 원 단위 반올림")

try:
    D=load_data()
except Exception as exc:
    st.error(str(exc));st.stop()

with st.sidebar:
    st.header("분석 설정")
    cluster_count=st.slider("공통 군집 수",3,8,8,1)
    st.divider();st.header("요금 가정")
    basic_fee=st.number_input("기본형 월 구독료(최종 납부액, 원)",0,500_000,84_900,1_000)
    basic_inc=st.number_input("기본형 제공량(kWh)",0,3_000,450,10)
    premium_fee=st.number_input("프리미엄형 월 구독료(최종 납부액, 원)",0,800_000,249_000,1_000)
    premium_inc=st.number_input("프리미엄형 제공량(kWh)",0,5_000,1_000,10)
    overage=st.selectbox("최종 초과단가(원/kWh, 부가요금·세금 포함)",[200,300,307.3,400],index=1)
    tou_contract_kw=st.number_input("제주 TOU 계약전력 가정(kW)",1.0,30.0,3.0,1.0)
    st.divider();st.header("부가요금·세금")
    fuel_rate=st.number_input("연료비조정단가(원/kWh)",-5.0,20.0,5.0,0.5)
    climate_rate=st.number_input("기후환경요금단가(원/kWh)",0.0,30.0,9.0,0.5)
    vat_percent=st.number_input("부가가치세율(%)",0.0,20.0,10.0,0.1)
    fund_percent=st.number_input("전력산업기반기금 요율(%)",0.0,10.0,2.7,0.1)
    # 모듈 전역 계산함수에서 현재 설정을 사용함.
    FUEL_ADJUSTMENT_RATE=float(fuel_rate)
    CLIMATE_ENV_RATE=float(climate_rate)
    VAT_RATE=float(vat_percent)/100.0
    POWER_FUND_RATE=float(fund_percent)/100.0
    TOU_CONTRACT_KW=float(tou_contract_kw)
    st.info("일반 주택용(저압)과 제주 TOU에는 연료비조정액·기후환경요금·부가가치세·전력산업기반기금을 별도 반영합니다. 구독 기본형·프리미엄형은 표시된 월 구독료와 초과단가가 모든 부가요금·세금을 포함한 최종 소비자가격입니다.")
    st.caption("2026년 6월 요금표를 2024·2025년 사용량에 동일 적용합니다. 일반 주택용은 저압 요율, 제주 TOU는 별도 계시별 요율과 설정한 계약전력을 적용합니다.")

stacked_cluster,cluster_summary,cluster_wide,cluster_transition=joint_dynamic_clusters(D["customers"],cluster_count)
customers=enrich_scores(D["customers"],cluster_wide)
cluster_col={y:(f"{y}군집_동적" if f"{y}군집_동적" in customers.columns else f"{y}군집") for y in (2024,2025)}
with st.spinner("현재 요금 설정으로 월별·연간 추천요금제를 다시 계산하고 있습니다..."):
    tariff_dynamic=dynamic_tariff_analysis(D["monthly"],basic_fee,basic_inc,premium_fee,premium_inc,overage,
                                                   fuel_rate,climate_rate,VAT_RATE,POWER_FUND_RATE,TOU_CONTRACT_KW)

T1,T2,T3,T4,T5,T6,T7=st.tabs(["2024~2025년 사용량 분석","고객별 요금 모니터링","고객별 진단·제어","고객 군집 분석","요금분석 및 추천","계통영향 분석 및 제어 시뮬레이션","방법론·한계"])

with T1:
    S=D["stats"];c=st.columns(5)
    c[0].metric("분석 대상 고객",f"{S['2개년핵심고객수']:,}명")
    c[1].metric("2024년 연평균 사용량",fmt_kwh(S["2024연평균kWh"]))
    c[2].metric("2025년 연평균 사용량",fmt_kwh(S["2025연평균kWh"]),fmt_pct(S["연평균증감률"]))
    c[3].metric("동일 군집 유지 비율",fmt_pct(S["군집유지율"]))
    c[4].metric("추천 요금제 유지 비율",fmt_pct(tariff_dynamic["annual_stability"]))
    om=D["overall_monthly"].copy();om["연도"]=om["연도"].astype(int).astype(str);fig=px.line(om,x="월",y="고객당평균_kWh",color="연도",markers=True,labels={"고객당평균_kWh":"고객당 평균 사용량(kWh)"},category_orders={"연도":["2024","2025"]});fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kWh<extra></extra>");st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns([1.15,1])
    with left:
        mc=D["monthly_change"][["월","2024고객당평균_kWh","2025고객당평균_kWh","증감_kWh","증감률","경부하비중증감p","최대부하비중증감p"]].copy()
        mc["증감률(%)"]=mc.pop("증감률")*100;mc["경부하 비중 증감(%p)"]=mc.pop("경부하비중증감p")*100;mc["최대부하 비중 증감(%p)"]=mc.pop("최대부하비중증감p")*100
        mc["월"]=pd.to_numeric(mc["월"],errors="coerce").astype("Int64").astype(str)
        mc=mc.rename(columns={
            "2024고객당평균_kWh":"2024년 고객당 평균 사용량(kWh)",
            "2025고객당평균_kWh":"2025년 고객당 평균 사용량(kWh)",
            "증감_kWh":"사용량 증감(kWh)",
        })
        mc=round_table(mc);st.dataframe(mc,hide_index=True,use_container_width=True,column_config=dataframe_config(mc))
    with right:
        bins=pd.cut(customers["연간사용량증감률"],[-np.inf,-0.2,-0.05,0.05,0.2,np.inf],labels=["20% 이상 감소","5~20% 감소","±5% 이내","5~20% 증가","20% 이상 증가"])
        dist=bins.value_counts(sort=False).rename_axis("구간").reset_index(name="고객수");dist["비중(%)"]=dist["고객수"]/len(customers)*100
        fig=px.bar(dist,x="구간",y="고객수",text="비중(%)");fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside");st.plotly_chart(fig,use_container_width=True)
    st.subheader("계절·주중/주말 평균 부하곡선")
    a,b=st.columns(2);season=a.selectbox("계절",list(SEASON_MONTHS),key="overview_season");daytype=b.radio("일 유형",["주중","주말"],horizontal=True,key="overview_day")
    pp=D["overall_profiles"][(D["overall_profiles"]["계절"]==season)&(D["overall_profiles"]["일유형"]==daytype)].copy();pp["연도"]=pp["연도"].astype(int).astype(str)
    fig=px.line(pp,x="시간",y="고객당평균_kWh",color="연도",markers=True,category_orders={"연도":["2024","2025"]});fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kWh/h<extra></extra>");st.plotly_chart(fig,use_container_width=True)

with T2:
    st.subheader("요금 모니터링 및 요금제 추천")
    a,b,c,d=st.columns(4)
    period=a.selectbox("분석 기간",["연간 전체","월중 모니터링"])
    year=b.selectbox("분석 연도",[2024,2025],index=1)
    month=c.selectbox("분석 월",list(range(1,13)),index=7,disabled=period=="연간 전체")
    maxday=monthrange(int(year),int(month))[1];cutoff=d.slider("조회일",5,maxday-1,min(20,maxday-1),1,disabled=period=="연간 전체")
    e,f=st.columns(2);current_plan=e.selectbox("정액 알림 기준 요금제",["기본형","프리미엄형"]);sort_key=f.selectbox("정렬 기준",["수요관리우선점수","TOU대비절감(원)","월말예상(kWh)","연간사용량(kWh)"])
    with st.spinner("고객별 사용량과 요금을 계산하고 있습니다..."):
        monitor=build_tariff_monitor(D["daily"],D["monthly"],customers,cluster_col[year],period,year,month,cutoff,current_plan,basic_fee,basic_inc,premium_fee,premium_inc,overage)
    if not monitor.empty:
        f1,f2=st.columns(2);plan_options=["전체"]+sorted(monitor["추천요금제"].unique().tolist());cluster_options=["전체"]+sorted(monitor["군집"].astype(str).unique().tolist())
        plan_filter=f1.selectbox("추천요금제 필터",plan_options);cluster_filter=f2.selectbox("군집 필터",cluster_options)
        show=monitor.copy()
        if plan_filter!="전체":show=show[show["추천요금제"]==plan_filter]
        if cluster_filter!="전체":show=show[show["군집"].astype(str)==cluster_filter]
        if sort_key in show.columns:show=show.sort_values(sort_key,ascending=False)
        show=round_table(show)
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("전체 고객",f"{len(show):,}명")
        c2.metric("일반주택용 추천",f"{(show['추천요금제']=='일반 주택용(저압)').sum():,}명")
        c3.metric("제주 TOU 추천",f"{(show['추천요금제']=='제주 TOU').sum():,}명")
        c4.metric("기본형 추천",f"{(show['추천요금제']=='구독 기본형').sum():,}명")
        c5.metric("프리미엄형 추천",f"{(show['추천요금제']=='구독 프리미엄형').sum():,}명")
        c6.metric("평균 절감가능액",fmt_won1(show["TOU대비절감(원)"].mean() if len(show) else 0))
        st.dataframe(show,hide_index=True,use_container_width=True,height=520,column_config=dataframe_config(show))
        st.download_button("고객별 요금 모니터링 CSV",show.to_csv(index=False).encode("utf-8-sig"),f"v21_{year}_{period}_요금모니터링.csv","text/csv")

with T3:
    st.subheader("고객별 요금분석 및 사용량 관리·제어")
    cid=st.selectbox("고객 선택",customers["고객ID"].sort_values().tolist())
    r=customers.set_index("고객ID").loc[cid]
    pattern_score=float(r["패턴안정성점수"]);peak_score=float(r["수요관리우선점수"])
    c=st.columns(5);c[0].metric("2024 사용량",fmt_kwh(r["2024_연간사용량_kWh"]));c[1].metric("2025 사용량",fmt_kwh(r["2025_연간사용량_kWh"]),fmt_pct(r["연간사용량증감률"]));c[2].metric("사용패턴 일관성",usage_pattern_label(pattern_score),f"{pattern_score:.1f}점",help="2024년과 2025년의 총사용량, 시간대별 비중, 주말·주중 패턴과 부하율이 비슷할수록 높습니다.");c[3].metric("피크관리 필요도",peak_management_label(peak_score),f"{peak_score:.1f}점",help="최대시간 부하, 최대부하시간대 사용비중, 연간 사용량, 냉난방 민감도와 예측 가능성을 종합한 상대순위입니다.");c[4].metric("최근 변화 신호",r["구조변화신호"])
    st.caption("**사용패턴 일관성**은 두 해의 생활·사용패턴이 얼마나 비슷한지를 뜻합니다. **피크관리 필요도**가 높을수록 한전의 피크 알림·부하이동·직접제어 실증을 우선 검토할 고객입니다.")
    annual_rec_customer=tariff_dynamic["annual_customer"][tariff_dynamic["annual_customer"]["고객ID"]==cid].set_index("연도")
    rec24=annual_rec_customer.loc[2024,"연간추천요금제"] if 2024 in annual_rec_customer.index else "자료 없음"
    rec25=annual_rec_customer.loc[2025,"연간추천요금제"] if 2025 in annual_rec_customer.index else "자료 없음"
    st.info(f"군집: {r[cluster_col[2024]]} → {r[cluster_col[2025]]} / 현재 요금 설정의 연간 추천: {rec24} → {rec25}")
    period=st.radio("진단 기간",["연간 종합진단","월별 목표관리·제어"],horizontal=True)
    if period=="연간 종합진단":
        cm=D["monthly"][D["monthly"]["고객ID"]==cid].copy();cm_chart=cm.copy();cm_chart["연도"]=cm_chart["연도"].astype(int).astype(str);fig=px.line(cm_chart,x="월",y="사용량_kWh",color="연도",markers=True,category_orders={"연도":["2024","2025"]});fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kWh<extra></extra>");st.plotly_chart(fig,use_container_width=True)
        st.subheader("연간 요금 및 연간 추천요금제")
        rows=[]
        annual_customer=tariff_dynamic["annual_customer"]
        for y in (2024,2025):
            ar=annual_customer[(annual_customer["고객ID"]==cid)&(annual_customer["연도"]==y)].iloc[0]
            for plan_name,bill_col in PLAN_BILL_COLUMNS.items():
                rows.append({"연도":str(y),"요금제":plan_name,"연간요금(원)":ar[bill_col],"월평균요금(원)":ar[bill_col]/12.0,
                             "판정":"연간 추천" if plan_name==ar["연간추천요금제"] else "비교"})
        tdf=round_table(pd.DataFrame(rows));display_full_text_table(tdf)
        st.subheader("월별 추천요금제 변화")
        mr=tariff_dynamic["monthly_customer"][tariff_dynamic["monthly_customer"]["고객ID"]==cid].copy()
        pivot=mr.pivot(index="월",columns="연도",values="월별추천요금제").reset_index().rename(columns={2024:"2024 월별추천",2025:"2025 월별추천"})
        pivot["변경여부"]=np.where(pivot["2024 월별추천"]==pivot["2025 월별추천"],"유지","변경")
        display_full_text_table(pivot)
        season=st.selectbox("대표 계절",list(SEASON_MONTHS),key="annual_diag_season");tabs=st.tabs(["주중","주말"])
        for tab,dt in zip(tabs,["주중","주말"]):
            with tab:
                arr=[]
                for y in (2024,2025):p=profile_for_customer(D["profiles"],cid,y,season,dt);p["연도"]=y;arr.append(p)
                pf=pd.concat(arr);pf["연도"]=pf["연도"].astype(int).astype(str);fig=px.line(pf,x="시간",y="평균사용량_kWh",color="연도",markers=True,category_orders={"연도":["2024","2025"]});fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kWh/h<extra></extra>");st.plotly_chart(fig,use_container_width=True)
    else:
        a,b,c=st.columns(3)
        year=a.selectbox("대상 연도",[2024,2025],index=1,key="diag_year")
        month=b.selectbox("대상 월",list(range(1,13)),index=7,key="diag_month")
        cutoff=c.slider("조회일",5,monthrange(year,month)[1]-1,min(20,monthrange(year,month)[1]-1),1,key="diag_cutoff")

        a,b,c=st.columns(3)
        plan_choices=["일반 주택용(저압)","제주 TOU","구독 기본형","구독 프리미엄형"]
        current_plan=a.selectbox("현재 적용 요금제",plan_choices,index=2)
        management=b.radio("관리 방식",["알림·행동권고","한전 직접제어 위임"],horizontal=False)
        mode=c.selectbox("제어·권고 강도",list(CONTROL_MODES),index=1)
        ownership=st.multiselect("등록·연결된 기기",sorted({str(x["ownership"]) for x in ACTION_LIBRARY}),default=sorted({str(x["ownership"]) for x in ACTION_LIBRARY}))

        cd=D["daily"][D["daily"]["고객ID"]==cid]
        f=forecast_month_longitudinal(cd,year,month,cutoff)
        mrow=D["monthly"][(D["monthly"]["고객ID"]==cid)&(D["monthly"]["연도"]==year)&(D["monthly"]["월"]==month)].iloc[0]
        bills=monthly_bill_map(f["forecast"],month,mrow,basic_fee,basic_inc,premium_fee,premium_inc,overage)
        rec=min(bills,key=bills.get)
        current_bill=float(bills[current_plan])
        is_subscription=current_plan in ("구독 기본형","구독 프리미엄형")
        if current_plan=="구독 기본형":
            inc,fee=basic_inc,basic_fee
        elif current_plan=="구독 프리미엄형":
            inc,fee=premium_inc,premium_fee
        else:
            inc,fee=None,None

        c=st.columns(6)
        c[0].metric("현재 누적",fmt_kwh(f["current"]))
        c[1].metric("남은 제공량",fmt_kwh(max(inc-f["current"],0)) if is_subscription else "정액제 아님")
        c[2].metric("월말 예상",fmt_kwh(f["forecast"]))
        c[3].metric("예상 범위",f"{f['lower']:,.1f}~{f['upper']:,.1f}kWh")
        c[4].metric("현재요금제 예상",fmt_won1(current_bill))
        c[5].metric("추천요금제",rec)

        st.subheader("4개 요금제 적용 시 예상 납부액 비교")
        comparison=round_table(tariff_comparison_table(bills,current_plan))
        display_full_text_table(comparison)
        saving=max(current_bill-float(bills[rec]),0.0)
        if rec!=current_plan and saving>0:
            st.success(f"현재 사용 추세에서는 **{rec}** 적용 시 현재 요금제보다 월 약 **{fmt_won1(saving)}** 절감될 것으로 예상됩니다.")
        else:
            st.info("현재 사용 추세에서는 현재 적용 요금제가 비용상 최저이거나 추천요금제와 동일합니다.")

        if is_subscription:
            target_options=["현재 요금제 제공량 이내","전월과 같은 요금","전년 동월과 같은 요금","목표요금 직접 입력"]
        else:
            target_options=["현재 요금제 예상요금 이내","전월과 같은 요금","전년 동월과 같은 요금","목표요금 직접 입력"]
        target_kind=st.radio("관리 목표",target_options,horizontal=True)
        target_usage=float(inc if is_subscription else f["forecast"])

        if target_kind=="전월과 같은 요금":
            py,pm=(year-1,12) if month==1 else (year,month-1)
            prev=D["monthly"][(D["monthly"]["고객ID"]==cid)&(D["monthly"]["연도"]==py)&(D["monthly"]["월"]==pm)]
            if len(prev):
                prev_row=prev.iloc[0]
                target_bill=bill_for_plan(current_plan,float(prev_row["사용량_kWh"]),pm,prev_row,basic_fee,basic_inc,premium_fee,premium_inc,overage)
                target_usage=inverse_bill_for_plan(target_bill,current_plan,month,mrow,basic_fee,basic_inc,premium_fee,premium_inc,overage)
            else:
                st.warning("전월 자료가 없어 기본 목표를 적용합니다.")
        elif target_kind=="전년 동월과 같은 요금":
            prev=D["monthly"][(D["monthly"]["고객ID"]==cid)&(D["monthly"]["연도"]==year-1)&(D["monthly"]["월"]==month)]
            if len(prev):
                prev_row=prev.iloc[0]
                target_bill=bill_for_plan(current_plan,float(prev_row["사용량_kWh"]),month,prev_row,basic_fee,basic_inc,premium_fee,premium_inc,overage)
                target_usage=inverse_bill_for_plan(target_bill,current_plan,month,mrow,basic_fee,basic_inc,premium_fee,premium_inc,overage)
            else:
                st.warning("전년 동월 자료가 없어 기본 목표를 적용합니다.")
        elif target_kind=="목표요금 직접 입력":
            target_bill=st.number_input("목표 월 납부액(원)",0,1_000_000,int(round(current_bill)),1_000)
            target_usage=inverse_bill_for_plan(target_bill,current_plan,month,mrow,basic_fee,basic_inc,premium_fee,premium_inc,overage)

        if target_usage<f["current"]:
            st.warning("이미 누적사용량이 목표 사용량을 초과하여 이번 달에는 목표 달성이 어렵습니다. 가능한 범위의 감축계획만 산정합니다.")
        required=max(f["forecast"]-target_usage,0)
        plan,gross,effective=optimize_actions(required,f["remaining_days"],season_for_month(month),ownership,mode,direct=management=="한전 직접제어 위임")
        controlled_forecast=max(f["forecast"]-effective,f["current"])
        remaining_gap=max(controlled_forecast-target_usage,0.0)
        c=st.columns(4)
        c[0].metric("목표 사용량",fmt_kwh(target_usage))
        c[1].metric("필요 감축량",fmt_kwh(required))
        c[2].metric("계획 실효감축",fmt_kwh(effective))
        c[3].metric("관리 후 예상",fmt_kwh(controlled_forecast))
        if not plan.empty:
            ps=round_table(plan)
            display_full_text_table(ps)
            st.download_button("행동·제어계획 CSV",ps.to_csv(index=False).encode("utf-8-sig"),f"{cid}_{year}_{month}월_제어계획.csv","text/csv")
        else:
            st.info("현재 목표를 위해 추가로 선택할 수 있는 행동대안이 없거나 감축이 필요하지 않습니다.")

        goal_tolerance=max(1.0,float(target_usage)*0.005)
        if remaining_gap>goal_tolerance:
            controlled_bills=monthly_bill_map(controlled_forecast,month,mrow,basic_fee,basic_inc,premium_fee,premium_inc,overage)
            controlled_rec=min(controlled_bills,key=controlled_bills.get)
            st.warning(f"현재 등록기기와 **{mode}** 설정만으로는 목표를 충족하기 어렵습니다. 관리 후에도 목표 사용량을 약 **{fmt_kwh(remaining_gap)}** 초과할 것으로 예상됩니다.")
            if controlled_rec in ("구독 기본형","구독 프리미엄형"):
                st.info(f"관리 후 예상 사용량을 기준으로는 **{controlled_rec}**가 비용상 가장 유리합니다. 비슷한 초과가 반복된다면 사용량을 무리하게 억제하기보다 해당 **구독서비스 전환**을 검토하는 편이 적절합니다.")
            else:
                st.info(f"관리 후 예상 사용량을 기준으로는 **{controlled_rec}**가 가장 유리합니다. 현재 구독형보다 이 요금제를 유지·전환하는 방안이 비용 측면에서 적합할 수 있습니다.")
            st.markdown("**추가 대안** ① 직접제어 허용기기 확대 또는 제어강도 상향 ② 목표요금·목표사용량의 현실적 조정 ③ 초과요금을 감수하고 고객 편의 유지 ④ 다음 달부터 조기 알림·제어 시작")
        elif required>0:
            st.success("현재 등록기기와 제어·권고 범위에서 목표 달성이 가능한 것으로 추정됩니다.")

        dm=cd[(cd["연도"]==year)&(cd["월"]==month)]
        proj=cumulative_projection(dm,cutoff,f["forecast"],effective*(0.8 if management=="알림·행동권고" else 0.0),effective if management=="한전 직접제어 위임" else 0.0)
        fig=go.Figure()
        for col in ["실제누적","미제어예상","행동권고예상","직접제어예상"]:
            fig.add_trace(go.Scatter(x=proj["일"],y=proj[col],name=col,mode="lines+markers"))
        fig.add_hline(y=target_usage,line_dash="dash",annotation_text="목표 사용량")
        fig.update_layout(xaxis_title="일",yaxis_title="월 누적 사용량(kWh)")
        fig.update_traces(hovertemplate="%{y:,.1f}kWh<extra></extra>")
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("주중·주말 평균 부하곡선: 관리 전후")
        tabs=st.tabs(["주중","주말"])
        for tab,dt in zip(tabs,["주중","주말"]):
            with tab:
                p=D["profiles"][(D["profiles"]["고객ID"]==cid)&(D["profiles"]["연도"]==year)&(D["profiles"]["월"]==month)&(D["profiles"]["일유형"]==dt)].sort_values("시간")
                base=p["평균사용량_kWh"].to_numpy(float)
                after=controlled_profile(base,plan,f["remaining_days"])
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=p["시간"],y=base,name="관리 전",mode="lines+markers"))
                fig.add_trace(go.Scatter(x=p["시간"],y=after,name="관리 후 예상",mode="lines+markers"))
                fig.update_xaxes(dtick=1)
                fig.update_yaxes(title="평균부하(kWh/h)")
                fig.update_traces(hovertemplate="%{y:,.1f}<extra></extra>")
                st.plotly_chart(fig,use_container_width=True)

with T4:
    st.subheader(f"공통 기준 {cluster_count}개 군집과 2024→2025 전이")
    cs=cluster_summary.copy()
    cs["연도"]=cs["연도"].astype(int).astype(str)
    cs["비중(%)"]=pd.to_numeric(cs.pop("비중"),errors="coerce")*100
    percentage_map={
        "주말주중비":"주중 대비 주말 사용량 비중(%)",
        "경부하비중":"경부하 비중(%)",
        "중간부하비중":"중간부하 비중(%)",
        "최대부하비중":"최대부하 비중(%)",
        "월변동계수":"월 변동계수(%)",
        "부하율":"부하율(%)",
        "하계민감도":"하계 민감도(%)",
        "동계민감도":"동계 민감도(%)",
    }
    for source_col,display_col in percentage_map.items():
        if source_col in cs.columns:
            cs[display_col]=pd.to_numeric(cs.pop(source_col),errors="coerce")*100
    cs=cs.rename(columns={
        "연간사용량_kWh":"연간 사용량(kWh)",
        "최대시간사용량_kWh":"최대시간 사용량(kWh)",
    })
    cs=round_table(cs)
    st.dataframe(cs.sort_values(["연도","고객수"],ascending=[True,False]),hide_index=True,use_container_width=True,column_config=dataframe_config(cs))
    stability=(cluster_wide["군집유지여부"]=="유지").mean();c=st.columns(3);c[0].metric("동일 군집 유지율",fmt_pct(stability));c[1].metric("군집 이동 고객",f"{(cluster_wide['군집유지여부']=='이동').sum():,}명");c[2].metric("군집 수",f"{cluster_count}개")
    matrix=cluster_transition.pivot(index="2024군집",columns="2025군집",values="고객수").fillna(0);fig=px.imshow(matrix,text_auto=True,aspect="auto",color_continuous_scale="Blues");st.plotly_chart(fig,use_container_width=True)
    tt=cluster_transition.copy()
    tt["2024군집내비중(%)"]=tt.pop("2024군집내비중")*100
    tt=tt.rename(columns={"고객수":"고객수(명)"})
    tt["고객수(명)"]=pd.to_numeric(tt["고객수(명)"],errors="coerce").fillna(0).astype(int)
    tt=round_table(tt)
    st.dataframe(tt,hide_index=True,use_container_width=True,column_config=dataframe_config(tt))

with T5:
    st.subheader("연간·월별 추천요금제와 예측성능")
    st.caption("사이드바의 기본형·프리미엄형 구독료, 제공량 및 초과단가를 바꾸면 아래 추천 고객 수와 요금이 즉시 다시 계산됩니다.")
    st.info(f"일반·TOU 반영값: 연료비조정 {fuel_rate:.1f}원/kWh + 기후환경 {climate_rate:.1f}원/kWh, 부가세 {vat_percent:.1f}%, 전력산업기반기금 {fund_percent:.1f}%. 구독료와 초과단가는 이 항목을 포함한 최종가격입니다.")

    st.markdown("### 1. 연간 추천 요금제")
    annual_summary=tariff_dynamic["annual_summary"].copy().drop(columns=["고객당중앙연간요금(원)","고객당중앙월요금(원)"],errors="ignore")
    annual_summary=annual_summary.rename(columns={"연간추천고객수":"연간 추천 고객 수(명)"})
    if "연간 추천 고객 수(명)" in annual_summary.columns:
        annual_summary["연간 추천 고객 수(명)"]=pd.to_numeric(annual_summary["연간 추천 고객 수(명)"],errors="coerce").fillna(0).astype(int)
    annual_summary=round_table(annual_summary)
    st.dataframe(annual_summary,hide_index=True,use_container_width=True,column_config=dataframe_config(annual_summary))
    annual_year=st.radio("연간 추천 분석연도",[2024,2025],index=1,horizontal=True,key="annual_tariff_year")
    annual_selected=tariff_dynamic["annual_customer"][tariff_dynamic["annual_customer"]["연도"]==annual_year].copy()
    counts=annual_selected["연간추천요금제"].value_counts().reindex(PLAN_ORDER,fill_value=0)
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("일반주택용 추천",f"{int(counts['일반 주택용(저압)']):,}명")
    c2.metric("제주 TOU 추천",f"{int(counts['제주 TOU']):,}명")
    c3.metric("기본형 추천",f"{int(counts['구독 기본형']):,}명")
    c4.metric("프리미엄형 추천",f"{int(counts['구독 프리미엄형']):,}명")
    c5.metric("2024→2025 연간추천 유지",fmt_pct(tariff_dynamic["annual_stability"]))
    acount=pd.DataFrame({"요금제":PLAN_ORDER,"추천고객수":[int(counts[p]) for p in PLAN_ORDER]})
    fig=px.bar(acount,x="요금제",y="추천고객수",text="추천고객수",title=f"{annual_year}년 연간 추천요금제 분포")
    fig.update_traces(textposition="outside");st.plotly_chart(fig,use_container_width=True)

    af1,af2=st.columns(2)
    annual_plan_filter=af1.selectbox("연간 추천요금제 필터",["전체"]+PLAN_ORDER,key="annual_plan_filter")
    annual_sort=af2.selectbox("연간 표 정렬",["연간TOU대비절감(원)","연간사용량(kWh)","월평균사용량(kWh)","연간최저요금(원)"],key="annual_sort")
    annual_show=annual_selected.copy()
    if annual_plan_filter!="전체":annual_show=annual_show[annual_show["연간추천요금제"]==annual_plan_filter]
    annual_show=annual_show.sort_values(annual_sort,ascending=False)
    annual_show=round_table(annual_show[["고객ID","연간사용량(kWh)","월평균사용량(kWh)","일반주택용(원)","제주TOU(원)","기본형(원)","프리미엄형(원)","연간추천요금제","연간TOU대비절감(원)"]])
    st.dataframe(annual_show,hide_index=True,use_container_width=True,height=420,column_config=dataframe_config(annual_show))
    st.download_button("연간 추천요금제 고객표 CSV",annual_show.to_csv(index=False).encode("utf-8-sig"),f"v21_{annual_year}_연간추천요금제.csv","text/csv")

    st.markdown("#### 연간 추천 요금제 조정·변경")
    transition=round_table(tariff_dynamic["annual_transition"].copy())
    st.dataframe(transition,hide_index=True,use_container_width=True,column_config=dataframe_config(transition))

    st.markdown("### 2. 월별 추천 요금제")
    my1,my2=st.columns(2)
    monthly_year=my1.selectbox("월별 추천 분석연도",[2024,2025],index=1,key="monthly_tariff_year")
    monthly_month=my2.selectbox("월별 추천 분석월",list(range(1,13)),index=7,key="monthly_tariff_month")
    monthly_selected=tariff_dynamic["monthly_customer"][(tariff_dynamic["monthly_customer"]["연도"]==monthly_year)&(tariff_dynamic["monthly_customer"]["월"]==monthly_month)].copy()
    mcounts=monthly_selected["월별추천요금제"].value_counts().reindex(PLAN_ORDER,fill_value=0)
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("일반주택용 추천",f"{int(mcounts['일반 주택용(저압)']):,}명")
    m2.metric("제주 TOU 추천",f"{int(mcounts['제주 TOU']):,}명")
    m3.metric("기본형 추천",f"{int(mcounts['구독 기본형']):,}명")
    m4.metric("프리미엄형 추천",f"{int(mcounts['구독 프리미엄형']):,}명")
    m5.metric("월별 추천 연도간 유지",fmt_pct(tariff_dynamic["monthly_stability"]))
    mcount_df=pd.DataFrame({"요금제":PLAN_ORDER,"추천고객수":[int(mcounts[p]) for p in PLAN_ORDER]})
    fig=px.bar(mcount_df,x="요금제",y="추천고객수",text="추천고객수",title=f"{monthly_year}년 {monthly_month}월 추천요금제 분포")
    fig.update_traces(textposition="outside");st.plotly_chart(fig,use_container_width=True)

    monthly_summary=tariff_dynamic["monthly_summary"].copy();monthly_summary["연도"]=monthly_summary["연도"].astype(int).astype(str)
    if "월" in monthly_summary.columns:
        monthly_summary["월"]=pd.to_numeric(monthly_summary["월"],errors="coerce").astype("Int64").astype(str)
    monthly_summary=round_table(monthly_summary)
    st.dataframe(monthly_summary[monthly_summary["연도"]==str(monthly_year)],hide_index=True,use_container_width=True,height=320,column_config=dataframe_config(monthly_summary))
    monthly_show=round_table(monthly_selected[["고객ID","사용량(kWh)","일반주택용(원)","제주TOU(원)","기본형(원)","프리미엄형(원)","월별추천요금제","연간추천요금제","월·연간추천일치"]])
    st.dataframe(monthly_show,hide_index=True,use_container_width=True,height=420,column_config=dataframe_config(monthly_show))
    st.download_button("월별 추천요금제 고객표 CSV",monthly_show.to_csv(index=False).encode("utf-8-sig"),f"v21_{monthly_year}_{monthly_month}월_추천요금제.csv","text/csv")

    st.markdown("### 3. 월말 사용량 예측성능")
    ff=customers[["고객ID","15일예측_MAPE","20일예측_MAPE","15일예측_오차10%이내","20일예측_오차10%이내"]].copy()
    ff["15일예측_MAPE(%)"]=ff.pop("15일예측_MAPE")*100;ff["20일예측_MAPE(%)"]=ff.pop("20일예측_MAPE")*100
    ff["15일예측_오차10%이내(%)"]=ff.pop("15일예측_오차10%이내")*100;ff["20일예측_오차10%이내(%)"]=ff.pop("20일예측_오차10%이내")*100
    ff=round_table(ff);st.dataframe(ff,hide_index=True,use_container_width=True,height=420,column_config=dataframe_config(ff))

with T6:
    st.subheader("100 가구 무작위 추출 분석")
    c1,c2,c3,c4=st.columns(4);seed=c1.number_input("표본 추출번호",0,9999,42,1);season=c2.selectbox("계절",list(SEASON_MONTHS),key="portfolio_season");daytype=c3.radio("일 유형",["주중","주말"],horizontal=True,key="portfolio_day");target_year=c4.selectbox("제어 적용연도",[2024,2025],index=1)
    rng=np.random.default_rng(int(seed));ids=rng.choice(customers["고객ID"].to_numpy(),size=100,replace=False).tolist();base24=aggregate_portfolio_profile(D["profiles"],ids,2024,season,daytype);base25=aggregate_portfolio_profile(D["profiles"],ids,2025,season,daytype)
    ptable=customers[customers["고객ID"].isin(ids)][["고객ID","2024_연간사용량_kWh","2025_연간사용량_kWh","연간사용량증감률",cluster_col[2024],cluster_col[2025]]].copy();ptable["연간사용량증감률(%)"]=ptable.pop("연간사용량증감률")*100
    ptable=ptable.rename(columns={
        "2024_연간사용량_kWh":"2024년 연간 사용량(kWh)",
        "2025_연간사용량_kWh":"2025년 연간 사용량(kWh)",
        cluster_col[2024]:"2024년 군집 유형",
        cluster_col[2025]:"2025년 군집 유형",
    })
    c=st.columns(4);c[0].metric("2024 연간 합계",f"{ptable['2024년 연간 사용량(kWh)'].sum()/1000:,.1f}MWh");c[1].metric("2025 연간 합계",f"{ptable['2025년 연간 사용량(kWh)'].sum()/1000:,.1f}MWh",fmt_pct(ptable['2025년 연간 사용량(kWh)'].sum()/ptable['2024년 연간 사용량(kWh)'].sum()-1));c[2].metric("2024 대표일 피크",f"{base24.max():,.1f}kW");c[3].metric("2025 대표일 피크",f"{base25.max():,.1f}kW",fmt_pct(base25.max()/max(base24.max(),1e-9)-1))
    fig=go.Figure();fig.add_trace(go.Scatter(x=np.arange(1,25),y=base24,name="2024 제어 전"));fig.add_trace(go.Scatter(x=np.arange(1,25),y=base25,name="2025 제어 전"));fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kW<extra></extra>");st.plotly_chart(fig,use_container_width=True)
    a,b=st.columns(2);limit_pct=a.slider("변압기 목표 운전한도(제어 전 최대부하 대비)",60,110,90,1);participation_pct=b.slider("직접제어 참여율",0,100,70,5);base=base24 if target_year==2024 else base25;result=optimize_transformer_profile(base,limit_pct/100,participation_pct/100)
    c=st.columns(6);c[0].metric("제어 전 피크",f"{result['peak_before']:,.1f}kW");c[1].metric("목표 운전한도",f"{result['limit']:,.1f}kW");c[2].metric("제어 후 피크",f"{result['peak_after']:,.1f}kW");c[3].metric("한도 초과시간",f"{result['hours_before']}→{result['hours_after']}시간");c[4].metric("시간이동량",f"{result['shifted']:,.1f}kWh");c[5].metric("실제 감축량",f"{result['reduced']:,.1f}kWh")
    control=pd.DataFrame({"시간":np.arange(1,25),"제어전(kW)":base,"시간이동출력(kW)":result["shift_out"],"실제감축출력(kW)":result["reduction"],"이동유입(kW)":result["shift_in"],"제어후(kW)":result["after"],"운전한도(kW)":result["limit"]});fig=go.Figure();fig.add_trace(go.Scatter(x=control["시간"],y=control["제어전(kW)"],name="제어 전"));fig.add_trace(go.Scatter(x=control["시간"],y=control["제어후(kW)"],name="제어 후"));fig.add_trace(go.Scatter(x=control["시간"],y=control["운전한도(kW)"],name="운전한도",line=dict(dash="dash")));fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kW<extra></extra>");st.plotly_chart(fig,use_container_width=True)
    ps=round_table(ptable);st.dataframe(ps.sort_values("2025년 연간 사용량(kWh)",ascending=False),hide_index=True,use_container_width=True,column_config=dataframe_config(ps));st.download_button("100가구 분석결과 ZIP",zip_results({"100가구_고객목록.csv":ps.to_csv(index=False).encode("utf-8-sig"),"100가구_변압기제어상세.csv":round_table(control).to_csv(index=False).encode("utf-8-sig")}),"v21_100가구_분석결과.zip","application/zip")

with T7:
    st.subheader("시뮬레이터의 개념적·구조적 한계")
    st.markdown("""
#### 1. 분석대상과 대표성의 한계
- 분석대상은 2024년과 2025년 모두 연간 자료와 높은 시간값 완전도를 갖춘 제주 TOU 공통고객 712명으로 한정됩니다.
- 중도 가입·해지, 이사, 장기결측 고객이 제외되어 전체 주택용 고객이나 신규 가입고객의 특성을 그대로 대표하지 않을 수 있습니다.
- 제주 TOU 고객은 이미 시간대별 요금제를 선택한 집단이므로 일반 주택용 전체 모집단과 사용행태가 다를 수 있습니다.

#### 2. 사용량 예측의 한계
- 예측은 2개년 계량자료, 주중·주말 패턴과 최근 사용추세를 중심으로 산정합니다.
- 기온·습도·강수, 가구원 수, 재택 여부, 전기차·태양광·전기난방 보유, 이사·가전 교체 등의 외부요인은 직접 반영하지 못합니다.
- 과거와 다른 폭염·한파 또는 생활여건 변화가 발생하면 월말 사용량과 요금의 예측오차가 커질 수 있습니다.

#### 3. 가전별 진단과 제어효과의 한계
- 세대 계량기는 가구 전체 사용량만 측정하므로 특정 가전의 실시간 사용량을 직접 확인할 수 없습니다.
- 가전별 절감량과 불편점수는 고객이 등록한 기기정보와 표준 가전자료를 결합한 시나리오값이며, 실제 절감량을 보장하지 않습니다.
- 한전 직접제어는 스마트가전·스마트플러그·HEMS 연결, 고객의 사전동의, 수동해제, 통신장애 시 복귀조건이 갖춰졌다는 가정입니다.

#### 4. 고객 군집의 한계
- 군집은 사용량·시간대 비중·계절민감도 등 선택된 지표를 기준으로 한 상대적 분류이며, 실제 생활유형이나 가전 보유형태를 확정적으로 의미하지 않습니다.
- 군집 수를 변경하면 고객의 소속과 군집 명칭이 달라질 수 있으며, 군집 이동이 곧 고객행동의 구조적 변화를 의미하지는 않습니다.
- 군집분석은 관계와 유사성을 보여주지만 TOU 요금제가 사용행태 변화를 유발했다는 인과관계를 입증하지 않습니다.

#### 5. 요금분석의 한계
- 2024·2025년 사용량에 2026년 6월 시행 요금표를 동일 적용한 비교 시나리오로, 과거 실제 청구액을 재현한 결과가 아닙니다.
- 제주 TOU 계약전력은 화면에서 설정한 가정값을 사용하며, 고객별 실제 계약전력이 다르면 요금이 달라질 수 있습니다.
- 구독 기본형·프리미엄형의 가격·제공량·초과단가는 연구용 상품가정이며, 규제승인·약관·원가회수·역선택과 도덕적 해이를 검증한 확정상품이 아닙니다.

#### 6. 계통영향 분석의 한계
- 100가구 포트폴리오는 712명 중 무작위로 추출한 집단이며 실제로 동일 변압기나 배전선로에 연결된 고객군이 아닙니다.
- 변압기 목표 운전한도는 제어 전 최대부하의 비율로 설정한 가상목표이며 실제 정격용량(kVA), 역률, 상별 불평형, 변압기 온도와 열적 수명은 반영하지 않습니다.
- 배전선로 임피던스, 전압·무효전력, 태양광 역송전, 고조파, 고장·정비조건과 N-1 제약을 계산하지 않으므로 실제 계통안정성을 보장하는 DMS 또는 전력조류 모델은 아닙니다.
- 부하이동·감축 가능비율과 참여율은 시뮬레이션 가정이며, 실제 고객반응·통신성공률·수동해제에 따라 확보 가능한 유연성이 달라집니다.

#### 7. 개인정보·운영체계의 한계
- 시간대별 전력사용량은 재실·생활패턴을 유추할 수 있는 민감정보이므로 비식별화, 접근통제, 보유기간 제한과 목적 외 사용금지가 필요합니다.
- 실제 서비스에는 고객동의, 설명가능한 추천근거, 이의제기·수동해제, 취약고객 보호, 제어실패 보상과 책임범위를 별도로 마련해야 합니다.
""")
    st.warning("본 시뮬레이터는 구독형 요금제와 수요관리 알고리즘의 개념검증·내부 의사결정 지원도구이며, 실제 요금상품 출시나 배전계통 운전명령에 직접 사용하는 운영시스템은 아닙니다.")
