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

APP_VERSION = "2026-07-20-actual-tou-v13.0"
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
    "프리미엄형": {"fee": 249_900.0, "included": 1_000.0},
}

SEASON_MONTHS = {
    "봄가을": [3, 4, 5, 9, 10],
    "여름": [6, 7, 8],
    "겨울": [1, 2, 11, 12],
}


def fmt_won(v: float) -> str:
    return f"{float(v):,.1f}원"


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


def residential_bill(kwh: float, month: int) -> float:
    u=max(float(kwh),0.0)
    if month in (7,8):
        basic=730 if u<=300 else (1260 if u<=450 else 6060)
        energy=min(u,300)*105.0 + min(max(u-300,0),150)*174.0 + max(u-450,0)*242.3
    else:
        basic=910 if u<=200 else (1600 if u<=400 else 7300)
        energy=min(u,200)*120.0 + min(max(u-200,0),200)*214.6 + max(u-400,0)*307.3
    return float((basic+energy)*1.127)


def tou_bill(kwh: float, month: int, off_share: float, mid_share: float, peak_share: float, contract_kw: float=3.0) -> float:
    rates=(125.8,153.8,172.4) if month in (3,4,5,9,10) else (138.7,184.7,220.5)
    energy=float(kwh)*(float(off_share)*rates[0]+float(mid_share)*rates[1]+float(peak_share)*rates[2])
    return float((4310*contract_kw+energy)*1.127)


def subscription_bill(kwh: float, fee: float, included: float, overage: float) -> float:
    return float(fee+max(float(kwh)-float(included),0.0)*float(overage))


def inverse_subscription_bill(target_bill: float, fee: float, included: float, overage: float) -> float:
    if target_bill<=fee or overage<=0: return float(included)
    return float(included+(target_bill-fee)/overage)


def fmt_won1(v: float) -> str:
    return f"{float(v):,.1f}원"


def fmt_pct(v: float) -> str:
    return f"{float(v)*100:.1f}%"


def round_table(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    for c in out.select_dtypes(include=[np.number]).columns:
        out[c]=out[c].round(1)
    return out


def dataframe_config(df: pd.DataFrame, percent_cols: List[str] | None=None) -> Dict[str, object]:
    percent_cols=set(percent_cols or [])
    cfg={}
    for c in df.columns:
        if c in percent_cols or "(%)" in c or "%p" in c:
            cfg[c]=st.column_config.NumberColumn(format="%.1f%%")
        elif "원" in c:
            cfg[c]=st.column_config.NumberColumn(format="₩%,.1f")
        elif pd.api.types.is_numeric_dtype(df[c]):
            cfg[c]=st.column_config.NumberColumn(format="%,.1f")
    return cfg


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
        "일반 주택용": residential_bill(usage,month),
        "제주 TOU(3kW)": tou_bill(usage,month,float(monthly_row["경부하비중"]),float(monthly_row["중간부하비중"]),float(monthly_row["최대부하비중"]),3.0),
        "구독 기본형": subscription_bill(usage,basic_fee,basic_inc,overage),
        "구독 프리미엄형": subscription_bill(usage,premium_fee,premium_inc,overage),
    }


def annual_bill_map(customer_monthly: pd.DataFrame, basic_fee: float, basic_inc: float, premium_fee: float,
                    premium_inc: float, overage: float) -> Dict[str,float]:
    totals={"일반 주택용":0.0,"제주 TOU(3kW)":0.0,"구독 기본형":0.0,"구독 프리미엄형":0.0}
    for _,r in customer_monthly.sort_values("월").iterrows():
        b=monthly_bill_map(float(r["사용량_kWh"]),int(r["월"]),r,basic_fee,basic_inc,premium_fee,premium_inc,overage)
        for k,v in b.items(): totals[k]+=v
    return totals


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
                         "일반주택용(원)":bills["일반 주택용"],"제주TOU(원)":bills["제주 TOU(3kW)"],"기본형(원)":bills["구독 기본형"],
                         "프리미엄형(원)":bills["구독 프리미엄형"],"추천요금제":rec,"TOU대비절감(원)":max(bills["제주 TOU(3kW)"]-bills[rec],0),
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
                         "일반주택용(원)":bills["일반 주택용"],"제주TOU(원)":bills["제주 TOU(3kW)"],"기본형(원)":bills["구독 기본형"],
                         "프리미엄형(원)":bills["구독 프리미엄형"],"추천요금제":rec,"TOU대비절감(원)":max(bills["제주 TOU(3kW)"]-bills[rec],0),
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


st.set_page_config(page_title="제주 TOU 2개년 심화분석", page_icon="⚡", layout="wide")
st.title("제주 TOU 공통고객 2개년 종단분석·요금 모니터링·제어 시뮬레이터")
st.caption(f"앱 버전 {APP_VERSION} · 2024~2025년 공통 핵심고객 712명 · 표 수치는 최대 소수점 첫째자리")

try:
    D=load_data()
except Exception as exc:
    st.error(str(exc));st.stop()

with st.sidebar:
    st.header("분석 설정")
    cluster_count=st.slider("공통 군집 수",3,8,8,1)
    st.divider();st.header("요금 가정")
    basic_fee=st.number_input("기본형 월 구독료(원)",0,500_000,84_900,1_000)
    basic_inc=st.number_input("기본형 제공량(kWh)",0,3_000,450,10)
    premium_fee=st.number_input("프리미엄형 월 구독료(원)",0,800_000,249_900,1_000)
    premium_inc=st.number_input("프리미엄형 제공량(kWh)",0,5_000,1_000,10)
    overage=st.selectbox("초과단가(원/kWh)",[200,300,307.3,400],index=1)
    st.info("2026년 6월 요금표를 2024·2025년 사용량에 동일 적용한 비교 시나리오입니다.")

stacked_cluster,cluster_summary,cluster_wide,cluster_transition=joint_dynamic_clusters(D["customers"],cluster_count)
customers=enrich_scores(D["customers"],cluster_wide)
cluster_col={y:(f"{y}군집_동적" if f"{y}군집_동적" in customers.columns else f"{y}군집") for y in (2024,2025)}

T1,T2,T3,T4,T5,T6,T7=st.tabs(["2개년 개요","고객별 요금 모니터링","고객별 진단·제어","군집·전이","요금·예측","실제 100가구 포트폴리오","방법론·한계"])

with T1:
    S=D["stats"];c=st.columns(5)
    c[0].metric("2개년 핵심고객",f"{S['2개년핵심고객수']:,}명")
    c[1].metric("2024 연평균",fmt_kwh(S["2024연평균kWh"]))
    c[2].metric("2025 연평균",fmt_kwh(S["2025연평균kWh"]),fmt_pct(S["연평균증감률"]))
    c[3].metric("동일 군집 유지",fmt_pct(S["군집유지율"]))
    c[4].metric("추천요금제 유지",fmt_pct(S["추천요금제유지율"]))
    om=D["overall_monthly"].copy();fig=px.line(om,x="월",y="고객당평균_kWh",color="연도",markers=True,labels={"고객당평균_kWh":"고객당 평균 사용량(kWh)"});fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kWh<extra></extra>");st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns([1.15,1])
    with left:
        mc=D["monthly_change"][["월","2024고객당평균_kWh","2025고객당평균_kWh","증감_kWh","증감률","경부하비중증감p","최대부하비중증감p"]].copy()
        mc["증감률(%)"]=mc.pop("증감률")*100;mc["경부하비중증감(%p)"]=mc.pop("경부하비중증감p")*100;mc["최대부하비중증감(%p)"]=mc.pop("최대부하비중증감p")*100
        mc=round_table(mc);st.dataframe(mc,hide_index=True,use_container_width=True,column_config=dataframe_config(mc))
    with right:
        bins=pd.cut(customers["연간사용량증감률"],[-np.inf,-0.2,-0.05,0.05,0.2,np.inf],labels=["20% 이상 감소","5~20% 감소","±5% 이내","5~20% 증가","20% 이상 증가"])
        dist=bins.value_counts(sort=False).rename_axis("구간").reset_index(name="고객수");dist["비중(%)"]=dist["고객수"]/len(customers)*100
        fig=px.bar(dist,x="구간",y="고객수",text="비중(%)");fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside");st.plotly_chart(fig,use_container_width=True)
    st.subheader("계절·주중/주말 평균 부하곡선")
    a,b=st.columns(2);season=a.selectbox("계절",list(SEASON_MONTHS),key="overview_season");daytype=b.radio("일 유형",["주중","주말"],horizontal=True,key="overview_day")
    pp=D["overall_profiles"][(D["overall_profiles"]["계절"]==season)&(D["overall_profiles"]["일유형"]==daytype)]
    fig=px.line(pp,x="시간",y="고객당평균_kWh",color="연도",markers=True);fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kWh/h<extra></extra>");st.plotly_chart(fig,use_container_width=True)

with T2:
    st.subheader("712명 고객별 요금 모니터링 및 추천")
    a,b,c,d=st.columns(4)
    period=a.selectbox("분석 기간",["연간 전체","월중 모니터링"])
    year=b.selectbox("분석 연도",[2024,2025],index=1)
    month=c.selectbox("분석 월",list(range(1,13)),index=7,disabled=period=="연간 전체")
    maxday=monthrange(int(year),int(month))[1];cutoff=d.slider("조회일",5,maxday-1,min(20,maxday-1),1,disabled=period=="연간 전체")
    e,f=st.columns(2);current_plan=e.selectbox("알림 기준 현재 요금제",["기본형","프리미엄형"]);sort_key=f.selectbox("정렬 기준",["수요관리우선점수","TOU대비절감(원)","월말예상(kWh)","연간사용량(kWh)"])
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
        c1,c2,c3,c4=st.columns(4);c1.metric("표시 고객",f"{len(show):,}명");c2.metric("기본형 추천",f"{(show['추천요금제']=='구독 기본형').sum():,}명");c3.metric("프리미엄형 추천",f"{(show['추천요금제']=='구독 프리미엄형').sum():,}명");c4.metric("평균 절감가능액",fmt_won1(show["TOU대비절감(원)"].mean() if len(show) else 0))
        st.dataframe(show,hide_index=True,use_container_width=True,height=520,column_config=dataframe_config(show))
        st.download_button("고객별 요금 모니터링 CSV",show.to_csv(index=False).encode("utf-8-sig"),f"v13_{year}_{period}_요금모니터링.csv","text/csv")

with T3:
    st.subheader("동일 고객의 종단진단과 목표관리·제어")
    cid=st.selectbox("고객 선택",customers["고객ID"].sort_values().tolist())
    r=customers.set_index("고객ID").loc[cid]
    c=st.columns(5);c[0].metric("2024 사용량",fmt_kwh(r["2024_연간사용량_kWh"]));c[1].metric("2025 사용량",fmt_kwh(r["2025_연간사용량_kWh"]),fmt_pct(r["연간사용량증감률"]));c[2].metric("패턴 안정성",f"{r['패턴안정성점수']:.1f}점");c[3].metric("수요관리 우선",f"{r['수요관리우선점수']:.1f}점");c[4].metric("구조변화",r["구조변화신호"])
    st.info(f"군집: {r[cluster_col[2024]]} → {r[cluster_col[2025]]} / 추천요금제: {r['2024_추천요금제']} → {r['2025_추천요금제']}")
    period=st.radio("진단 기간",["연간 종합진단","월별 목표관리·제어"],horizontal=True)
    if period=="연간 종합진단":
        cm=D["monthly"][D["monthly"]["고객ID"]==cid].copy();fig=px.line(cm,x="월",y="사용량_kWh",color="연도",markers=True);fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kWh<extra></extra>");st.plotly_chart(fig,use_container_width=True)
        rows=[]
        for y in (2024,2025):
            bills=annual_bill_map(cm[cm["연도"]==y],basic_fee,basic_inc,premium_fee,premium_inc,overage)
            for k,v in bills.items():rows.append({"연도":y,"요금제":k,"연간요금(원)":v})
        tdf=round_table(pd.DataFrame(rows));st.dataframe(tdf,hide_index=True,use_container_width=True,column_config=dataframe_config(tdf))
        st.subheader("월별 추천요금제 변화")
        mr=D["monthly_recommendation"][D["monthly_recommendation"]["고객ID"]==cid].copy();pivot=mr.pivot(index="월",columns="연도",values="추천요금제").reset_index().rename(columns={2024:"2024 추천",2025:"2025 추천"});pivot["변경여부"]=np.where(pivot["2024 추천"]==pivot["2025 추천"],"유지","변경");st.dataframe(pivot,hide_index=True,use_container_width=True)
        season=st.selectbox("대표 계절",list(SEASON_MONTHS),key="annual_diag_season");tabs=st.tabs(["주중","주말"])
        for tab,dt in zip(tabs,["주중","주말"]):
            with tab:
                arr=[]
                for y in (2024,2025):p=profile_for_customer(D["profiles"],cid,y,season,dt);p["연도"]=y;arr.append(p)
                pf=pd.concat(arr);fig=px.line(pf,x="시간",y="평균사용량_kWh",color="연도",markers=True);fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kWh/h<extra></extra>");st.plotly_chart(fig,use_container_width=True)
    else:
        a,b,c=st.columns(3);year=a.selectbox("대상 연도",[2024,2025],index=1,key="diag_year");month=b.selectbox("대상 월",list(range(1,13)),index=7,key="diag_month");cutoff=c.slider("조회일",5,monthrange(year,month)[1]-1,min(20,monthrange(year,month)[1]-1),1,key="diag_cutoff")
        a,b,c=st.columns(3);current_plan=a.selectbox("현재 구독요금제",["기본형","프리미엄형"]);management=b.radio("관리 방식",["알림·행동권고","한전 직접제어 위임"],horizontal=False);mode=c.selectbox("제어·권고 강도",list(CONTROL_MODES),index=1)
        ownership=st.multiselect("등록·연결된 기기",sorted({str(x["ownership"]) for x in ACTION_LIBRARY}),default=sorted({str(x["ownership"]) for x in ACTION_LIBRARY}))
        cd=D["daily"][D["daily"]["고객ID"]==cid];f=forecast_month_longitudinal(cd,year,month,cutoff);mrow=D["monthly"][(D["monthly"]["고객ID"]==cid)&(D["monthly"]["연도"]==year)&(D["monthly"]["월"]==month)].iloc[0]
        bills=monthly_bill_map(f["forecast"],month,mrow,basic_fee,basic_inc,premium_fee,premium_inc,overage);rec=min(bills,key=bills.get);inc=basic_inc if current_plan=="기본형" else premium_inc;fee=basic_fee if current_plan=="기본형" else premium_fee
        c=st.columns(6);c[0].metric("현재 누적",fmt_kwh(f["current"]));c[1].metric("남은 제공량",fmt_kwh(max(inc-f["current"],0)));c[2].metric("월말 예상",fmt_kwh(f["forecast"]));c[3].metric("예상 범위",f"{f['lower']:,.1f}~{f['upper']:,.1f}kWh");c[4].metric("현재요금제 예상",fmt_won1(subscription_bill(f["forecast"],fee,inc,overage)));c[5].metric("추천요금제",rec)
        target_options=["현재 요금제 제공량 이내","전월과 같은 요금","전년 동월과 같은 요금","목표요금 직접 입력"]
        target_kind=st.radio("관리 목표",target_options,horizontal=True)
        target_usage=inc
        if target_kind=="전월과 같은 요금":
            py,pm=(year-1,12) if month==1 else (year,month-1);prev=D["monthly"][(D["monthly"]["고객ID"]==cid)&(D["monthly"]["연도"]==py)&(D["monthly"]["월"]==pm)]
            if len(prev):target_bill=subscription_bill(float(prev.iloc[0]["사용량_kWh"]),fee,inc,overage);target_usage=inverse_subscription_bill(target_bill,fee,inc,overage)
        elif target_kind=="전년 동월과 같은 요금":
            prev=D["monthly"][(D["monthly"]["고객ID"]==cid)&(D["monthly"]["연도"]==year-1)&(D["monthly"]["월"]==month)]
            if len(prev):target_bill=subscription_bill(float(prev.iloc[0]["사용량_kWh"]),fee,inc,overage);target_usage=inverse_subscription_bill(target_bill,fee,inc,overage)
            else:st.warning("전년 동월 자료가 없어 제공량 목표를 적용합니다.")
        elif target_kind=="목표요금 직접 입력":
            target_bill=st.number_input("목표 월 납부액(원)",0,1_000_000,int(subscription_bill(inc,fee,inc,overage)),1000);target_usage=inverse_subscription_bill(target_bill,fee,inc,overage)
        required=max(f["forecast"]-target_usage,0);plan,gross,effective=optimize_actions(required,f["remaining_days"],season_for_month(month),ownership,mode,direct=management=="한전 직접제어 위임")
        c=st.columns(4);c[0].metric("목표 사용량",fmt_kwh(target_usage));c[1].metric("필요 감축량",fmt_kwh(required));c[2].metric("계획 실효감축",fmt_kwh(effective));c[3].metric("제어 후 예상",fmt_kwh(max(f["forecast"]-effective,f["current"])))
        if not plan.empty:
            ps=round_table(plan);st.dataframe(ps,hide_index=True,use_container_width=True,column_config=dataframe_config(ps));st.download_button("행동·제어계획 CSV",ps.to_csv(index=False).encode("utf-8-sig"),f"{cid}_{year}_{month}월_제어계획.csv","text/csv")
        else:st.info("현재 목표를 위해 추가로 선택할 수 있는 행동대안이 없거나 감축이 필요하지 않습니다.")
        dm=cd[(cd["연도"]==year)&(cd["월"]==month)];proj=cumulative_projection(dm,cutoff,f["forecast"],effective*(0.8 if management=="알림·행동권고" else 0.0),effective if management=="한전 직접제어 위임" else 0.0)
        fig=go.Figure()
        for col in ["실제누적","미제어예상","행동권고예상","직접제어예상"]:fig.add_trace(go.Scatter(x=proj["일"],y=proj[col],name=col,mode="lines+markers"))
        fig.add_hline(y=target_usage,line_dash="dash",annotation_text="목표 사용량");fig.update_layout(xaxis_title="일",yaxis_title="월 누적 사용량(kWh)");fig.update_traces(hovertemplate="%{y:,.1f}kWh<extra></extra>");st.plotly_chart(fig,use_container_width=True)
        st.subheader("주중·주말 평균 부하곡선: 관리 전후")
        tabs=st.tabs(["주중","주말"])
        for tab,dt in zip(tabs,["주중","주말"]):
            with tab:
                p=D["profiles"][(D["profiles"]["고객ID"]==cid)&(D["profiles"]["연도"]==year)&(D["profiles"]["월"]==month)&(D["profiles"]["일유형"]==dt)].sort_values("시간");base=p["평균사용량_kWh"].to_numpy(float);after=controlled_profile(base,plan,f["remaining_days"])
                fig=go.Figure();fig.add_trace(go.Scatter(x=p["시간"],y=base,name="관리 전",mode="lines+markers"));fig.add_trace(go.Scatter(x=p["시간"],y=after,name="관리 후 예상",mode="lines+markers"));fig.update_xaxes(dtick=1);fig.update_yaxes(title="평균부하(kWh/h)");fig.update_traces(hovertemplate="%{y:,.1f}<extra></extra>");st.plotly_chart(fig,use_container_width=True)

with T4:
    st.subheader(f"공통 기준 {cluster_count}개 군집과 2024→2025 전이")
    cs=cluster_summary.copy();cs["비중(%)"]=cs.pop("비중")*100;cs=round_table(cs);st.dataframe(cs.sort_values(["연도","고객수"],ascending=[True,False]),hide_index=True,use_container_width=True,column_config=dataframe_config(cs))
    stability=(cluster_wide["군집유지여부"]=="유지").mean();c=st.columns(3);c[0].metric("동일 군집 유지율",fmt_pct(stability));c[1].metric("군집 이동 고객",f"{(cluster_wide['군집유지여부']=='이동').sum():,}명");c[2].metric("군집 수",f"{cluster_count}개")
    matrix=cluster_transition.pivot(index="2024군집",columns="2025군집",values="고객수").fillna(0);fig=px.imshow(matrix,text_auto=True,aspect="auto",color_continuous_scale="Blues");st.plotly_chart(fig,use_container_width=True)
    tt=cluster_transition.copy();tt["2024군집내비중(%)"]=tt.pop("2024군집내비중")*100;tt=round_table(tt);st.dataframe(tt,hide_index=True,use_container_width=True,column_config=dataframe_config(tt))

with T5:
    st.subheader("추천요금제 안정성과 예측성능")
    ts=D["tariff_summary"].copy();ts["최저추천비중(%)"]=ts.pop("최저추천비중")*100;ts=round_table(ts);st.dataframe(ts,hide_index=True,use_container_width=True,column_config=dataframe_config(ts))
    fig=px.bar(ts,x="요금제",y="최저추천고객수",color="연도",barmode="group",text="최저추천고객수");st.plotly_chart(fig,use_container_width=True)
    tr=D["tariff_transition"].copy();tr["2024추천군내비중(%)"]=tr.pop("2024추천군내비중")*100;tr=round_table(tr);st.dataframe(tr,hide_index=True,use_container_width=True,column_config=dataframe_config(tr))
    ff=customers[["고객ID","15일예측_MAPE","20일예측_MAPE","15일예측_오차10%이내","20일예측_오차10%이내"]].copy();ff["15일예측_MAPE(%)"]=ff.pop("15일예측_MAPE")*100;ff["20일예측_MAPE(%)"]=ff.pop("20일예측_MAPE")*100;ff["15일예측_오차10%이내(%)"]=ff.pop("15일예측_오차10%이내")*100;ff["20일예측_오차10%이내(%)"]=ff.pop("20일예측_오차10%이내")*100;ff=round_table(ff);st.dataframe(ff,hide_index=True,use_container_width=True,height=420,column_config=dataframe_config(ff))

with T6:
    st.subheader("동일 고객 100가구 포트폴리오")
    c1,c2,c3,c4=st.columns(4);seed=c1.number_input("표본 추출번호",0,9999,42,1);season=c2.selectbox("계절",list(SEASON_MONTHS),key="portfolio_season");daytype=c3.radio("일 유형",["주중","주말"],horizontal=True,key="portfolio_day");target_year=c4.selectbox("제어 적용연도",[2024,2025],index=1)
    rng=np.random.default_rng(int(seed));ids=rng.choice(customers["고객ID"].to_numpy(),size=100,replace=False).tolist();base24=aggregate_portfolio_profile(D["profiles"],ids,2024,season,daytype);base25=aggregate_portfolio_profile(D["profiles"],ids,2025,season,daytype)
    ptable=customers[customers["고객ID"].isin(ids)][["고객ID","2024_연간사용량_kWh","2025_연간사용량_kWh","연간사용량증감률",cluster_col[2024],cluster_col[2025],"수요관리우선점수"]].copy();ptable["연간사용량증감률(%)"]=ptable.pop("연간사용량증감률")*100
    c=st.columns(4);c[0].metric("2024 연간 합계",f"{ptable['2024_연간사용량_kWh'].sum()/1000:,.1f}MWh");c[1].metric("2025 연간 합계",f"{ptable['2025_연간사용량_kWh'].sum()/1000:,.1f}MWh",fmt_pct(ptable['2025_연간사용량_kWh'].sum()/ptable['2024_연간사용량_kWh'].sum()-1));c[2].metric("2024 대표일 피크",f"{base24.max():,.1f}kW");c[3].metric("2025 대표일 피크",f"{base25.max():,.1f}kW",fmt_pct(base25.max()/max(base24.max(),1e-9)-1))
    fig=go.Figure();fig.add_trace(go.Scatter(x=np.arange(1,25),y=base24,name="2024 제어 전"));fig.add_trace(go.Scatter(x=np.arange(1,25),y=base25,name="2025 제어 전"));fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kW<extra></extra>");st.plotly_chart(fig,use_container_width=True)
    a,b=st.columns(2);limit_pct=a.slider("변압기 목표 운전한도(제어 전 최대부하 대비)",60,110,90,1);participation_pct=b.slider("직접제어 참여율",0,100,70,5);base=base24 if target_year==2024 else base25;result=optimize_transformer_profile(base,limit_pct/100,participation_pct/100)
    c=st.columns(6);c[0].metric("제어 전 피크",f"{result['peak_before']:,.1f}kW");c[1].metric("목표 운전한도",f"{result['limit']:,.1f}kW");c[2].metric("제어 후 피크",f"{result['peak_after']:,.1f}kW");c[3].metric("한도 초과시간",f"{result['hours_before']}→{result['hours_after']}시간");c[4].metric("시간이동량",f"{result['shifted']:,.1f}kWh");c[5].metric("실제 감축량",f"{result['reduced']:,.1f}kWh")
    control=pd.DataFrame({"시간":np.arange(1,25),"제어전(kW)":base,"시간이동출력(kW)":result["shift_out"],"실제감축출력(kW)":result["reduction"],"이동유입(kW)":result["shift_in"],"제어후(kW)":result["after"],"운전한도(kW)":result["limit"]});fig=go.Figure();fig.add_trace(go.Scatter(x=control["시간"],y=control["제어전(kW)"],name="제어 전"));fig.add_trace(go.Scatter(x=control["시간"],y=control["제어후(kW)"],name="제어 후"));fig.add_trace(go.Scatter(x=control["시간"],y=control["운전한도(kW)"],name="운전한도",line=dict(dash="dash")));fig.update_xaxes(dtick=1);fig.update_traces(hovertemplate="%{y:,.1f}kW<extra></extra>");st.plotly_chart(fig,use_container_width=True)
    ps=round_table(ptable);st.dataframe(ps.sort_values("수요관리우선점수",ascending=False),hide_index=True,use_container_width=True,column_config=dataframe_config(ps));st.download_button("100가구 분석결과 ZIP",zip_results({"100가구_고객목록.csv":ps.to_csv(index=False).encode("utf-8-sig"),"100가구_변압기제어상세.csv":round_table(control).to_csv(index=False).encode("utf-8-sig")}),"v13_100가구_포트폴리오.zip","application/zip")

with T7:
    st.markdown("""
### v13 핵심 변경점
- v11의 **고객별 요금 모니터링**과 **고객별 진단·제어** 기능을 2024·2025년 공통 핵심고객 712명에 적용했습니다.
- 월중 모니터링은 실제 일별 총계량값을 사용하며, 2025년 예측에는 2024년 동일 월 주중·주말 패턴을 보조정보로 활용합니다.
- 고객은 목표 사용량·전월 요금·전년 동월 요금·직접 입력 목표요금 중 하나를 선택할 수 있습니다.
- 행동권고와 한전 직접제어 위임을 구분하며, 가전별 대안은 고객이 등록·연결한 기기를 전제로 한 시나리오입니다.
- 모든 표의 연속형 수치는 소수점 첫째자리까지만 표시하고, 비중·증감률·예측오차·사용률은 % 단위로 표시합니다.

### 유의사항
- 가전별 사용량은 단일 계량값으로 직접 관측한 것이 아닙니다. 제어계획은 등록기기와 표준 절감 라이브러리를 결합한 추정치입니다.
- 요금은 2026년 6월 요금표를 2024·2025년 사용량에 적용한 비교 시나리오입니다.
- 100가구 포트폴리오는 실제 동일 변압기 연결고객이 아니며, 변압기 한도는 제어 전 피크의 가상 비율입니다.
""")
    st.subheader("사용 데이터 파일");st.code("\n".join(DATA_FILES.values()))
