from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ortools.sat.python import cp_model

APP_VERSION = "2026-07-20-actual-tou-v12.0"
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


st.set_page_config(page_title="제주 TOU 2개년 종단분석", page_icon="⚡", layout="wide")
st.title("제주 TOU 공통고객 2개년 종단분석·수요관리 시뮬레이터")
st.caption(f"앱 버전 {APP_VERSION} · 2024~2025년 고객번호 일치 및 2개년 완전자료 기준 712명")

try:
    D = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("분석 설정")
    cluster_count = st.slider("공통 군집 수", 3, 8, 8, 1, help="두 해를 같은 군집 중심에 배치하므로 연도 간 이동을 비교할 수 있습니다.")
    st.divider()
    st.header("요금 가정")
    basic_fee = st.number_input("기본형 월 구독료(원)", 0, 500_000, 84_900, 1_000)
    basic_inc = st.number_input("기본형 제공량(kWh)", 0, 3_000, 450, 10)
    premium_fee = st.number_input("프리미엄형 월 구독료(원)", 0, 800_000, 249_900, 1_000)
    premium_inc = st.number_input("프리미엄형 제공량(kWh)", 0, 5_000, 1_000, 10)
    overage = st.selectbox("초과단가(원/kWh)", [200, 300, 307.3, 400], index=1)
    st.info("요금표는 2026년 6월 시행 단가를 2024·2025년 사용량에 동일 적용한 비교 시나리오입니다.")

stacked_cluster, cluster_summary, cluster_wide, cluster_transition = joint_dynamic_clusters(D["customers"], cluster_count)
customers = enrich_scores(D["customers"], cluster_wide)

T1, T2, T3, T4, T5, T6 = st.tabs([
    "2개년 개요", "고객별 종단진단", "군집·전이", "요금·예측", "실제 100가구 포트폴리오", "방법론·한계"
])

with T1:
    S = D["stats"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("2개년 핵심고객", f"{S['2개년핵심고객수']:,}명")
    c2.metric("2024 연평균", fmt_kwh(S["2024연평균kWh"]))
    c3.metric("2025 연평균", fmt_kwh(S["2025연평균kWh"]), f"{S['연평균증감률']:.1%}")
    c4.metric("동일 군집 유지", f"{S['군집유지율']:.1%}")
    c5.metric("추천요금제 유지", f"{S['추천요금제유지율']:.1%}")

    st.subheader("월별 고객당 평균 사용량")
    om = D["overall_monthly"].copy()
    fig = px.line(om, x="월", y="고객당평균_kWh", color="연도", markers=True,
                  labels={"고객당평균_kWh": "고객당 평균 사용량(kWh)"})
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.subheader("월별 전년 대비 변화")
        mc = D["monthly_change"].copy()
        mc_show = mc[["월", "2024고객당평균_kWh", "2025고객당평균_kWh", "증감_kWh", "증감률", "경부하비중증감p", "최대부하비중증감p"]].copy()
        mc_show["증감률(%)"] = mc_show.pop("증감률") * 100
        mc_show["경부하비중증감(%p)"] = mc_show.pop("경부하비중증감p") * 100
        mc_show["최대부하비중증감(%p)"] = mc_show.pop("최대부하비중증감p") * 100
        st.dataframe(mc_show, hide_index=True, use_container_width=True)
    with c2:
        st.subheader("고객별 증감 분포")
        bins = pd.cut(customers["연간사용량증감률"], [-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf],
                      labels=["20% 이상 감소", "5~20% 감소", "±5% 이내", "5~20% 증가", "20% 이상 증가"])
        dist = bins.value_counts(sort=False).rename_axis("구간").reset_index(name="고객수")
        dist["비중"] = dist["고객수"] / len(customers) * 100
        fig2 = px.bar(dist, x="구간", y="고객수", text="비중")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("계절·주중/주말 평균 부하곡선")
    season = st.selectbox("계절", list(SEASON_MONTHS), key="overview_season")
    daytype = st.radio("일 유형", ["주중", "주말"], horizontal=True, key="overview_day")
    op = D["overall_profiles"]
    pp = op[(op["계절"] == season) & (op["일유형"] == daytype)]
    fig3 = px.line(pp, x="시간", y="고객당평균_kWh", color="연도", markers=True,
                   labels={"고객당평균_kWh": "고객당 평균부하(kWh/h)"})
    fig3.update_xaxes(dtick=1)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("심화 분석용 고객 모니터링 표")
    show_cols = ["고객ID", "2024_연간사용량_kWh", "2025_연간사용량_kWh", "연간사용량증감률",
                 "2024군집", "2025군집", "패턴안정성점수", "수요관리우선점수", "구조변화신호",
                 "2024_추천요금제", "2025_추천요금제"]
    monitor = customers[show_cols].copy()
    monitor["연간사용량증감률(%)"] = monitor.pop("연간사용량증감률") * 100
    st.dataframe(monitor.sort_values("수요관리우선점수", ascending=False), hide_index=True, use_container_width=True, height=420)
    st.download_button("2개년 고객 모니터링 CSV", monitor.to_csv(index=False).encode("utf-8-sig"), "v12_고객모니터링.csv", "text/csv")

with T2:
    st.subheader("동일 고객의 2024→2025 변화 진단")
    cid = st.selectbox("고객 선택", customers["고객ID"].tolist(), index=0)
    r = customers.set_index("고객ID").loc[cid]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("2024 사용량", fmt_kwh(r["2024_연간사용량_kWh"]))
    c2.metric("2025 사용량", fmt_kwh(r["2025_연간사용량_kWh"]), f"{r['연간사용량증감률']:.1%}")
    c3.metric("패턴 안정성", f"{r['패턴안정성점수']:.1f}점")
    c4.metric("수요관리 우선", f"{r['수요관리우선점수']:.1f}점")
    c5.metric("구조변화", r["구조변화신호"])

    st.info(f"군집: {r['2024군집']} → {r['2025군집']} / 추천요금제: {r['2024_추천요금제']} → {r['2025_추천요금제']}")

    cm = D["monthly"][D["monthly"]["고객ID"] == cid].copy()
    fig = px.line(cm, x="월", y="사용량_kWh", color="연도", markers=True,
                  labels={"사용량_kWh": "월사용량(kWh)"})
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("연간 요금 비교")
        tariff_cols = [
            "일반 주택용_연간요금원", "제주 TOU(3kW)_연간요금원", "구독 기본형_연간요금원", "구독 프리미엄형_연간요금원"
        ]
        rows = []
        for y in (2024, 2025):
            for col in tariff_cols:
                rows.append({"연도": y, "요금제": col.replace("_연간요금원", ""), "연간요금(원)": r[f"{y}_{col}"]})
        tdf = pd.DataFrame(rows)
        st.plotly_chart(px.bar(tdf, x="요금제", y="연간요금(원)", color="연도", barmode="group"), use_container_width=True)
        st.dataframe(tdf.pivot(index="요금제", columns="연도", values="연간요금(원)").reset_index(), hide_index=True, use_container_width=True)
    with right:
        st.subheader("월말 예측 검증")
        fdf = pd.DataFrame([
            {"조회시점": "15일", "MAE(kWh)": r["15일예측_MAE_kWh"], "MAPE(%)": r["15일예측_MAPE"] * 100, "10% 이내 월비중(%)": r["15일예측_오차10%이내"] * 100, "평균편향(kWh)": r["15일예측_평균편향_kWh"]},
            {"조회시점": "20일", "MAE(kWh)": r["20일예측_MAE_kWh"], "MAPE(%)": r["20일예측_MAPE"] * 100, "10% 이내 월비중(%)": r["20일예측_오차10%이내"] * 100, "평균편향(kWh)": r["20일예측_평균편향_kWh"]},
        ])
        st.dataframe(fdf, hide_index=True, use_container_width=True)
        st.caption("2024년 동일 월의 주중·주말 패턴과 2025년 월중 실적을 결합한 사후검증 결과입니다.")

    st.subheader("시간대 부하곡선 변화")
    season2 = st.selectbox("계절", list(SEASON_MONTHS), key="customer_season")
    day2 = st.radio("일 유형", ["주중", "주말"], horizontal=True, key="customer_day")
    pp = []
    for y in (2024, 2025):
        p = profile_for_customer(D["profiles"], cid, y, season2, day2)
        p["연도"] = y
        pp.append(p)
    p2 = pd.concat(pp, ignore_index=True)
    figp = px.line(p2, x="시간", y="평균사용량_kWh", color="연도", markers=True,
                   labels={"평균사용량_kWh": "평균부하(kWh/h)"})
    figp.update_xaxes(dtick=1)
    st.plotly_chart(figp, use_container_width=True)

    st.subheader("월별 요금제 추천 변화")
    mr = D["monthly_recommendation"][D["monthly_recommendation"]["고객ID"] == cid].copy()
    pivot = mr.pivot(index="월", columns="연도", values="추천요금제").reset_index().rename(columns={2024: "2024 추천", 2025: "2025 추천"})
    pivot["변경여부"] = np.where(pivot["2024 추천"] == pivot["2025 추천"], "유지", "변경")
    st.dataframe(pivot, hide_index=True, use_container_width=True)

with T3:
    st.subheader(f"공통 기준 {cluster_count}개 군집과 2024→2025 전이")
    cs = cluster_summary.copy()
    cs["비중(%)"] = cs.pop("비중") * 100
    st.dataframe(cs.sort_values(["연도", "고객수"], ascending=[True, False]), hide_index=True, use_container_width=True)

    stability = (cluster_wide["군집유지여부"] == "유지").mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("동일 군집 유지율", f"{stability:.1%}")
    c2.metric("군집 이동 고객", f"{(cluster_wide['군집유지여부'] == '이동').sum():,}명")
    c3.metric("군집 수", f"{cluster_count}개")

    matrix = cluster_transition.pivot(index="2024군집", columns="2025군집", values="고객수").fillna(0)
    fig = px.imshow(matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues", labels={"color": "고객수"})
    fig.update_layout(xaxis_title="2025 군집", yaxis_title="2024 군집")
    st.plotly_chart(fig, use_container_width=True)

    tshow = cluster_transition.copy()
    tshow["2024군집내비중(%)"] = tshow.pop("2024군집내비중") * 100
    st.dataframe(tshow.sort_values(["2024군집", "고객수"], ascending=[True, False]), hide_index=True, use_container_width=True)

    st.subheader("군집 이동 원인 후보")
    move = customers[customers["군집유지여부_동적"] == "이동"].copy()
    reason = move.groupby("구조변화신호").size().rename("고객수").reset_index()
    st.plotly_chart(px.bar(reason, x="구조변화신호", y="고객수", text="고객수"), use_container_width=True)

with T4:
    st.subheader("연간 추천요금제와 안정성")
    ts = D["tariff_summary"].copy()
    ts["최저추천비중(%)"] = ts.pop("최저추천비중") * 100
    st.dataframe(ts, hide_index=True, use_container_width=True)
    fig = px.bar(ts, x="요금제", y="최저추천고객수", color="연도", barmode="group", text="최저추천고객수")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("연간 추천요금제 전이")
    ttrans = D["tariff_transition"].copy()
    ttrans["2024추천군내비중(%)"] = ttrans.pop("2024추천군내비중") * 100
    st.dataframe(ttrans, hide_index=True, use_container_width=True)

    st.subheader("월별 추천요금제 안정성")
    mr = D["monthly_recommendation"]
    p = mr.pivot_table(index=["고객ID", "월"], columns="연도", values="추천요금제", aggfunc="first").reset_index()
    p["월추천유지"] = p[2024] == p[2025]
    stability_by_customer = p.groupby("고객ID")["월추천유지"].mean().rename("월별추천유지율").reset_index()
    stability_by_customer["월별추천유지율(%)"] = stability_by_customer.pop("월별추천유지율") * 100
    merged = customers[["고객ID", "2024_추천요금제", "2025_추천요금제", "2024_TOU대비최대절감원", "2025_TOU대비최대절감원", "20일예측_MAPE", "수요관리우선점수"]].merge(stability_by_customer, on="고객ID")
    merged["20일예측_MAPE(%)"] = merged.pop("20일예측_MAPE") * 100
    st.dataframe(merged.sort_values("수요관리우선점수", ascending=False), hide_index=True, use_container_width=True, height=420)

    st.subheader("예측 성능 분포")
    ff = customers[["고객ID", "15일예측_MAPE", "20일예측_MAPE", "15일예측_오차10%이내", "20일예측_오차10%이내"]].copy()
    long = ff.melt(id_vars="고객ID", value_vars=["15일예측_MAPE", "20일예측_MAPE"], var_name="조회시점", value_name="MAPE")
    long["MAPE(%)"] = long["MAPE"] * 100
    figf = px.histogram(long, x="MAPE(%)", color="조회시점", barmode="overlay", nbins=40)
    st.plotly_chart(figf, use_container_width=True)
    st.caption("MAPE는 고객별 12개월 평균입니다. 이상적으로는 2024년 외에 기상·고객특성을 추가해 개선해야 합니다.")

with T5:
    st.subheader("동일 고객 100가구 포트폴리오")
    c1, c2, c3, c4 = st.columns(4)
    seed = c1.number_input("표본 추출번호", 0, 9999, 42, 1)
    season = c2.selectbox("계절", list(SEASON_MONTHS), key="portfolio_season")
    daytype = c3.radio("일 유형", ["주중", "주말"], horizontal=True, key="portfolio_day")
    target_year = c4.selectbox("제어 적용연도", [2024, 2025], index=1)
    rng = np.random.default_rng(int(seed))
    ids = rng.choice(customers["고객ID"].to_numpy(), size=100, replace=False).tolist()
    base24 = aggregate_portfolio_profile(D["profiles"], ids, 2024, season, daytype)
    base25 = aggregate_portfolio_profile(D["profiles"], ids, 2025, season, daytype)

    ptable = customers[customers["고객ID"].isin(ids)][["고객ID", "2024_연간사용량_kWh", "2025_연간사용량_kWh", "연간사용량증감률", "2024군집", "2025군집", "수요관리우선점수"]].copy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2024 연간 합계", f"{ptable['2024_연간사용량_kWh'].sum()/1000:,.1f}MWh")
    c2.metric("2025 연간 합계", f"{ptable['2025_연간사용량_kWh'].sum()/1000:,.1f}MWh", f"{ptable['2025_연간사용량_kWh'].sum()/ptable['2024_연간사용량_kWh'].sum()-1:.1%}")
    c3.metric("2024 대표일 피크", f"{base24.max():,.1f}kW")
    c4.metric("2025 대표일 피크", f"{base25.max():,.1f}kW", f"{base25.max()/max(base24.max(),1e-9)-1:.1%}")

    compare = pd.DataFrame({"시간": np.arange(1, 25), "2024": base24, "2025": base25})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=compare["시간"], y=compare["2024"], name="2024 제어 전"))
    fig.add_trace(go.Scatter(x=compare["시간"], y=compare["2025"], name="2025 제어 전"))
    fig.update_layout(xaxis_title="시간", yaxis_title="100가구 합계부하(kW)")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("변압기 목표 운전한도 기반 직접제어")
    c1, c2 = st.columns(2)
    limit_pct = c1.slider("변압기 목표 운전한도(제어 전 최대부하 대비)", 60, 110, 90, 1)
    participation_pct = c2.slider("직접제어 참여율", 0, 100, 70, 5)
    base = base24 if target_year == 2024 else base25
    result = optimize_transformer_profile(base, limit_pct / 100.0, participation_pct / 100.0)
    cc = st.columns(6)
    cc[0].metric("제어 전 피크", f"{result['peak_before']:,.1f}kW")
    cc[1].metric("목표 운전한도", f"{result['limit']:,.1f}kW")
    cc[2].metric("제어 후 피크", f"{result['peak_after']:,.1f}kW")
    cc[3].metric("한도 초과시간", f"{result['hours_before']}→{result['hours_after']}시간")
    cc[4].metric("시간이동량", f"{result['shifted']:,.1f}kWh")
    cc[5].metric("실제 감축량", f"{result['reduced']:,.1f}kWh")
    if result["status"] == "운전한도 충족":
        st.success(result["status"])
    elif "완화" in result["status"]:
        st.warning(result["status"])
    else:
        st.info(result["status"])

    control_df = pd.DataFrame({
        "시간": np.arange(1, 25), "제어전(kW)": base, "시간이동출력(kW)": result["shift_out"],
        "실제감축출력(kW)": result["reduction"], "이동유입(kW)": result["shift_in"],
        "제어후(kW)": result["after"], "운전한도(kW)": result["limit"],
    })
    figc = go.Figure()
    figc.add_trace(go.Scatter(x=control_df["시간"], y=control_df["제어전(kW)"], name="제어 전"))
    figc.add_trace(go.Scatter(x=control_df["시간"], y=control_df["제어후(kW)"], name="제어 후"))
    figc.add_trace(go.Scatter(x=control_df["시간"], y=control_df["운전한도(kW)"], name="운전한도", line=dict(dash="dash")))
    figc.update_xaxes(dtick=1)
    figc.update_layout(xaxis_title="시간", yaxis_title="100가구 합계부하(kW)")
    st.plotly_chart(figc, use_container_width=True)

    st.dataframe(ptable.sort_values("수요관리우선점수", ascending=False), hide_index=True, use_container_width=True, height=360)
    files = {
        "100가구_고객목록.csv": ptable.to_csv(index=False).encode("utf-8-sig"),
        "100가구_변압기제어상세.csv": control_df.to_csv(index=False).encode("utf-8-sig"),
    }
    st.download_button("100가구 분석결과 ZIP", zip_results(files), "v12_100가구_포트폴리오.zip", "application/zip")
    st.caption("동일한 100명의 2024·2025 부하를 비교하지만, 실제로 같은 변압기에 연결된 고객군이라는 의미는 아닙니다.")

with T6:
    st.markdown("""
### v12의 분석 범위
- 고객번호가 일치하고 2024년 366일·2025년 365일, 양 연도 시간값 완전도 99.5% 이상인 712명만 사용합니다.
- 동일 고객의 연간·월별 사용량, TOU 시간대 비중, 계절 부하곡선, 군집 이동, 추천요금제 변화와 예측성능을 분석합니다.
- 군집 수는 3~8개로 조정하며, 두 해를 동일한 군집 중심에 배치해 전이를 비교합니다.
- 패턴 안정성점수는 사용량·TOU 비중·주말/주중 비율·부하율 변화의 합성지표입니다.
- 수요관리 우선점수는 2025년 최대시간부하, 최대부하시간대 비중, 연간사용량, 계절민감도와 예측가능성을 결합한 시뮬레이션용 순위점수입니다.

### 요금 분석 주의사항
- 2024·2025년 실제 청구액을 재현한 것이 아니라 2026년 6월 시행 요금표를 동일 적용한 비교 시나리오입니다.
- 제주 TOU 계약전력은 3kW로 가정하며, 기후환경요금과 연료비조정요금은 제외합니다.
- 기본형과 프리미엄형의 초과단가가 같으면 프리미엄형의 가격 경쟁력이 제한될 수 있습니다.

### 계통운영 주의사항
- 변압기 운전한도는 선택된 100가구 대표일 최대부하에 대한 가상 비율입니다.
- 실제 고객-변압기 연결정보, 변압기 정격 kVA, 역률, 온도, 선로 전압·전류는 반영하지 않습니다.
- 시간이동 최대 14%, 실제 감축 최대 6%라는 유연성 가정은 직접제어 참여율에 비례합니다.
- 따라서 결과는 실제 과부하 방지를 보장하는 DMS 결과가 아니라 수요유연성의 개념검증입니다.
""")
    st.subheader("사용 데이터 파일")
    st.code("\n".join(DATA_FILES.values()))
