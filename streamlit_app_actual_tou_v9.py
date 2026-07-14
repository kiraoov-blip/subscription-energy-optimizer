from __future__ import annotations

import io
import json
import math
import os
import zipfile
from calendar import monthrange
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ortools.sat.python import cp_model

APP_VERSION = "2026-07-14-actual-tou-v9.0"
BASE_DIR = Path(__file__).resolve().parent

DATA_FILES = {
    "summary": "tou_v8_summary.json",
    "metrics": "tou_v8_customer_metrics.csv",
    "daily": "tou_v8_daily_usage.csv.gz",
    "profiles": "tou_v8_customer_profiles.csv.gz",
    "monthly": "tou_v8_monthly_tou.csv.gz",
}

PLAN_DEFAULTS = {
    "기본형": {"fee": 84_900.0, "included": 450.0},
    "프리미엄형": {"fee": 249_900.0, "included": 1_000.0},
}

# 최초 4인 가구 가전자료를 참고해 만든 행동대안 라이브러리입니다.
# 특정 가전의 실제 사용을 계량값만으로 식별했다는 의미는 아닙니다.
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
    if month in (6,7,8): return "여름"
    if month in (1,2,11,12): return "겨울"
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
    energy=kwh*(off_share*rates[0]+mid_share*rates[1]+peak_share*rates[2])
    return float((4310*contract_kw+energy)*1.127)


def subscription_bill(kwh: float, fee: float, included: float, overage: float) -> float:
    return float(fee+max(float(kwh)-float(included),0.0)*float(overage))


def inverse_subscription_bill(target_bill: float, fee: float, included: float, overage: float) -> float:
    if target_bill<=fee or overage<=0: return float(included)
    return float(included+(target_bill-fee)/overage)


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[dict,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    missing=[fn for fn in DATA_FILES.values() if not (BASE_DIR/fn).exists()]
    if missing:
        raise FileNotFoundError("필요한 데이터 파일이 없습니다: "+", ".join(missing))
    with open(BASE_DIR/DATA_FILES["summary"],encoding="utf-8") as f: summary=json.load(f)
    metrics=pd.read_csv(BASE_DIR/DATA_FILES["metrics"])
    daily=pd.read_csv(BASE_DIR/DATA_FILES["daily"],compression="gzip",parse_dates=["날짜"])
    profiles=pd.read_csv(BASE_DIR/DATA_FILES["profiles"],compression="gzip")
    monthly=pd.read_csv(BASE_DIR/DATA_FILES["monthly"],compression="gzip")
    return summary,metrics,daily,profiles,monthly


def forecast_month(customer_daily: pd.DataFrame, month: int, cutoff_day: int) -> Dict[str,float]:
    dm=customer_daily[customer_daily["월"]==month].sort_values("일")
    observed=dm[dm["일"]<=cutoff_day]
    remaining=dm[dm["일"]>cutoff_day]
    actual_total=float(dm["일사용량_kWh"].sum())
    current=float(observed["일사용량_kWh"].sum())
    days_in=int(dm["일"].max()) if len(dm) else monthrange(2025,month)[1]
    # Personalized weekday/weekend forecast using only observations available at the cutoff.
    overall=float(observed["일사용량_kWh"].mean()) if len(observed) else 0.0
    means=observed.groupby("일유형")["일사용량_kWh"].mean().to_dict()
    pred_remaining=0.0
    for _,r in remaining.iterrows(): pred_remaining+=float(means.get(r["일유형"],overall))
    forecast=current+pred_remaining
    daily_std=float(observed["일사용량_kWh"].std(ddof=0)) if len(observed)>1 else 0.0
    uncertainty=1.28*daily_std*math.sqrt(max(len(remaining),1))
    return {"current":current,"forecast":forecast,"lower":max(current,forecast-uncertainty),"upper":forecast+uncertainty,"actual":actual_total,"remaining_days":len(remaining),"days_in_month":days_in,"observed_days":len(observed)}


def alert_level(current: float, forecast: float, included: float) -> str:
    used=current/max(included,1e-9); projected=forecast/max(included,1e-9)
    if current>=included or used>=0.95 or projected>=1.25: return "긴급"
    if used>=0.85 or projected>=1.10: return "경고"
    if used>=0.70 or projected>1.00: return "주의"
    if used>=0.50 or projected>=0.90: return "관심"
    return "정상"


def optimize_actions(required_kwh: float, remaining_days: int, season: str, ownership: List[str], mode: str, direct: bool=False) -> Tuple[pd.DataFrame,float,float]:
    """Select reduction actions for a monthly-kWh target and add separate peak-shift actions.

    Time-shift actions never count as monthly energy reduction.
    """
    if remaining_days<=0:
        return pd.DataFrame(columns=["대안","유형","실행횟수","예상절감·이동량(kWh)","실효량(kWh)","불편점수"]),0.0,0.0
    reduction_candidates=[]
    shift_candidates=[]
    for a in ACTION_LIBRARY:
        if a["ownership"] not in ownership: continue
        if "seasons" in a and season not in a["seasons"]: continue
        max_count=int(a.get("daily_max",0)*remaining_days or a.get("weekly_max",0)*math.ceil(remaining_days/7))
        if max_count<=0: continue
        avg=(float(a["low"])+float(a["high"]))/2
        delivery=CONTROL_MODES[mode]["delivery"] if direct else float(a["reliability"])
        item=(a,max_count,avg,delivery)
        (shift_candidates if a["kind"]=="shift" else reduction_candidates).append(item)
    rows=[]; gross_reduction=0.0; effective_reduction=0.0
    if required_kwh>0 and reduction_candidates:
        model=cp_model.CpModel(); vars=[]; scale=1000
        for i,(a,mx,avg,delivery) in enumerate(reduction_candidates): vars.append(model.NewIntVar(0,mx,f"x{i}"))
        target=int(math.ceil(required_kwh*CONTROL_MODES[mode]["target_factor"]*scale))
        delivered=[int(round(avg*delivery*scale)) for a,mx,avg,delivery in reduction_candidates]
        model.Add(sum(v*d for v,d in zip(vars,delivered))>=target)
        dw=CONTROL_MODES[mode]["discomfort_weight"]
        model.Minimize(sum(v*(int(reduction_candidates[i][0]["discomfort"])*dw+5) for i,v in enumerate(vars)))
        solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=2.0
        status=solver.Solve(model)
        counts=[solver.Value(v) for v in vars] if status in (cp_model.OPTIMAL,cp_model.FEASIBLE) else [mx for a,mx,avg,delivery in reduction_candidates]
        for count,(a,mx,avg,delivery) in zip(counts,reduction_candidates):
            if count<=0: continue
            g=count*avg; e=g*delivery; gross_reduction+=g; effective_reduction+=e
            rows.append({"대안":a["name"],"유형":"사용량감축","실행횟수":count,"예상절감·이동량(kWh)":g,"실효량(kWh)":e,"불편점수":count*int(a["discomfort"])})
    # Peak-shift recommendations are separate from the monthly reduction target.
    shift_ratio={"편의 우선":0.15,"균형":0.35,"목표달성 우선":0.55}[mode]*(1.0 if direct else 0.55)
    for a,mx,avg,delivery in shift_candidates:
        count=int(round(mx*shift_ratio))
        if count<=0: continue
        g=count*avg; e=g*delivery
        rows.append({"대안":a["name"],"유형":"시간이동","실행횟수":count,"예상절감·이동량(kWh)":g,"실효량(kWh)":e,"불편점수":count*int(a["discomfort"])})
    return pd.DataFrame(rows),gross_reduction,effective_reduction

def controlled_profile(base: np.ndarray, action_plan: pd.DataFrame, remaining_days: int, direct: bool) -> np.ndarray:
    p=np.array(base,dtype=float).copy()
    if action_plan is None or action_plan.empty or remaining_days<=0: return p
    reduce_total=float(action_plan.loc[action_plan["유형"]=="사용량감축","실효량(kWh)"].sum())/remaining_days
    shift_total=float(action_plan.loc[action_plan["유형"]=="시간이동","실효량(kWh)"].sum())/remaining_days
    # Reduction is allocated to high-usage afternoon/evening hours; it never drives a value below 35% of baseline.
    reduction_hours=np.arange(14,24)
    weights=p[reduction_hours]; weights=weights/weights.sum() if weights.sum()>0 else np.ones(len(weights))/len(weights)
    for h,w in zip(reduction_hours,weights): p[h]=max(p[h]-reduce_total*w,base[h]*0.35)
    # Shift load from 16~22 to 22~08 while preserving shifted energy.
    peak_hours=np.arange(16,22); off_hours=np.array(list(range(22,24))+list(range(0,8)))
    pw=p[peak_hours]; pw=pw/pw.sum() if pw.sum()>0 else np.ones(len(pw))/len(pw)
    removed=0.0
    for h,w in zip(peak_hours,pw):
        x=min(shift_total*w,p[h]*0.45); p[h]-=x; removed+=x
    # Place shifted energy into the lowest off-peak hours without creating a new maximum above the original peak.
    cap=float(np.max(base))
    residual=removed
    for h in off_hours[np.argsort(p[off_hours])]:
        room=max(cap-p[h],0.0); add=min(room,residual); p[h]+=add; residual-=add
        if residual<=1e-9: break
    # If the available off-peak envelope is insufficient, the residual is treated as an unexecuted shift.
    return np.maximum(p,0)


def bill_comparison(forecast: float, month: int, metric: pd.Series, basic_fee:float,basic_inc:float,premium_fee:float,premium_inc:float,overage:float) -> pd.DataFrame:
    off=float(metric["경부하비중"]); mid=float(metric["중간부하비중"]); peak=float(metric["최대부하비중"])
    rows=[
        ("일반 주택용",residential_bill(forecast,month)),
        ("제주 TOU(3kW)",tou_bill(forecast,month,off,mid,peak,3)),
        ("기본형",subscription_bill(forecast,basic_fee,basic_inc,overage)),
        ("프리미엄형",subscription_bill(forecast,premium_fee,premium_inc,overage)),
    ]
    df=pd.DataFrame(rows,columns=["요금제","예상요금(원)"]).sort_values("예상요금(원)")
    df["최저 대비 차이(원)"]=df["예상요금(원)"]-df["예상요금(원)"].min()
    return df


def cumulative_projection(dm: pd.DataFrame, cutoff:int, forecast_total:float, advisory_reduction:float, direct_reduction:float) -> pd.DataFrame:
    dm=dm.sort_values("일").copy(); obs=dm[dm["일"]<=cutoff]; rem=dm[dm["일"]>cutoff]
    current=float(obs["일사용량_kWh"].sum()); base_remaining=max(forecast_total-current,0)
    if len(rem):
        pattern=rem["일사용량_kWh"].to_numpy(float); pattern=pattern/pattern.sum() if pattern.sum()>0 else np.ones(len(rem))/len(rem)
    else: pattern=np.array([])
    rows=[]; cum=current
    for _,r in obs.iterrows():
        # actual cumulative is reconstructed progressively
        pass
    actual_cum=obs["일사용량_kWh"].cumsum().to_numpy(float)
    for i,(_,r) in enumerate(obs.iterrows()): rows.append({"일":int(r["일"]),"실제누적":actual_cum[i],"미제어예상":np.nan,"행동권고예상":np.nan,"직접제어예상":np.nan})
    base_cum=current; adv_cum=current; dir_cum=current
    for i,(_,r) in enumerate(rem.iterrows()):
        base_day=base_remaining*(pattern[i] if len(pattern) else 0)
        adv_day=max(base_day-advisory_reduction/max(len(rem),1),0)
        dir_day=max(base_day-direct_reduction/max(len(rem),1),0)
        base_cum+=base_day;adv_cum+=adv_day;dir_cum+=dir_day
        rows.append({"일":int(r["일"]),"실제누적":np.nan,"미제어예상":base_cum,"행동권고예상":adv_cum,"직접제어예상":dir_cum})
    return pd.DataFrame(rows)


def zip_results(files: Dict[str,bytes]) -> bytes:
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as z:
        for name,content in files.items(): z.writestr(name,content)
    return bio.getvalue()




def _robust_scaled_features(metrics: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """군집화용 특성을 중앙값/IQR 기준으로 표준화합니다."""
    feature_cols = [
        "월평균사용량_kWh", "최대시간사용량_kWh", "주말/주중_일사용량비",
        "경부하비중", "중간부하비중", "최대부하비중", "부하율",
        "월변동계수", "하계민감도", "동계민감도",
    ]
    frame = metrics[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        s = frame[col].astype(float)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi).fillna(s.median())
        med = float(s.median())
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        scale = q3 - q1
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(s.std(ddof=0)) or 1.0
        frame[col] = (s - med) / scale
    # 사용량은 긴 꼬리를 완화하기 위해 로그값을 추가 반영합니다.
    usage = np.log1p(metrics["월평균사용량_kWh"].clip(lower=0).astype(float))
    med = float(usage.median()); iqr = float(usage.quantile(.75)-usage.quantile(.25)) or 1.0
    frame["월평균사용량_kWh"] = (usage-med)/iqr
    return frame.to_numpy(float), feature_cols


def _kmeans_numpy(x: np.ndarray, n_clusters: int, seed: int = 42, max_iter: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    """추가 라이브러리 없이 실행되는 결정적 K-means입니다."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n_clusters < 2 or n_clusters > n:
        raise ValueError("군집 수가 데이터 범위를 벗어났습니다.")
    centers = [x[int(rng.integers(0, n))].copy()]
    for _ in range(1, n_clusters):
        d2 = np.min(np.stack([np.sum((x-c)**2, axis=1) for c in centers], axis=1), axis=1)
        total = float(d2.sum())
        idx = int(rng.choice(n, p=d2/total)) if total > 0 else int(rng.integers(0, n))
        centers.append(x[idx].copy())
    centers = np.vstack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        distances = np.stack([np.sum((x-c)**2, axis=1) for c in centers], axis=1)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for k in range(n_clusters):
            members = x[new_labels == k]
            if len(members):
                new_centers[k] = members.mean(axis=0)
            else:
                far = int(np.argmax(np.min(distances, axis=1)))
                new_centers[k] = x[far]
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers, atol=1e-7):
            labels = new_labels
            centers = new_centers
            break
        labels, centers = new_labels, new_centers
    return labels, centers


def _cluster_names(cluster_means: pd.DataFrame) -> Dict[int, str]:
    """군집 중심의 상대적 특성에 따라 이해하기 쉬운 이름을 부여합니다."""
    cols = [
        "월평균사용량_kWh", "최대시간사용량_kWh", "주말/주중_일사용량비",
        "경부하비중", "최대부하비중", "부하율", "월변동계수",
        "하계민감도", "동계민감도",
    ]
    z = cluster_means[cols].copy()
    for c in cols:
        std = float(z[c].std(ddof=0)) or 1.0
        z[c] = (z[c] - float(z[c].mean())) / std
    trait_map = {
        "월평균사용량_kWh": ("고사용", "저사용"),
        "최대시간사용량_kWh": ("고피크", "저피크"),
        "주말/주중_일사용량비": ("주말집중", "주중집중"),
        "경부하비중": ("야간집중", "주간집중"),
        "최대부하비중": ("최대부하집중", "비피크중심"),
        "부하율": ("평탄", "첨두"),
        "월변동계수": ("변동", "규칙"),
        "하계민감도": ("하계민감", "하계둔감"),
        "동계민감도": ("동계민감", "동계둔감"),
    }
    order = cluster_means["월평균사용량_kWh"].sort_values().index.tolist()
    names = {}
    used = set()
    for rank, idx in enumerate(order, 1):
        row = z.loc[idx]
        ranked = sorted(cols, key=lambda c: abs(float(row[c])), reverse=True)
        traits = []
        for c in ranked:
            value = float(row[c])
            if abs(value) < 0.35:
                continue
            traits.append(trait_map[c][0] if value >= 0 else trait_map[c][1])
            if len(traits) == 2:
                break
        if not traits:
            traits = ["표준"]
        desc = "·".join(traits) + "형"
        label = f"군집 {rank} · {desc}"
        if label in used:
            label = f"군집 {rank} · {desc}-{rank}"
        used.add(label)
        names[int(idx)] = label
    return names


def dynamic_cluster_metrics(metrics: pd.DataFrame, n_clusters: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x, _ = _robust_scaled_features(metrics)
    # 극단값이 독립된 초소형 군집을 만들지 않도록 중앙 98%로 중심을 학습한 뒤
    # 모든 고객을 가장 가까운 중심에 다시 배정합니다.
    score = np.max(np.abs(x), axis=1)
    fit_mask = score <= np.quantile(score, 0.98)
    fit_x = x[fit_mask]
    _, centers = _kmeans_numpy(fit_x, int(n_clusters), seed=42)
    distances = np.stack([np.sum((x-c)**2, axis=1) for c in centers], axis=1)
    labels = np.argmin(distances, axis=1)
    out = metrics.copy()
    out["_cluster_id"] = labels
    means = out.groupby("_cluster_id").agg(
        고객수=("비식별고객ID", "size"),
        월평균사용량_kWh=("월평균사용량_kWh", "mean"),
        최대시간사용량_kWh=("최대시간사용량_kWh", "mean"),
        **{
            "주말/주중_일사용량비": ("주말/주중_일사용량비", "mean"),
            "경부하비중": ("경부하비중", "mean"),
            "중간부하비중": ("중간부하비중", "mean"),
            "최대부하비중": ("최대부하비중", "mean"),
            "부하율": ("부하율", "mean"),
            "월변동계수": ("월변동계수", "mean"),
            "하계민감도": ("하계민감도", "mean"),
            "동계민감도": ("동계민감도", "mean"),
            "20일예측_MAPE": ("20일예측_MAPE", "mean"),
        }
    )
    names = _cluster_names(means)
    out["군집"] = out["_cluster_id"].map(names)
    summary = out.groupby("군집", as_index=False).agg(
        고객수=("비식별고객ID", "size"),
        월평균사용량_kWh=("월평균사용량_kWh", "mean"),
        최대시간사용량_kWh=("최대시간사용량_kWh", "mean"),
        **{
            "주말/주중_일사용량비": ("주말/주중_일사용량비", "mean"),
            "경부하비중": ("경부하비중", "mean"),
            "중간부하비중": ("중간부하비중", "mean"),
            "최대부하비중": ("최대부하비중", "mean"),
            "부하율": ("부하율", "mean"),
            "월변동계수": ("월변동계수", "mean"),
            "하계민감도": ("하계민감도", "mean"),
            "동계민감도": ("동계민감도", "mean"),
            "20일예측_MAPE": ("20일예측_MAPE", "mean"),
        }
    )
    summary["비중"] = summary["고객수"] / max(len(out), 1)
    summary = summary.sort_values("월평균사용량_kWh").reset_index(drop=True)
    return out.drop(columns=["_cluster_id"]), summary


def percent_point_table(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """0~1 비율을 표에서 읽기 쉬운 0~100 퍼센트 포인트로 변환합니다."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * 100.0
    return out


def build_customer_tariff_monitor(
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    month: int,
    cutoff: int,
    basic_fee: float,
    basic_inc: float,
    premium_fee: float,
    premium_inc: float,
    overage: float,
) -> pd.DataFrame:
    """전체 고객의 월중 사용량·월말 예상요금·추천요금제를 한 표로 작성합니다."""
    month_daily = daily[daily["월"] == month]
    metric_lookup = metrics.set_index("비식별고객ID")
    rows = []
    for cid, g in month_daily.groupby("비식별고객ID", sort=False):
        f = forecast_month(g, month, cutoff)
        if cid not in metric_lookup.index:
            continue
        mt = metric_lookup.loc[cid]
        off, mid, peak = float(mt["경부하비중"]), float(mt["중간부하비중"]), float(mt["최대부하비중"])
        bills = {
            "일반 주택용": residential_bill(f["forecast"], month),
            "제주 TOU(3kW)": tou_bill(f["forecast"], month, off, mid, peak, 3.0),
            "기본형": subscription_bill(f["forecast"], basic_fee, basic_inc, overage),
            "프리미엄형": subscription_bill(f["forecast"], premium_fee, premium_inc, overage),
        }
        recommended = min(bills, key=bills.get)
        tou_saving = bills["제주 TOU(3kW)"] - bills[recommended]
        rows.append({
            "비식별고객ID": cid,
            "군집": mt["군집"],
            "현재누적(kWh)": f["current"],
            "월말예상(kWh)": f["forecast"],
            "예측하한(kWh)": f["lower"],
            "예측상한(kWh)": f["upper"],
            "실제월사용량(kWh)": f["actual"],
            "일반주택용(원)": bills["일반 주택용"],
            "제주TOU(원)": bills["제주 TOU(3kW)"],
            "기본형(원)": bills["기본형"],
            "프리미엄형(원)": bills["프리미엄형"],
            "추천요금제": recommended,
            "TOU대비예상절감(원)": max(tou_saving, 0.0),
            "기본형제공량사용률(%)": f["forecast"] / max(basic_inc, 1e-9) * 100.0,
            "프리미엄형제공량사용률(%)": f["forecast"] / max(premium_inc, 1e-9) * 100.0,
            "기본형기준알림": alert_level(f["current"], f["forecast"], basic_inc),
            "예측오차(%)": abs(f["forecast"] - f["actual"]) / max(f["actual"], 1e-9) * 100.0,
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="제주 TOU 실제고객 기반 구독형 요금·수요관리",page_icon="⚡",layout="wide")
st.title("제주 TOU 실제고객 기반 구독형 요금·수요관리 시뮬레이터")
st.caption(f"앱 버전 {APP_VERSION} · 2025년 실제 시간대별 계량자료의 비식별 핵심표본을 사용")

try:
    summary,metrics_raw,daily,profiles,monthly=load_data()
except Exception as e:
    st.error(str(e)); st.stop()

with st.sidebar:
    st.header("공통 요금 설정")
    basic_fee=st.number_input("기본형 월 구독료(원)",0,500000,int(PLAN_DEFAULTS["기본형"]["fee"]),1000)
    basic_inc=st.number_input("기본형 제공량(kWh)",0,3000,int(PLAN_DEFAULTS["기본형"]["included"]),10)
    premium_fee=st.number_input("프리미엄형 월 구독료(원)",0,800000,int(PLAN_DEFAULTS["프리미엄형"]["fee"]),1000)
    premium_inc=st.number_input("프리미엄형 제공량(kWh)",0,5000,int(PLAN_DEFAULTS["프리미엄형"]["included"]),10)
    overage=st.selectbox("초과단가(원/kWh)",[200,300,400,307.3],index=1)
    st.divider()
    st.header("고객 군집 설정")
    cluster_count=st.slider("군집 수",3,8,5,1,help="실제 고객 사용패턴을 3~8개 군집으로 다시 분류합니다.")
    st.caption("군집 수를 바꾸면 전체 현황, 고객별 표, 100가구 포트폴리오에 즉시 반영됩니다.")
    st.divider()
    st.info("요금 비교는 2025년 사용량에 2026년 6월 시행 요금표를 적용한 시뮬레이션입니다. 기후환경요금·연료비조정요금은 제외했습니다.")

metrics, dynamic_cluster_summary = dynamic_cluster_metrics(metrics_raw, cluster_count)

TAB1,TAB2,TAB3,TAB4,TAB5=st.tabs([
    "전체 현황","고객별 요금 모니터링","고객별 진단·제어","실제 100가구 포트폴리오","방법론·한계"
])

with TAB1:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("핵심 고객",f"{summary['core_customers']:,}명")
    c2.metric("고객당 월평균",f"{summary['overall']['monthly_mean_kWh']:,.1f} kWh")
    c3.metric("경부하 사용비중",f"{summary['overall']['offpeak_share']:.1%}")
    c4.metric("20일 시점 예측 MAPE",f"{summary['overall']['forecast20_mape']:.1%}")
    st.subheader(f"실제 사용패턴 군집 · {cluster_count}개")
    cs=dynamic_cluster_summary.sort_values("고객수",ascending=False)
    fig=go.Figure(go.Bar(x=cs["군집"],y=cs["고객수"],text=cs["고객수"],textposition="auto"))
    fig.update_layout(height=400,xaxis_title="군집",yaxis_title="고객 수",xaxis={"tickangle":-20})
    st.plotly_chart(fig,use_container_width=True)
    cs_display=percent_point_table(cs,["비중","경부하비중","중간부하비중","최대부하비중","부하율","20일예측_MAPE"])
    st.dataframe(
        cs_display,hide_index=True,use_container_width=True,
        column_config={
            "비중":st.column_config.NumberColumn("비중",format="%.1f%%"),
            "월평균사용량_kWh":st.column_config.NumberColumn("월평균 사용량(kWh)",format="%.1f"),
            "최대시간사용량_kWh":st.column_config.NumberColumn("최대 시간사용량(kWh)",format="%.2f"),
            "경부하비중":st.column_config.NumberColumn("경부하 비중",format="%.1f%%"),
            "중간부하비중":st.column_config.NumberColumn("중간부하 비중",format="%.1f%%"),
            "최대부하비중":st.column_config.NumberColumn("최대부하 비중",format="%.1f%%"),
            "부하율":st.column_config.NumberColumn("부하율",format="%.1f%%"),
            "20일예측_MAPE":st.column_config.NumberColumn("20일 예측 MAPE",format="%.1f%%"),
        }
    )
    st.subheader("연간 최저요금제 분포")
    counts=metrics["연간최저요금제"].value_counts().reset_index();counts.columns=["요금제","고객수"]
    counts["비중(%)"]=counts["고객수"]/len(metrics)*100.0
    st.dataframe(counts,hide_index=True,use_container_width=True,column_config={"비중(%)":st.column_config.NumberColumn(format="%.1f%%")})
    st.bar_chart(counts.set_index("요금제")[["고객수"]])

with TAB2:
    st.subheader("고객별 요금 모니터링 및 추천 요금제")
    m1,m2,m3=st.columns([1,1,2])
    monitor_month=m1.selectbox("분석 월",list(range(1,13)),index=7,key="monitor_month",format_func=lambda x:f"{x}월")
    monitor_cutoff=m2.slider("조회 시점",5,monthrange(2025,monitor_month)[1]-1,min(20,monthrange(2025,monitor_month)[1]-1),key="monitor_cutoff",format="%d일")
    st.caption("현재까지의 사용량과 해당 고객의 주중·주말 평균을 이용해 월말 사용량과 네 가지 요금제의 예상요금을 계산합니다.")
    monitor=build_customer_tariff_monitor(daily,metrics,monitor_month,monitor_cutoff,basic_fee,basic_inc,premium_fee,premium_inc,overage)
    f1,f2,f3=st.columns(3)
    plan_options=["전체"]+sorted(monitor["추천요금제"].dropna().unique().tolist())
    cluster_options=["전체"]+sorted(monitor["군집"].dropna().unique().tolist())
    selected_plan=f1.selectbox("추천요금제 필터",plan_options)
    selected_cluster=f2.selectbox("군집 필터",cluster_options)
    sort_key=f3.selectbox("정렬 기준",["TOU대비예상절감(원)","월말예상(kWh)","제주TOU(원)","예측오차(%)"])
    shown=monitor.copy()
    if selected_plan!="전체": shown=shown[shown["추천요금제"]==selected_plan]
    if selected_cluster!="전체": shown=shown[shown["군집"]==selected_cluster]
    shown=shown.sort_values(sort_key,ascending=False)
    r1,r2,r3,r4=st.columns(4)
    r1.metric("표시 고객",f"{len(shown):,}명")
    r2.metric("기본형 추천",f"{(shown['추천요금제']=='기본형').sum():,}명")
    r3.metric("프리미엄형 추천",f"{(shown['추천요금제']=='프리미엄형').sum():,}명")
    r4.metric("TOU 대비 평균 절감",f"{shown['TOU대비예상절감(원)'].mean():,.0f}원" if len(shown) else "-")
    st.dataframe(
        shown,hide_index=True,use_container_width=True,height=620,
        column_config={
            "현재누적(kWh)":st.column_config.NumberColumn(format="%.1f"),
            "월말예상(kWh)":st.column_config.NumberColumn(format="%.1f"),
            "예측하한(kWh)":st.column_config.NumberColumn(format="%.1f"),
            "예측상한(kWh)":st.column_config.NumberColumn(format="%.1f"),
            "실제월사용량(kWh)":st.column_config.NumberColumn(format="%.1f"),
            "일반주택용(원)":st.column_config.NumberColumn(format="₩%,.0f"),
            "제주TOU(원)":st.column_config.NumberColumn(format="₩%,.0f"),
            "기본형(원)":st.column_config.NumberColumn(format="₩%,.0f"),
            "프리미엄형(원)":st.column_config.NumberColumn(format="₩%,.0f"),
            "TOU대비예상절감(원)":st.column_config.NumberColumn(format="₩%,.0f"),
            "기본형제공량사용률(%)":st.column_config.NumberColumn(format="%.1f%%"),
            "프리미엄형제공량사용률(%)":st.column_config.NumberColumn(format="%.1f%%"),
            "예측오차(%)":st.column_config.NumberColumn(format="%.1f%%"),
        }
    )
    st.download_button(
        "고객별 요금 모니터링표 CSV 다운로드",
        shown.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{monitor_month}월_{monitor_cutoff}일_고객별_요금모니터링.csv",
        mime="text/csv"
    )

with TAB3:
    left,right=st.columns([1,2.2])
    with left:
        customer=st.selectbox("비식별 고객",metrics.sort_values("비식별고객ID")["비식별고객ID"].tolist())
        month=st.selectbox("분석 월",list(range(1,13)),index=7,format_func=lambda x:f"{x}월")
        maxday=monthrange(2025,month)[1]
        cutoff=st.slider("조회 시점",5,maxday-1,min(20,maxday-1),1,format="%d일")
        current_plan=st.selectbox("현재 선택 요금제",["기본형","프리미엄형"],index=0)
        management=st.radio("관리 방식",["알림·행동권고","한전 직접제어 위임"],horizontal=False)
        control_mode=st.selectbox("제어·권고 강도",list(CONTROL_MODES),index=1)
        ownership=st.multiselect("고객이 등록·연결한 기기",["대기전력차단","냉난방기","건조기","식기세척기","세탁기","게임TV","공기관리기기"],default=["대기전력차단","냉난방기","건조기","식기세척기","세탁기"])
    metric=metrics.loc[metrics["비식별고객ID"]==customer].iloc[0]
    cd=daily[daily["비식별고객ID"]==customer]
    dm=cd[cd["월"]==month].sort_values("일")
    fc=forecast_month(cd,month,cutoff)
    comp=bill_comparison(fc["forecast"],month,metric,basic_fee,basic_inc,premium_fee,premium_inc,overage)
    recommended=str(comp.iloc[0]["요금제"])
    plan_inc=basic_inc if current_plan=="기본형" else premium_inc
    level=alert_level(fc["current"],fc["forecast"],plan_inc)
    with right:
        st.subheader(f"{customer} · {metric['군집']}")
        a,b,c,d=st.columns(4)
        a.metric("현재 누적",f"{fc['current']:,.1f} kWh")
        b.metric("월말 예상",f"{fc['forecast']:,.1f} kWh",f"범위 {fc['lower']:,.0f}~{fc['upper']:,.0f}")
        c.metric("비용상 추천",recommended)
        d.metric("알림 단계",level)
        st.caption(f"사후검증용 실제 월사용량은 {fc['actual']:,.1f}kWh이며, 조회시점에서는 알고리즘 입력으로 사용하지 않습니다.")
        st.dataframe(comp,hide_index=True,use_container_width=True,column_config={"예상요금(원)":st.column_config.NumberColumn(format="₩%,.0f"),"최저 대비 차이(원)":st.column_config.NumberColumn(format="₩%,.0f")})

    st.subheader("목표 설정과 필요한 감축량")
    target_options=["현재 요금제 제공량 이내","직접 목표요금 지정"]
    if month>1: target_options.insert(1,"전월과 같은 요금")
    target_kind=st.radio("목표",target_options,horizontal=True)
    if current_plan=="기본형": fee,inc=basic_fee,basic_inc
    else: fee,inc=premium_fee,premium_inc
    if target_kind=="현재 요금제 제공량 이내": target_usage=float(inc)
    elif target_kind=="전월과 같은 요금":
        prev_month=12 if month==1 else month-1
        prev=monthly[(monthly["비식별고객ID"]==customer)&(monthly["월"]==prev_month)].iloc[0]
        prev_bill=float(prev["구독기본형요금원"] if current_plan=="기본형" else prev["구독프리미엄형요금원"])
        target_usage=inverse_subscription_bill(prev_bill,fee,inc,overage)
        st.caption(f"전월 기준요금 {prev_bill:,.0f}원을 구독요금 구조로 환산한 목표 사용량입니다.")
    else:
        target_bill=st.number_input("목표 월 납부액(원)",0,500000,int(comp.iloc[0]["예상요금(원)"]),1000)
        target_usage=inverse_subscription_bill(target_bill,fee,inc,overage)
    required=max(fc["forecast"]-target_usage,0.0)
    r1,r2,r3=st.columns(3);r1.metric("목표 월사용량",f"{target_usage:,.1f} kWh");r2.metric("필요 감축량",f"{required:,.1f} kWh");r3.metric("남은 기간 일평균",f"{required/max(fc['remaining_days'],1):,.2f} kWh/일")

    advisory_plan,_,advisory_effect=optimize_actions(required,fc["remaining_days"],season_for_month(month),ownership,control_mode,direct=False)
    direct_plan,_,direct_effect=optimize_actions(required,fc["remaining_days"],season_for_month(month),ownership,control_mode,direct=True)
    st.subheader("목표 달성을 위한 대안")
    show_plan=direct_plan if management=="한전 직접제어 위임" else advisory_plan
    if show_plan.empty:
        st.info("현재 목표를 위한 추가 행동이 없거나, 등록된 기기로 산출 가능한 대안이 없습니다.")
    else:
        st.dataframe(show_plan,hide_index=True,use_container_width=True,column_config={"예상절감·이동량(kWh)":st.column_config.NumberColumn(format="%.1f"),"실효량(kWh)":st.column_config.NumberColumn(format="%.1f")})
        selected_effect=direct_effect if management=="한전 직접제어 위임" else advisory_effect
        gap=max(required-selected_effect,0.0)
        g1,g2=st.columns(2);g1.metric("계획상 실효 감축량",f"{selected_effect:,.1f} kWh");g2.metric("목표 대비 잔여 부족량",f"{gap:,.1f} kWh")
        if gap>0.5: st.warning("현재 등록·승인 범위만으로는 목표를 완전히 달성하기 어렵습니다. 상위 요금제 전환, 목표요금 조정 또는 추가 기기 연결을 함께 검토해야 합니다.")
        st.caption("가전별 수치는 계량 분해 결과가 아니라 고객이 등록한 기기와 표준 가전 사용량을 이용한 계획값입니다. 시간이동은 월 사용량 감축에 포함되지 않습니다.")

    st.subheader("조회시점 이후 월 누적 사용량 전망")
    proj=cumulative_projection(dm,cutoff,fc["forecast"],advisory_effect,direct_effect)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=proj["일"],y=proj["실제누적"],name="조회시점까지 실제",mode="lines+markers"))
    fig.add_trace(go.Scatter(x=proj["일"],y=proj["미제어예상"],name="미제어 예상",mode="lines"))
    fig.add_trace(go.Scatter(x=proj["일"],y=proj["행동권고예상"],name="행동권고 이행 예상",mode="lines"))
    fig.add_trace(go.Scatter(x=proj["일"],y=proj["직접제어예상"],name="한전 직접제어 예상",mode="lines"))
    fig.add_hline(y=plan_inc,line_dash="dash",annotation_text="기본 제공량")
    fig.update_layout(height=430,xaxis_title="일",yaxis_title="누적 사용량(kWh)")
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("주중·주말 평균 시간대 패턴")
    prof=profiles[(profiles["비식별고객ID"]==customer)&(profiles["계절"]==season_for_month(month))]
    t1,t2=st.tabs(["주중","주말"])
    for tab,dt in [(t1,"주중"),(t2,"주말")]:
        with tab:
            p=prof[prof["일유형"]==dt].sort_values("시간"); base=p["평균사용량_kWh"].to_numpy(float)
            after=controlled_profile(base,direct_plan if management=="한전 직접제어 위임" else advisory_plan,fc["remaining_days"],management=="한전 직접제어 위임")
            fig=go.Figure();fig.add_trace(go.Scatter(x=p["시간대"],y=base,name="미제어 평균",mode="lines+markers"));fig.add_trace(go.Scatter(x=p["시간대"],y=after,name="관리 적용 평균",mode="lines+markers"))
            fig.update_layout(height=390,xaxis_title="시간",yaxis_title="평균 사용량(kWh)",xaxis={"tickangle":-45})
            st.plotly_chart(fig,use_container_width=True)
            x1,x2,x3=st.columns(3);x1.metric("미제어 일사용량",f"{base.sum():.2f} kWh");x2.metric("관리 적용 일사용량",f"{after.sum():.2f} kWh");x3.metric("최대시간부하 변화",f"{base.max():.2f} → {after.max():.2f} kW")

    files={
        f"{customer}_{month}월_요금비교.csv":comp.to_csv(index=False).encode("utf-8-sig"),
        f"{customer}_{month}월_누적전망.csv":proj.to_csv(index=False).encode("utf-8-sig"),
        f"{customer}_{month}월_행동계획.csv":show_plan.to_csv(index=False).encode("utf-8-sig"),
    }
    st.download_button("선택 고객 결과 ZIP 다운로드",zip_results(files),file_name=f"{customer}_{month}월_진단결과.zip",mime="application/zip")

with TAB4:
    st.subheader("실제 고객 100가구 포트폴리오")
    p1,p2,p3,p4=st.columns(4)
    seed=p1.number_input("표본 추출번호",1,9999,42,1)
    pmonth=p2.selectbox("분석 월",list(range(1,13)),index=7,key="pm",format_func=lambda x:f"{x}월")
    pcut=p3.slider("조회 시점",5,monthrange(2025,pmonth)[1]-1,min(20,monthrange(2025,pmonth)[1]-1),key="pcut")
    participation=p4.slider("직접제어 참여율",0,100,50,5)/100
    sample=metrics.sample(n=100,random_state=int(seed))
    sd=daily[daily["비식별고객ID"].isin(sample["비식별고객ID"])]
    forecast_rows=[]
    for cid,g in sd.groupby("비식별고객ID"):
        f=forecast_month(g,pmonth,pcut); mt=sample[sample["비식별고객ID"]==cid].iloc[0]
        all_bills={
            "일반 주택용":residential_bill(f["forecast"],pmonth),
            "제주 TOU(3kW)":tou_bill(f["forecast"],pmonth,float(mt["경부하비중"]),float(mt["중간부하비중"]),float(mt["최대부하비중"]),3),
            "기본형":subscription_bill(f["forecast"],basic_fee,basic_inc,overage),
            "프리미엄형":subscription_bill(f["forecast"],premium_fee,premium_inc,overage),
        }
        rec=min(all_bills,key=all_bills.get)
        forecast_rows.append({"비식별고객ID":cid,"군집":mt["군집"],"현재누적(kWh)":f["current"],"월말예상(kWh)":f["forecast"],"실제월사용량(kWh)":f["actual"],"추천요금제":rec,"일반주택용(원)":all_bills["일반 주택용"],"제주TOU(원)":all_bills["제주 TOU(3kW)"],"기본형(원)":all_bills["기본형"],"프리미엄형(원)":all_bills["프리미엄형"],"TOU대비예상절감(원)":max(all_bills["제주 TOU(3kW)"]-all_bills[rec],0.0)})
    pf=pd.DataFrame(forecast_rows)
    k1,k2,k3,k4=st.columns(4);k1.metric("표본 고객",f"{len(pf)}명");k2.metric("월말 예상합계",f"{pf['월말예상(kWh)'].sum():,.0f} kWh");k3.metric("기본형 추천",f"{(pf['추천요금제']=='기본형').sum()}명");k4.metric("프리미엄형 추천",f"{(pf['추천요금제']=='프리미엄형').sum()}명")
    cdist=pf["군집"].value_counts().reset_index();cdist.columns=["군집","고객수"];cdist["비중(%)"]=cdist["고객수"]/len(pf)*100.0
    st.dataframe(cdist,hide_index=True,use_container_width=True,column_config={"비중(%)":st.column_config.NumberColumn(format="%.1f%%")})
    st.bar_chart(cdist.set_index("군집")[["고객수"]])
    sprof=profiles[(profiles["비식별고객ID"].isin(sample["비식별고객ID"]))&(profiles["계절"]==season_for_month(pmonth))]
    agg=sprof.groupby(["일유형","시간","시간대"],as_index=False)["평균사용량_kWh"].sum()
    dt=st.radio("포트폴리오 부하곡선",["주중","주말"],horizontal=True)
    ap=agg[agg["일유형"]==dt].sort_values("시간");base=ap["평균사용량_kWh"].to_numpy(float)
    intensity=0.04+0.08*participation
    after=base.copy();peak_hours=np.arange(16,22);off_hours=np.array(list(range(22,24))+list(range(0,8)))
    removed=after[peak_hours]*intensity;after[peak_hours]-=removed;shifted=removed.sum()*0.70;reduced=removed.sum()*0.30;after[off_hours]+=shifted/len(off_hours)
    fig=go.Figure();fig.add_trace(go.Scatter(x=ap["시간대"],y=base,name="실제 평균 패턴",mode="lines+markers"));fig.add_trace(go.Scatter(x=ap["시간대"],y=after,name="직접제어 시나리오",mode="lines+markers"));fig.update_layout(height=430,xaxis_title="시간",yaxis_title="100가구 합계부하(kW)",xaxis={"tickangle":-45});st.plotly_chart(fig,use_container_width=True)
    q1,q2,q3=st.columns(3);q1.metric("기준 최대부하",f"{base.max():.1f} kW");q2.metric("제어 후 최대부하",f"{after.max():.1f} kW");q3.metric("일일 실제 감축",f"{reduced:.1f} kWh")
    st.subheader("100가구 고객별 요금 모니터링 및 추천")
    st.dataframe(pf.sort_values("월말예상(kWh)",ascending=False),hide_index=True,use_container_width=True,column_config={
        "현재누적(kWh)":st.column_config.NumberColumn(format="%.1f"),"월말예상(kWh)":st.column_config.NumberColumn(format="%.1f"),"실제월사용량(kWh)":st.column_config.NumberColumn(format="%.1f"),
        "일반주택용(원)":st.column_config.NumberColumn(format="₩%,.0f"),"제주TOU(원)":st.column_config.NumberColumn(format="₩%,.0f"),"기본형(원)":st.column_config.NumberColumn(format="₩%,.0f"),"프리미엄형(원)":st.column_config.NumberColumn(format="₩%,.0f"),"TOU대비예상절감(원)":st.column_config.NumberColumn(format="₩%,.0f")})
    st.download_button("100가구 결과 CSV 다운로드",pf.to_csv(index=False).encode("utf-8-sig"),file_name="실제100가구_요금모니터링_추천.csv",mime="text/csv")

with TAB5:
    st.markdown(f"""
### v9 변경점
- 표에 표시되는 비중·부하율·예측오차는 모두 **% 단위**로 표시합니다.
- 고객 군집 수를 **3개부터 최대 8개**까지 선택할 수 있으며, 선택 결과가 전체 현황·고객 진단·100가구 분석에 일관되게 반영됩니다.
- 전체 914명을 대상으로 월중 누적 사용량, 월말 예상량, 일반 주택용·제주 TOU·기본형·프리미엄형 예상요금, 추천요금제와 예상절감액을 한 표로 제공합니다.

### 데이터와 분석 범위
- 2025년 제주 TOU 시간대별 계량자료 중 365일 자료와 시간값 완전도 99.5% 이상을 만족한 **914명**을 사용합니다.
- 월 중간 시점 예측은 조회시점까지 관측된 해당 고객의 주중·주말 일평균을 이용합니다.
- 군집은 월사용량, 최대시간 사용량, 주중·주말 차이, TOU 시간대 비중, 부하율, 변동성, 하·동계 민감도를 종합하여 재분류합니다.
- 가전별 대안은 계량값에서 기기를 식별한 결과가 아니라, 고객이 등록하거나 연결한 기기와 최초 가전 부하자료를 바탕으로 만든 계획값입니다.

### 반드시 구분할 사항
1. **실제 계량자료로 확정되는 것**: 전체 사용량, 시간대 패턴, 월말 예측, 요금 비교, 군집과 피크 기여도
2. **추정 또는 시나리오인 것**: 가전별 절감량, 고객의 행동 이행률, 스마트가전 직접제어 성공률
3. **요금 비교의 성격**: 2025년 실제 사용량에 2026년 6월 시행 요금표를 적용한 비교이며, 실제 2025년 청구액 재현이 아닙니다.

### 개인정보·배포
이 앱의 데이터 파일은 고객번호를 제거했지만 개인별 시간대 사용패턴을 포함합니다. 내부 보고·시뮬레이션 용도로만 사용해야 합니다.
""")
