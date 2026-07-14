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

APP_VERSION = "2026-07-14-actual-tou-v8.0"
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


st.set_page_config(page_title="제주 TOU 실제고객 기반 구독형 요금·수요관리",page_icon="⚡",layout="wide")
st.title("제주 TOU 실제고객 기반 구독형 요금·수요관리 시뮬레이터")
st.caption(f"앱 버전 {APP_VERSION} · 2025년 실제 시간대별 계량자료의 비식별 핵심표본을 사용")

try:
    summary,metrics,daily,profiles,monthly=load_data()
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
    st.info("요금 비교는 2025년 사용량에 2026년 6월 시행 요금표를 적용한 시뮬레이션입니다. 기후환경요금·연료비조정요금은 제외했습니다.")

TAB1,TAB2,TAB3,TAB4=st.tabs(["전체 현황","고객별 진단·제어","실제 100가구 포트폴리오","방법론·한계"])

with TAB1:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("핵심 고객",f"{summary['core_customers']:,}명")
    c2.metric("고객당 월평균",f"{summary['overall']['monthly_mean_kWh']:,.1f} kWh")
    c3.metric("경부하 사용비중",f"{summary['overall']['offpeak_share']:.1%}")
    c4.metric("20일 시점 예측 MAPE",f"{summary['overall']['forecast20_mape']:.1%}")
    st.subheader("실제 사용패턴 군집")
    cs=pd.DataFrame(summary["cluster_summary"]).sort_values("고객수",ascending=False)
    fig=go.Figure(go.Bar(x=cs["군집"],y=cs["고객수"],text=cs["고객수"],textposition="auto"))
    fig.update_layout(height=380,xaxis_title="군집",yaxis_title="고객 수")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(cs,hide_index=True,use_container_width=True,column_config={"비중":st.column_config.NumberColumn(format="%.1%%"),"월평균사용량_kWh":st.column_config.NumberColumn(format="%.1f"),"경부하비중":st.column_config.NumberColumn(format="%.1%%"),"최대부하비중":st.column_config.NumberColumn(format="%.1%%"),"20일예측_MAPE":st.column_config.NumberColumn(format="%.1%%")})
    st.subheader("연간 최저요금제 분포")
    counts=metrics["연간최저요금제"].value_counts().reset_index();counts.columns=["요금제","고객수"]
    st.bar_chart(counts.set_index("요금제"))

with TAB2:
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

    # Download selected customer results
    files={
        f"{customer}_{month}월_요금비교.csv":comp.to_csv(index=False).encode("utf-8-sig"),
        f"{customer}_{month}월_누적전망.csv":proj.to_csv(index=False).encode("utf-8-sig"),
        f"{customer}_{month}월_행동계획.csv":show_plan.to_csv(index=False).encode("utf-8-sig"),
    }
    st.download_button("선택 고객 결과 ZIP 다운로드",zip_results(files),file_name=f"{customer}_{month}월_진단결과.zip",mime="application/zip")

with TAB3:
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
        b=subscription_bill(f["forecast"],basic_fee,basic_inc,overage); pr=subscription_bill(f["forecast"],premium_fee,premium_inc,overage)
        forecast_rows.append({"비식별고객ID":cid,"군집":mt["군집"],"현재누적":f["current"],"월말예상":f["forecast"],"실제월사용량":f["actual"],"추천구독형":"기본형" if b<=pr else "프리미엄형","기본형예상요금":b,"프리미엄형예상요금":pr})
    pf=pd.DataFrame(forecast_rows)
    k1,k2,k3,k4=st.columns(4);k1.metric("표본 고객",f"{len(pf)}명");k2.metric("월말 예상합계",f"{pf['월말예상'].sum():,.0f} kWh");k3.metric("기본형 추천",f"{(pf['추천구독형']=='기본형').sum()}명");k4.metric("프리미엄형 추천",f"{(pf['추천구독형']=='프리미엄형').sum()}명")
    cdist=pf["군집"].value_counts().reset_index();cdist.columns=["군집","고객수"]
    st.bar_chart(cdist.set_index("군집"))
    # Aggregate real profiles and direct-control scenario
    sprof=profiles[(profiles["비식별고객ID"].isin(sample["비식별고객ID"]))&(profiles["계절"]==season_for_month(pmonth))]
    agg=sprof.groupby(["일유형","시간","시간대"],as_index=False)["평균사용량_kWh"].sum()
    dt=st.radio("포트폴리오 부하곡선",["주중","주말"],horizontal=True)
    ap=agg[agg["일유형"]==dt].sort_values("시간");base=ap["평균사용량_kWh"].to_numpy(float)
    # Simple portfolio direct-control envelope grounded in the real aggregate curve.
    intensity=0.04+0.08*participation
    after=base.copy();peak_hours=np.arange(16,22);off_hours=np.array(list(range(22,24))+list(range(0,8)))
    removed=after[peak_hours]*intensity;after[peak_hours]-=removed;shifted=removed.sum()*0.70;reduced=removed.sum()*0.30;after[off_hours]+=shifted/len(off_hours)
    fig=go.Figure();fig.add_trace(go.Scatter(x=ap["시간대"],y=base,name="실제 평균 패턴",mode="lines+markers"));fig.add_trace(go.Scatter(x=ap["시간대"],y=after,name="직접제어 시나리오",mode="lines+markers"));fig.update_layout(height=430,xaxis_title="시간",yaxis_title="100가구 합계부하(kW)",xaxis={"tickangle":-45});st.plotly_chart(fig,use_container_width=True)
    q1,q2,q3=st.columns(3);q1.metric("기준 최대부하",f"{base.max():.1f} kW");q2.metric("제어 후 최대부하",f"{after.max():.1f} kW");q3.metric("일일 실제 감축",f"{reduced:.1f} kWh")
    st.dataframe(pf.sort_values("월말예상",ascending=False),hide_index=True,use_container_width=True)
    st.download_button("100가구 결과 CSV 다운로드",pf.to_csv(index=False).encode("utf-8-sig"),file_name="실제100가구_요금제추천.csv",mime="text/csv")

with TAB4:
    st.markdown("""
### 이번 버전의 변경점
- 표준 4인 가구에서 임의 생성한 고객 대신, 2025년 제주 TOU 시간대별 계량자료 중 365일 자료와 시간값 완전도 99.5% 이상을 만족한 **914명**을 사용합니다.
- 고객별 월사용량, 주중·주말 패턴, TOU 시간대 비중, 계절 민감도와 군집을 실제 자료로 산출합니다.
- 월 중간 시점 예측은 조회시점까지 관측된 해당 고객의 주중·주말 일평균을 이용합니다.
- 가전별 대안은 계량값에서 기기를 식별한 결과가 아니라, 고객이 등록하거나 연결한 기기와 최초 가전 부하자료를 바탕으로 만든 계획값입니다.

### 반드시 구분할 사항
1. **실제 계량자료로 확정되는 것**: 전체 사용량, 시간대 패턴, 월말 예측, 요금 비교, 군집과 피크 기여도
2. **추정 또는 시나리오인 것**: 가전별 절감량, 고객의 행동 이행률, 스마트가전 직접제어 성공률
3. **요금 비교의 성격**: 2025년 실제 사용량에 2026년 6월 시행 요금표를 적용한 비교이며, 실제 2025년 청구액 재현이 아닙니다.

### 개인정보·배포
이 앱의 데이터 파일은 고객번호를 제거했지만 개인별 시간대 사용패턴을 포함합니다. 공개 GitHub 또는 공개 Streamlit Community Cloud에 올리지 말고, 사내 접근통제 환경에서 사용해야 합니다.
""")
