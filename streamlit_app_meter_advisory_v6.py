from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ortools.sat.python import cp_model

# 기존 v5의 원본 Excel 보정·가상가구 생성 로직을 재사용합니다.
# v6에서는 이 가전별 자료를 '시뮬레이션의 숨은 정답'에만 사용하고,
# 고객 안내 알고리즘에는 총계량값과 가입 시 등록한 가전 보유정보만 제공합니다.
from streamlit_app_100households_v5 import (
    ARCHETYPES,
    SOURCE_COMPONENTS,
    generate_population,
    simple_kmeans,
    source_components,
    source_monthly_kwh,
    smooth_noise,
)

APP_VERSION = "2026-07-14-single-meter-advisory-v6.0"
HOURS = np.arange(24)
DAYS_IN_MONTH = 30
WEEKDAYS_PER_MONTH = 22
WEEKENDS_PER_MONTH = 8

PLAN_DEFAULTS = {
    "기본형": {"monthly_fee": 84_900.0, "included_kwh": 450.0},
    "프리미엄형": {"monthly_fee": 249_900.0, "included_kwh": 1_000.0},
}

ALERT_ORDER = ["정상", "관심", "주의", "경고", "긴급"]
ALERT_SCORE = {name: i for i, name in enumerate(ALERT_ORDER)}

# 행동대안의 절감량은 개별 가전 계측값이 아니라 표준 4인가구 자료와
# 고객이 가입 시 등록한 가전 보유정보를 토대로 제시하는 추정 범위입니다.
ACTION_LIBRARY: List[Dict[str, object]] = [
    {
        "id": "standby",
        "name": "취침·외출 시 대기전력 일괄 차단",
        "ownership": "대기전력차단가능",
        "seasons": ["봄가을", "여름", "겨울"],
        "low": 0.12,
        "high": 0.30,
        "frequency": "daily",
        "max_per_day": 1.0,
        "unit": "일",
        "discomfort": 1,
        "reliability": 0.82,
        "description": "스마트플러그·멀티탭 또는 외출모드를 활용해 불필요한 상시전력을 줄임",
    },
    {
        "id": "hvac_setpoint",
        "name": "냉난방 설정온도 1℃ 완화·외출 절전",
        "ownership": "냉난방기",
        "seasons": ["여름", "겨울"],
        "low": 0.45,
        "high": 1.20,
        "frequency": "daily",
        "max_per_day": 1.0,
        "unit": "일",
        "discomfort": 4,
        "reliability": 0.72,
        "description": "실내 쾌적범위를 유지하면서 설정온도를 완화하고 외출 시 절전운전을 적용",
    },
    {
        "id": "hvac_hour",
        "name": "냉난방 운전시간 1시간 단축",
        "ownership": "냉난방기",
        "seasons": ["여름", "겨울"],
        "low": 0.55,
        "high": 1.35,
        "frequency": "hourly",
        "max_per_day": 2.0,
        "unit": "시간",
        "discomfort": 7,
        "reliability": 0.68,
        "description": "예냉·예열 또는 외출 전 조기 종료로 하루 운전시간을 일부 단축",
    },
    {
        "id": "dryer_skip",
        "name": "건조기 1회 자연건조로 대체",
        "ownership": "건조기",
        "seasons": ["봄가을", "여름", "겨울"],
        "low": 1.40,
        "high": 2.80,
        "frequency": "weekly",
        "events_per_week": 3.0,
        "unit": "회",
        "discomfort": 5,
        "reliability": 0.88,
        "description": "가능한 세탁물에 한해 건조기 대신 자연건조를 선택",
    },
    {
        "id": "dishwasher_eco",
        "name": "식기세척기 절전모드·모아서 사용",
        "ownership": "식기세척기",
        "seasons": ["봄가을", "여름", "겨울"],
        "low": 0.22,
        "high": 0.55,
        "frequency": "weekly",
        "events_per_week": 6.0,
        "unit": "회",
        "discomfort": 2,
        "reliability": 0.76,
        "description": "소량 운전을 줄이고 절전모드를 사용",
    },
    {
        "id": "laundry_eco",
        "name": "세탁기 냉수·절전코스 사용",
        "ownership": "세탁기",
        "seasons": ["봄가을", "여름", "겨울"],
        "low": 0.10,
        "high": 0.28,
        "frequency": "weekly",
        "events_per_week": 6.0,
        "unit": "회",
        "discomfort": 1,
        "reliability": 0.74,
        "description": "고온세탁이 불필요한 경우 냉수·절전코스를 선택",
    },
    {
        "id": "game_tv",
        "name": "게임·TV 이용시간 2시간 단축",
        "ownership": "게임TV",
        "seasons": ["봄가을", "여름", "겨울"],
        "low": 0.25,
        "high": 0.70,
        "frequency": "daily",
        "max_per_day": 1.0,
        "unit": "일",
        "discomfort": 4,
        "reliability": 0.78,
        "description": "고객이 설정한 시간한도·종료알림을 활용해 이용시간을 단축",
    },
    {
        "id": "aircare",
        "name": "공기청정기·제습기 절전운전",
        "ownership": "공기관리기기",
        "seasons": ["봄가을", "여름", "겨울"],
        "low": 0.16,
        "high": 0.55,
        "frequency": "daily",
        "max_per_day": 1.0,
        "unit": "일",
        "discomfort": 2,
        "reliability": 0.67,
        "description": "자동·저속모드, 예약종료를 활용해 불필요한 연속운전을 줄임",
    },
    {
        "id": "clothing_care",
        "name": "의류관리기 사용 1회 축소",
        "ownership": "의류관리기",
        "seasons": ["봄가을", "여름", "겨울"],
        "low": 0.30,
        "high": 0.85,
        "frequency": "weekly",
        "events_per_week": 3.0,
        "unit": "회",
        "discomfort": 3,
        "reliability": 0.74,
        "description": "필요도가 낮은 운전은 다음 주기로 미루거나 생략",
    },
    {
        "id": "robot_vacuum",
        "name": "로봇청소기 운전횟수 축소",
        "ownership": "로봇청소기",
        "seasons": ["봄가을", "여름", "겨울"],
        "low": 0.05,
        "high": 0.16,
        "frequency": "weekly",
        "events_per_week": 5.0,
        "unit": "회",
        "discomfort": 1,
        "reliability": 0.71,
        "description": "오염도가 낮은 날의 반복 운전을 줄임",
    },
]

GRID_SHIFT_LIBRARY = [
    {"name": "세탁기·건조기 운전을 22시 이후로 이동", "ownership": "세탁기", "shift_kwh": 1.8, "discomfort": 2},
    {"name": "식기세척기를 취침 후 예약운전", "ownership": "식기세척기", "shift_kwh": 0.8, "discomfort": 1},
    {"name": "냉난방 예냉·예열 후 피크시간 출력 완화", "ownership": "냉난방기", "shift_kwh": 0.9, "discomfort": 3},
    {"name": "의류관리기·로봇청소기를 저부하시간으로 이동", "ownership": "의류관리기", "shift_kwh": 0.5, "discomfort": 1},
]

PACKAGE_STYLES = {
    "최소불편형": {"safety": 1.00, "discomfort_weight": 140, "count_weight": 18, "reliability_weight": 20},
    "균형형": {"safety": 1.10, "discomfort_weight": 75, "count_weight": 10, "reliability_weight": 12},
    "목표달성 우선형": {"safety": 1.25, "discomfort_weight": 35, "count_weight": 5, "reliability_weight": 6},
}


def subscription_bill(usage_kwh: float, monthly_fee: float, included_kwh: float, overage_rate: float) -> float:
    return float(monthly_fee + max(float(usage_kwh) - float(included_kwh), 0.0) * float(overage_rate))


def inverse_bill_to_usage(target_bill: float, monthly_fee: float, included_kwh: float, overage_rate: float) -> float:
    """목표 납부액을 넘지 않는 최대 월사용량을 계산합니다."""
    if target_bill <= monthly_fee + 1e-9 or overage_rate <= 0:
        return float(included_kwh)
    return float(included_kwh + (target_bill - monthly_fee) / overage_rate)


def month_calendar() -> List[str]:
    # 4주(주중 5일+주말 2일) + 주중 2일 = 주중 22일, 주말 8일
    return (["주중"] * 5 + ["주말"] * 2) * 4 + ["주중", "주중"]


def add_customer_metadata(households: pd.DataFrame, season: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 4211)
    result = households.copy()
    n = len(result)

    archetype = result["생성유형"].astype(str)
    high_use = archetype.eq("고사용량형").to_numpy()
    home = archetype.eq("재택형").to_numpy()
    evening = archetype.eq("저녁집중형").to_numpy()

    def flags(prob: np.ndarray | float) -> np.ndarray:
        p = np.full(n, float(prob)) if np.isscalar(prob) else np.asarray(prob, dtype=float)
        return rng.random(n) < np.clip(p, 0.02, 0.99)

    result["냉난방기"] = flags(np.where(high_use | home, 0.98, 0.90))
    result["건조기"] = flags(np.where(high_use, 0.92, 0.72))
    result["식기세척기"] = flags(np.where(high_use | home, 0.86, 0.66))
    result["세탁기"] = True
    result["게임TV"] = flags(np.where(evening, 0.92, 0.68))
    result["공기관리기기"] = flags(np.where(home, 0.94, 0.76))
    result["의류관리기"] = flags(np.where(high_use, 0.76, 0.48))
    result["로봇청소기"] = flags(np.where(home | high_use, 0.86, 0.64))
    result["대기전력차단가능"] = flags(0.95)

    # 실제 데이터가 없는 단계의 비교 기준: 가구별 기준사용량에 합리적 변동을 부여한 가상 이력
    baseline = result["월사용량(kWh)"].to_numpy(dtype=float)
    prev_month_factor = rng.normal(0.92 if season in ["여름", "겨울"] else 1.03, 0.10, n)
    prev_year_factor = rng.normal(0.98, 0.09, n)
    result["전월사용량(kWh)"] = np.maximum(baseline * np.clip(prev_month_factor, 0.65, 1.30), 1.0)
    result["전년동월사용량(kWh)"] = np.maximum(baseline * np.clip(prev_year_factor, 0.70, 1.30), 1.0)
    result["과거알림횟수"] = rng.integers(0, 6, n)
    result["알림수용도"] = np.clip(
        0.55 * result["제어수용도"].to_numpy(dtype=float) + rng.normal(0.25, 0.08, n),
        0.25,
        0.98,
    )
    return result


def build_monthly_meter_streams(
    households: pd.DataFrame,
    population: Dict[str, Dict[str, np.ndarray]],
    season: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """가구별 30일×24시간 단일 계량값을 생성합니다.

    baseline: 과거 AMI로 학습했다고 가정한 고객별 정상 패턴
    actual: 당월 기상·재실·행동 변동을 반영한 시뮬레이션 실제 계량값
    """
    rng = np.random.default_rng(seed + 8191)
    n = len(households)
    calendar = month_calendar()
    baseline = np.zeros((n, DAYS_IN_MONTH * 24), dtype=float)
    actual = np.zeros_like(baseline)
    daily_rows: List[Dict[str, object]] = []

    common_weather = rng.normal(1.0, 0.10 if season in ["여름", "겨울"] else 0.05, DAYS_IN_MONTH)
    common_weather = np.clip(common_weather, 0.82, 1.22)
    monthly_factor = rng.normal(1.0, 0.09, n)
    monthly_factor = np.clip(monthly_factor, 0.78, 1.25)

    for day_index, day_type in enumerate(calendar):
        start = day_index * 24
        end = start + 24
        base_day = population[day_type]["total"]
        baseline[:, start:end] = base_day

        daily_individual = rng.lognormal(mean=-0.5 * 0.07**2, sigma=0.07, size=n)
        for i in range(n):
            profile_noise = smooth_noise(rng, 0.045)
            actual[i, start:end] = base_day[i] * common_weather[day_index] * daily_individual[i] * profile_noise

    # 가구별 당월 총사용량은 기준사용량×당월변동계수로 맞추되 시간형태는 변동을 유지합니다.
    target_actual = households["월사용량(kWh)"].to_numpy(dtype=float) * monthly_factor
    actual_sum = actual.sum(axis=1)
    actual *= np.divide(target_actual, actual_sum, out=np.ones(n), where=actual_sum > 1e-12)[:, None]

    for day_index, day_type in enumerate(calendar):
        start = day_index * 24
        end = start + 24
        day_total = actual[:, start:end].sum(axis=1)
        for i in range(n):
            daily_rows.append({
                "고객ID": households.iloc[i]["고객ID"],
                "일자": day_index + 1,
                "대표일": day_type,
                "일사용량(kWh)": float(day_total[i]),
            })

    return actual, baseline, pd.DataFrame(daily_rows)


def assign_current_plans(
    households: pd.DataFrame,
    assignment_mode: str,
    basic_quota: float,
    basic_share_percent: int,
) -> pd.DataFrame:
    result = households.copy()
    usage = result["월사용량(kWh)"].to_numpy(dtype=float)
    if assignment_mode == "전 가구 기본형":
        result["현재요금제"] = "기본형"
    elif assignment_mode == "전 가구 프리미엄형":
        result["현재요금제"] = "프리미엄형"
    elif assignment_mode == "지정 비율(저사용량 기본형 우선)":
        count = int(round(len(result) * basic_share_percent / 100.0))
        order = np.argsort(usage, kind="stable")
        plans = np.full(len(result), "프리미엄형", dtype=object)
        plans[order[:count]] = "기본형"
        result["현재요금제"] = plans
    else:
        result["현재요금제"] = np.where(usage <= basic_quota, "기본형", "프리미엄형")
    return result


@dataclass
class ForecastResult:
    observed_kwh: float
    forecast_kwh: float
    lower_kwh: float
    upper_kwh: float
    trend_factor: float
    recent_factor: float


def forecast_month_usage(actual: np.ndarray, baseline: np.ndarray, cutoff: int) -> ForecastResult:
    cutoff = int(np.clip(cutoff, 1, len(actual)))
    observed = float(actual[:cutoff].sum())
    expected_observed = float(baseline[:cutoff].sum())
    ratio_mtd = observed / max(expected_observed, 1e-9)

    recent_window = min(72, cutoff)
    recent_actual = float(actual[cutoff - recent_window:cutoff].sum())
    recent_base = float(baseline[cutoff - recent_window:cutoff].sum())
    recent_ratio = recent_actual / max(recent_base, 1e-9)

    trend = float(np.clip(0.68 * ratio_mtd + 0.32 * recent_ratio, 0.50, 1.80))
    remaining = float(baseline[cutoff:].sum())
    forecast = observed + remaining * trend

    valid = baseline[:cutoff] > 0.02
    ratios = np.divide(actual[:cutoff][valid], baseline[:cutoff][valid]) if valid.any() else np.array([1.0])
    if len(ratios) > 168:
        ratios = ratios[-168:]
    dispersion = float(np.std(np.clip(ratios, 0.2, 3.0)))
    uncertainty = float(np.clip(0.055 + 0.40 * dispersion, 0.06, 0.24))
    lower = max(observed, forecast * (1.0 - uncertainty))
    upper = forecast * (1.0 + uncertainty)
    return ForecastResult(observed, forecast, lower, upper, trend, recent_ratio)


def plan_values(plan: str, basic_fee: float, basic_quota: float, premium_fee: float, premium_quota: float) -> Tuple[float, float]:
    if plan == "기본형":
        return float(basic_fee), float(basic_quota)
    return float(premium_fee), float(premium_quota)


def alert_level(observed: float, forecast: float, quota: float, elapsed_fraction: float) -> str:
    utilization = observed / max(quota, 1e-9)
    pace = observed / max(quota * max(elapsed_fraction, 0.03), 1e-9)
    forecast_ratio = forecast / max(quota, 1e-9)
    if observed >= quota or utilization >= 0.95 or forecast_ratio >= 1.20:
        return "긴급"
    if utilization >= 0.85 or forecast_ratio >= 1.08 or pace >= 1.25:
        return "경고"
    if utilization >= 0.70 or forecast_ratio > 1.00 or pace >= 1.12:
        return "주의"
    if utilization >= 0.50 or forecast_ratio >= 0.92 or pace >= 1.03:
        return "관심"
    return "정상"


def build_monitoring_table(
    households: pd.DataFrame,
    actual: np.ndarray,
    baseline: np.ndarray,
    cutoff: int,
    basic_fee: float,
    basic_quota: float,
    premium_fee: float,
    premium_quota: float,
    overage_rate: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    elapsed_fraction = cutoff / float(DAYS_IN_MONTH * 24)
    remaining_hours = DAYS_IN_MONTH * 24 - cutoff
    remaining_days = remaining_hours / 24.0

    for i, row in households.reset_index(drop=True).iterrows():
        fc = forecast_month_usage(actual[i], baseline[i], cutoff)
        plan = str(row["현재요금제"])
        fee, quota = plan_values(plan, basic_fee, basic_quota, premium_fee, premium_quota)
        bill = subscription_bill(fc.forecast_kwh, fee, quota, overage_rate)
        remaining_quota = quota - fc.observed_kwh
        recommended_basic = subscription_bill(fc.forecast_kwh, basic_fee, basic_quota, overage_rate)
        recommended_premium = subscription_bill(fc.forecast_kwh, premium_fee, premium_quota, overage_rate)
        recommended_plan = "기본형" if recommended_basic <= recommended_premium else "프리미엄형"
        current_plan_bill = bill
        recommended_bill = min(recommended_basic, recommended_premium)
        level = alert_level(fc.observed_kwh, fc.forecast_kwh, quota, elapsed_fraction)
        allowed_daily = max(remaining_quota, 0.0) / max(remaining_days, 1e-9)

        rows.append({
            "고객ID": row["고객ID"],
            "생성유형": row["생성유형"],
            "현재요금제": plan,
            "현재누적사용량(kWh)": fc.observed_kwh,
            "남은정액사용량(kWh)": remaining_quota,
            "남은일수": remaining_days,
            "정액내일평균허용량(kWh/일)": allowed_daily,
            "월말예상사용량(kWh)": fc.forecast_kwh,
            "예측하한(kWh)": fc.lower_kwh,
            "예측상한(kWh)": fc.upper_kwh,
            "월말예상요금(원)": bill,
            "기본형예상요금(원)": recommended_basic,
            "프리미엄형예상요금(원)": recommended_premium,
            "추천요금제": recommended_plan,
            "요금제전환예상절감액(원)": max(current_plan_bill - recommended_bill, 0.0),
            "알림단계": level,
            "알림점수": ALERT_SCORE[level],
            "추세계수": fc.trend_factor,
            "최근72시간계수": fc.recent_factor,
            "전월사용량(kWh)": row["전월사용량(kWh)"],
            "전년동월사용량(kWh)": row["전년동월사용량(kWh)"],
            "알림수용도": row["알림수용도"],
            "과거알림횟수": row["과거알림횟수"],
        })
    return pd.DataFrame(rows)


def add_meter_clusters(monitoring: pd.DataFrame, actual: np.ndarray, baseline: np.ndarray, cutoff: int, cluster_count: int, seed: int) -> pd.DataFrame:
    result = monitoring.copy()
    n = len(result)
    observed_profiles = actual[:, :cutoff]
    if cutoff < 24:
        # 데이터가 하루 미만이면 기준 24시간 곡선으로 군집 특성을 보완합니다.
        shape_source = baseline[:, :24]
    else:
        recent_days = min(7, cutoff // 24)
        shape_source = observed_profiles[:, cutoff - recent_days * 24:cutoff].reshape(n, recent_days, 24).mean(axis=1)

    total = shape_source.sum(axis=1)
    evening = shape_source[:, 18:24].sum(axis=1) / np.maximum(total, 1e-9)
    daytime = shape_source[:, 9:17].sum(axis=1) / np.maximum(total, 1e-9)
    night = np.r_[22:24, 0:7]
    night_share = shape_source[:, night].sum(axis=1) / np.maximum(total, 1e-9)
    peak = shape_source.max(axis=1)
    base = np.percentile(shape_source, 10, axis=1)
    variability = shape_source.std(axis=1) / np.maximum(shape_source.mean(axis=1), 1e-9)

    features = np.column_stack([
        result["월말예상사용량(kWh)"].to_numpy(dtype=float),
        peak,
        evening,
        daytime,
        night_share,
        base,
        variability,
        result["알림수용도"].to_numpy(dtype=float),
    ])
    labels = simple_kmeans(features, cluster_count, seed)
    result["계량패턴군집"] = labels + 1
    return result


def pattern_insights(actual: np.ndarray, baseline: np.ndarray, cutoff: int, season: str) -> pd.DataFrame:
    if cutoff < 24:
        profile = baseline[:24]
    else:
        days = min(7, cutoff // 24)
        profile = actual[cutoff - days * 24:cutoff].reshape(days, 24).mean(axis=0)
    mean = float(profile.mean())
    base = float(np.percentile(profile, 10))
    evening = float(profile[18:24].mean())
    night = float(np.r_[profile[22:24], profile[0:7]].mean())
    daytime = float(profile[9:17].mean())
    peak_hour = int(np.argmax(profile))

    rows: List[Dict[str, str]] = []
    if base > mean * 0.55:
        rows.append({"관측패턴": "기저부하가 높은 편", "가능한 원인": "상시가전·대기전력·연속운전 기기 가능성", "권고": "취침·외출 시 대기전력과 연속운전 설정 확인"})
    if evening > mean * 1.35:
        rows.append({"관측패턴": "18~24시 사용 집중", "가능한 원인": "조리·세탁·건조·게임·TV·냉난방 등이 중첩될 가능성", "권고": "예약 가능한 가전은 22시 이후로 이동하고 이용시간 한도 설정"})
    if night > mean * 1.15:
        rows.append({"관측패턴": "야간 지속부하 관측", "가능한 원인": "충전·건조·공기관리기기·대기전력 가능성", "권고": "예약종료와 충전 완료 후 자동차단 설정 확인"})
    if daytime > mean * 1.25:
        rows.append({"관측패턴": "주간 사용비중이 높음", "가능한 원인": "재택·냉난방·생활가전 이용 가능성", "권고": "외출모드·자동절전·실내온도 범위를 활용"})
    if season in ["여름", "겨울"] and profile[14:22].mean() > mean * 1.25:
        rows.append({"관측패턴": f"{season} 피크시간 부하가 높음", "가능한 원인": "냉난방과 저녁 생활부하의 동시 사용 가능성", "권고": "예냉·예열 후 피크시간 출력을 완화하고 고출력 가전 동시사용 회피"})
    if not rows:
        rows.append({"관측패턴": f"최대 사용시간은 {peak_hour}시", "가능한 원인": "뚜렷한 비정상 패턴 없음", "권고": "월말 예상사용량과 남은 정액 사용량을 중심으로 관리"})
    return pd.DataFrame(rows)


def action_candidates(household: pd.Series, season: str, remaining_days: float, intensity: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    days = max(int(math.ceil(remaining_days)), 0)
    weeks = max(remaining_days / 7.0, 0.0)
    scale = float(np.clip(math.sqrt(max(intensity, 0.25)), 0.70, 1.45))

    for action in ACTION_LIBRARY:
        if season not in action["seasons"]:
            continue
        if not bool(household.get(str(action["ownership"]), False)):
            continue
        frequency = str(action["frequency"])
        if frequency == "daily":
            max_units = int(math.floor(days * float(action.get("max_per_day", 1.0))))
        elif frequency == "hourly":
            max_units = int(math.floor(days * float(action.get("max_per_day", 1.0))))
        else:
            max_units = int(math.ceil(weeks * float(action.get("events_per_week", 1.0))))
        if max_units <= 0:
            continue
        low = float(action["low"]) * scale
        high = float(action["high"]) * scale
        rows.append({
            "행동ID": action["id"],
            "행동대안": action["name"],
            "1단위절감하한(kWh)": low,
            "1단위절감상한(kWh)": high,
            "1단위기대절감(kWh)": (low + high) / 2.0,
            "최대단위": max_units,
            "단위": action["unit"],
            "불편점수": int(action["discomfort"]),
            "신뢰도": float(action["reliability"]),
            "설명": action["description"],
        })
    return pd.DataFrame(rows)


def optimize_action_package(target_kwh: float, candidates: pd.DataFrame, style_name: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if candidates.empty or target_kwh <= 1e-6:
        return pd.DataFrame(columns=["행동대안", "권고량", "단위", "예상절감하한(kWh)", "예상절감(kWh)", "예상절감상한(kWh)", "불편점수", "설명"]), {
            "목표감축량(kWh)": max(target_kwh, 0.0), "계획감축량(kWh)": 0.0, "계획하한(kWh)": 0.0, "계획상한(kWh)": 0.0, "미달량(kWh)": max(target_kwh, 0.0), "총불편점수": 0.0,
        }

    cfg = PACKAGE_STYLES[style_name]
    target = float(target_kwh * cfg["safety"])
    scale = 1000
    target_wh = int(round(target * scale))
    model = cp_model.CpModel()
    variables = []
    saving_terms = []
    discomfort_terms = []
    count_terms = []
    reliability_terms = []

    for idx, row in candidates.reset_index(drop=True).iterrows():
        var = model.NewIntVar(0, int(row["최대단위"]), f"a_{idx}")
        variables.append(var)
        saving_wh = max(1, int(round(float(row["1단위기대절감(kWh)"]) * scale)))
        saving_terms.append(saving_wh * var)
        discomfort_terms.append(int(row["불편점수"]) * var)
        count_terms.append(var)
        reliability_terms.append(int(round((1.0 - float(row["신뢰도"])) * 100)) * var)

    total_saving = sum(saving_terms)
    max_total = int(sum(int(row["최대단위"]) * max(1, int(round(float(row["1단위기대절감(kWh)"]) * scale))) for _, row in candidates.iterrows()))
    shortfall = model.NewIntVar(0, max(target_wh, 1), "shortfall")
    excess = model.NewIntVar(0, max(max_total, target_wh, 1), "excess")
    model.Add(total_saving + shortfall >= target_wh)
    model.Add(excess >= total_saving - target_wh)

    model.Minimize(
        shortfall * 10_000
        + excess
        + sum(discomfort_terms) * int(cfg["discomfort_weight"])
        + sum(count_terms) * int(cfg["count_weight"])
        + sum(reliability_terms) * int(cfg["reliability_weight"])
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    selected_rows: List[Dict[str, object]] = []
    low_total = mid_total = high_total = discomfort_total = 0.0
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for idx, row in candidates.reset_index(drop=True).iterrows():
            units = int(solver.Value(variables[idx]))
            if units <= 0:
                continue
            low = units * float(row["1단위절감하한(kWh)"])
            mid = units * float(row["1단위기대절감(kWh)"])
            high = units * float(row["1단위절감상한(kWh)"])
            discomfort = units * int(row["불편점수"])
            low_total += low
            mid_total += mid
            high_total += high
            discomfort_total += discomfort
            selected_rows.append({
                "행동대안": row["행동대안"],
                "권고량": units,
                "단위": row["단위"],
                "예상절감하한(kWh)": low,
                "예상절감(kWh)": mid,
                "예상절감상한(kWh)": high,
                "불편점수": discomfort,
                "설명": row["설명"],
            })

    metrics = {
        "목표감축량(kWh)": target_kwh,
        "안전여유반영목표(kWh)": target,
        "계획감축량(kWh)": mid_total,
        "계획하한(kWh)": low_total,
        "계획상한(kWh)": high_total,
        "미달량(kWh)": max(target - mid_total, 0.0),
        "총불편점수": discomfort_total,
    }
    return pd.DataFrame(selected_rows), metrics


def goal_target(
    goal_type: str,
    household: pd.Series,
    current_plan: str,
    basic_fee: float,
    basic_quota: float,
    premium_fee: float,
    premium_quota: float,
    overage_rate: float,
    custom_bill: float,
) -> Tuple[float, float, str]:
    fee, quota = plan_values(current_plan, basic_fee, basic_quota, premium_fee, premium_quota)
    if goal_type == "현재 요금제 제공량 이내":
        return quota, fee, f"{current_plan} 제공량 {quota:,.0f}kWh 이내"
    if goal_type == "전월과 같은 요금":
        target_bill = subscription_bill(float(household["전월사용량(kWh)"]), fee, quota, overage_rate)
        target_usage = inverse_bill_to_usage(target_bill, fee, quota, overage_rate)
        return target_usage, target_bill, "전월 납부액 수준"
    if goal_type == "전년 동월과 같은 요금":
        target_bill = subscription_bill(float(household["전년동월사용량(kWh)"]), fee, quota, overage_rate)
        target_usage = inverse_bill_to_usage(target_bill, fee, quota, overage_rate)
        return target_usage, target_bill, "전년 동월 납부액 수준"
    target_bill = max(float(custom_bill), fee)
    target_usage = inverse_bill_to_usage(target_bill, fee, quota, overage_rate)
    return target_usage, target_bill, "사용자 지정 목표요금"


def grid_alert_selection(
    monitoring: pd.DataFrame,
    actual: np.ndarray,
    baseline: np.ndarray,
    cutoff: int,
    capacity_kw: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    n = len(monitoring)
    current_index = max(min(cutoff - 1, actual.shape[1] - 1), 0)
    current_load = actual[:, current_index]
    # 다음 3시간은 고객별 예측 추세계수를 기준곡선에 적용
    horizon_indices = np.arange(current_index, min(current_index + 3, actual.shape[1]))
    predicted = baseline[:, horizon_indices] * monitoring["추세계수"].to_numpy(dtype=float)[:, None]
    predicted_peak = float(predicted.sum(axis=0).max()) if predicted.size else float(current_load.sum())
    required = max(predicted_peak - capacity_kw, 0.0)

    expected_response = np.maximum(current_load, predicted.max(axis=1) if predicted.size else current_load) * (
        0.04 + 0.13 * monitoring["알림수용도"].to_numpy(dtype=float)
    )
    expected_response = np.clip(expected_response, 0.02, None)

    if required <= 1e-6:
        return pd.DataFrame(columns=["고객ID", "현재부하(kW)", "예상반응(kW)", "알림수용도", "과거알림횟수", "권고"]), {
            "예측피크(kW)": predicted_peak, "변압기용량(kW)": capacity_kw, "필요감축(kW)": 0.0, "선정가구수": 0, "기대감축(kW)": 0.0,
        }

    scale = 1000
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    response_terms = [int(round(expected_response[i] * scale)) * x[i] for i in range(n)]
    target = int(round(required * scale))
    shortfall = model.NewIntVar(0, max(target, 1), "shortfall")
    model.Add(sum(response_terms) + shortfall >= target)
    costs = []
    for i in range(n):
        history = int(monitoring.iloc[i]["과거알림횟수"])
        acceptance = float(monitoring.iloc[i]["알림수용도"])
        alert_cost = 20 + 12 * history + int(round((1.0 - acceptance) * 80))
        costs.append(alert_cost * x[i])
    model.Minimize(shortfall * 10_000 + sum(costs))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    rows = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for i in range(n):
            if solver.Value(x[i]) == 1:
                rows.append({
                    "고객ID": monitoring.iloc[i]["고객ID"],
                    "현재부하(kW)": float(current_load[i]),
                    "예상반응(kW)": float(expected_response[i]),
                    "알림수용도": float(monitoring.iloc[i]["알림수용도"]),
                    "과거알림횟수": int(monitoring.iloc[i]["과거알림횟수"]),
                    "권고": "향후 피크시간 사용량 감축 또는 예약가전 운전시간 이동",
                })
    selected = pd.DataFrame(rows)
    expected = float(selected["예상반응(kW)"].sum()) if not selected.empty else 0.0
    return selected, {
        "예측피크(kW)": predicted_peak,
        "변압기용량(kW)": capacity_kw,
        "필요감축(kW)": required,
        "선정가구수": len(selected),
        "기대감축(kW)": expected,
    }


def alert_message(row: pd.Series, fee: float, quota: float, overage_rate: float) -> str:
    remaining = float(row["남은정액사용량(kWh)"])
    days = float(row["남은일수"])
    forecast = float(row["월말예상사용량(kWh)"])
    bill = float(row["월말예상요금(원)"])
    if remaining >= 0:
        remain_text = f"정액 제공량은 {remaining:,.1f}kWh 남아 있으며, 남은 기간의 정액 내 사용 가능량은 하루 평균 {max(remaining, 0)/max(days, 1e-9):,.1f}kWh입니다."
    else:
        remain_text = f"정액 제공량을 {-remaining:,.1f}kWh 초과했습니다."
    return (
        f"현재까지 {row['현재누적사용량(kWh)']:,.1f}kWh를 사용했습니다. {remain_text} "
        f"현재 추세가 유지되면 월말 사용량은 약 {forecast:,.1f}kWh, 예상 납부액은 약 {bill:,.0f}원입니다. "
        f"예측 범위는 {row['예측하한(kWh)']:,.1f}~{row['예측상한(kWh)']:,.1f}kWh입니다."
    )


def make_zip(tables: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, table in tables.items():
            archive.writestr(f"{name}.csv", table.to_csv(index=False).encode("utf-8-sig"))
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="단일계량값 기반 100가구 전력예산 관리 v6", page_icon="⚡", layout="wide")
    st.title("⚡ 단일계량값 기반 100가구 전력예산·알림·행동추천 시뮬레이터")
    st.caption(f"앱 버전 {APP_VERSION}")
    st.success(
        "현실성 개선: 고객별 가전기기 사용량은 알고리즘에 제공하지 않습니다. 알고리즘은 가구별 총계량값, 요금제, 과거 총사용패턴, 가입 시 등록한 가전 보유정보만 이용해 월말 사용량·요금을 예측하고 행동대안을 추천합니다. 가전별 자료는 시뮬레이션 생성과 성능검증의 숨은 정답으로만 사용합니다."
    )

    with st.sidebar:
        st.header("시뮬레이션 설정")
        household_count = st.slider("가구 수", 50, 300, 100, 10)
        cluster_count = st.slider("계량패턴 군집 수", 3, 8, 5, 1)
        season = st.selectbox("계절", ["봄가을", "여름", "겨울"], index=1)
        source_percent = st.slider("원본 Excel 평균사용량 반영률", 60, 120, 100, 5, format="%d%%")
        seed = st.number_input("가상가구 생성번호", 1, 9999, 42, 1)

        st.divider()
        st.subheader("현재 조회시점")
        current_day = st.slider("월 중 일자", 1, 30, 25, 1)
        current_hour = st.slider("현재 시각(해당 시간 사용 완료 기준)", 0, 23, 18, 1)

        st.divider()
        st.subheader("요금제")
        basic_fee = st.number_input("기본형 월 구독료(원)", 0, 84_900, 1_000)
        basic_quota = st.number_input("기본형 제공량(kWh)", 1, 450, 10)
        premium_fee = st.number_input("프리미엄형 월 구독료(원)", 0, 249_900, 1_000)
        premium_quota = st.number_input("프리미엄형 제공량(kWh)", 1, 1_000, 10)
        overage_label = st.selectbox("초과단가", ["200원/kWh", "300원/kWh", "400원/kWh", "현행 한계단가 307.3원/kWh"], index=1)
        overage_rate = {"200원/kWh": 200.0, "300원/kWh": 300.0, "400원/kWh": 400.0, "현행 한계단가 307.3원/kWh": 307.3}[overage_label]
        assignment_mode = st.selectbox(
            "현재 요금제 배정 방식",
            ["사용량 기준(450kWh 이하 기본형)", "전 가구 기본형", "전 가구 프리미엄형", "지정 비율(저사용량 기본형 우선)"],
            index=0,
        )
        basic_share = st.slider("기본형 가입비중", 0, 100, 50, 5, format="%d%%") if assignment_mode.startswith("지정") else 50

        st.divider()
        st.subheader("계통 알림")
        capacity_ratio = st.slider("변압기 용량/기준 최대피크", 70, 110, 90, 1, format="%d%%")
        run = st.button("단일계량값 시뮬레이션 실행", type="primary", use_container_width=True)

    if not run and "meter_v6" not in st.session_state:
        st.subheader("이 버전에서 구현한 기능")
        st.markdown(
            """
            - 시간별 **총계량값 하나**만으로 누적사용량, 남은 정액량, 월말 사용량과 요금을 추정
            - 전월·전년 동월·정액 제공량·사용자 지정 요금을 목표로 필요한 감축량 계산
            - 가입 시 등록한 가전정보와 표준 절감 라이브러리를 이용해 **최소불편형·균형형·목표달성 우선형** 행동대안 제시
            - 계산은 매시간 갱신하되 알림은 관심·주의·경고·긴급 단계가 바뀔 때 제공
            - 100가구를 총계량 패턴만으로 군집화하고, 변압기 과부하 예상 시 공정하게 계통 알림 대상 선정
            - 기본형·프리미엄형 예상요금을 동시에 비교해 유리한 요금제 권고
            """
        )
        return

    if run:
        with st.spinner("원본 Excel 기반 100가구 생성 → 30일 시간별 단일계량값 생성 → 월말 예측·군집화·요금 계산 중..."):
            source_multiplier = source_percent / 100.0
            households, population, calibration = generate_population(
                int(household_count), season, int(seed), source_multiplier
            )
            households = add_customer_metadata(households, season, int(seed))
            households = assign_current_plans(
                households, assignment_mode, float(basic_quota), int(basic_share)
            )
            actual, baseline, daily = build_monthly_meter_streams(households, population, season, int(seed))
            cutoff = (int(current_day) - 1) * 24 + int(current_hour) + 1
            monitoring = build_monitoring_table(
                households, actual, baseline, cutoff,
                float(basic_fee), float(basic_quota), float(premium_fee), float(premium_quota), float(overage_rate),
            )
            monitoring = add_meter_clusters(monitoring, actual, baseline, cutoff, int(cluster_count), int(seed))

            # 군집 요약은 가전별 정보 없이 계량·예측지표만 사용합니다.
            cluster_summary = monitoring.groupby("계량패턴군집", as_index=False).agg(
                가구수=("고객ID", "count"),
                평균현재누적사용량_kWh=("현재누적사용량(kWh)", "mean"),
                평균월말예상사용량_kWh=("월말예상사용량(kWh)", "mean"),
                평균예상요금_원=("월말예상요금(원)", "mean"),
                평균알림수용도=("알림수용도", "mean"),
                경고이상가구수=("알림점수", lambda s: int((s >= ALERT_SCORE["경고"]).sum())),
            )

            aggregate_baseline_peak = float(baseline.reshape(len(households), DAYS_IN_MONTH, 24).sum(axis=0).max())
            capacity_kw = aggregate_baseline_peak * capacity_ratio / 100.0
            grid_selected, grid_metrics = grid_alert_selection(monitoring, actual, baseline, cutoff, capacity_kw)

            st.session_state["meter_v6"] = {
                "households": households,
                "population": population,
                "calibration": calibration,
                "actual": actual,
                "baseline": baseline,
                "daily": daily,
                "monitoring": monitoring,
                "cluster_summary": cluster_summary,
                "grid_selected": grid_selected,
                "grid_metrics": grid_metrics,
                "cutoff": cutoff,
                "current_day": current_day,
                "current_hour": current_hour,
                "season": season,
                "basic_fee": basic_fee,
                "basic_quota": basic_quota,
                "premium_fee": premium_fee,
                "premium_quota": premium_quota,
                "overage_rate": overage_rate,
                "overage_label": overage_label,
                "source_percent": source_percent,
                "seed": seed,
                "capacity_kw": capacity_kw,
            }

    result = st.session_state.get("meter_v6")
    if not result:
        return

    households = result["households"]
    monitoring = result["monitoring"]
    actual = result["actual"]
    baseline = result["baseline"]
    cutoff = int(result["cutoff"])
    season = str(result["season"])
    basic_fee = float(result["basic_fee"])
    basic_quota = float(result["basic_quota"])
    premium_fee = float(result["premium_fee"])
    premium_quota = float(result["premium_quota"])
    overage_rate = float(result["overage_rate"])

    st.subheader("1. 100가구 단일계량값 모니터링")
    level_counts = monitoring["알림단계"].value_counts().reindex(ALERT_ORDER, fill_value=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재까지 100가구 누적", f"{monitoring['현재누적사용량(kWh)'].sum():,.0f} kWh")
    c2.metric("월말 예상사용량", f"{monitoring['월말예상사용량(kWh)'].sum():,.0f} kWh")
    c3.metric("예상 구독요금 합계", f"{monitoring['월말예상요금(원)'].sum():,.0f}원")
    c4.metric("경고·긴급 가구", f"{int(level_counts['경고'] + level_counts['긴급'])}가구")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("기본형 가입", f"{int((monitoring['현재요금제']=='기본형').sum())}가구")
    c6.metric("프리미엄형 가입", f"{int((monitoring['현재요금제']=='프리미엄형').sum())}가구")
    switches = monitoring[monitoring["추천요금제"] != monitoring["현재요금제"]]
    c7.metric("요금제 전환 권고", f"{len(switches)}가구")
    c8.metric("전환 시 절감 가능", f"{switches['요금제전환예상절감액(원)'].sum():,.0f}원/월")

    alert_fig = go.Figure(go.Bar(x=ALERT_ORDER, y=level_counts.values, text=level_counts.values, textposition="auto"))
    alert_fig.update_layout(title="알림 단계별 가구 수", xaxis_title="알림 단계", yaxis_title="가구 수", height=330)
    st.plotly_chart(alert_fig, use_container_width=True)

    st.subheader("2. 총계량 패턴 군집")
    st.caption("군집화에는 월말 예상사용량, 총계량 부하의 시간대별 비중·최대값·기저부하·변동성, 알림수용도만 사용합니다.")
    cluster_display = result["cluster_summary"].copy()
    numeric_cols = cluster_display.select_dtypes(include=["number"]).columns
    cluster_display[numeric_cols] = cluster_display[numeric_cols].round(2)
    st.dataframe(cluster_display, use_container_width=True, hide_index=True)

    st.subheader("3. 배전계통 피크 알림 배분")
    gm = result["grid_metrics"]
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("향후 3시간 예측피크", f"{gm['예측피크(kW)']:,.1f} kW")
    g2.metric("가정 변압기 용량", f"{gm['변압기용량(kW)']:,.1f} kW")
    g3.metric("필요 피크감축", f"{gm['필요감축(kW)']:,.1f} kW")
    g4.metric("알림 대상", f"{int(gm['선정가구수'])}가구")
    if gm["필요감축(kW)"] > 0:
        st.warning(
            f"향후 3시간 동안 약 {gm['필요감축(kW)']:.1f}kW의 피크 완화가 필요합니다. "
            f"총계량값·알림수용도·과거 알림횟수만 이용해 {int(gm['선정가구수'])}가구를 선정했으며 기대 반응은 {gm['기대감축(kW)']:.1f}kW입니다."
        )
        st.dataframe(result["grid_selected"].round(3), use_container_width=True, hide_index=True)
    else:
        st.info("현재 조회시점에는 변압기 용량을 넘는 피크가 예상되지 않아 추가 계통 알림이 필요하지 않습니다.")

    st.subheader("4. 샘플 고객의 매시간 전력예산·요금 안내")
    customer_options = monitoring.sort_values(["알림점수", "월말예상사용량(kWh)"], ascending=False)["고객ID"].tolist()
    selected_id = st.selectbox("확인할 고객", customer_options)
    idx = int(households.index[households["고객ID"] == selected_id][0])
    customer = households.iloc[idx]
    mon = monitoring[monitoring["고객ID"] == selected_id].iloc[0]
    fee, quota = plan_values(str(customer["현재요금제"]), basic_fee, basic_quota, premium_fee, premium_quota)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("현재 요금제", str(customer["현재요금제"]))
    s2.metric("현재 누적", f"{mon['현재누적사용량(kWh)']:,.1f} kWh")
    s3.metric("남은 정액량", f"{mon['남은정액사용량(kWh)']:,.1f} kWh")
    s4.metric("월말 예상", f"{mon['월말예상사용량(kWh)']:,.1f} kWh")
    s5.metric("예상 납부액", f"{mon['월말예상요금(원)']:,.0f}원")

    st.markdown(f"**고객 알림문 예시 — {mon['알림단계']} 단계**")
    st.info(alert_message(mon, fee, quota, overage_rate))

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("예측 범위", f"{mon['예측하한(kWh)']:,.0f}~{mon['예측상한(kWh)']:,.0f} kWh")
    p2.metric("전월 사용량", f"{customer['전월사용량(kWh)']:,.1f} kWh")
    p3.metric("전년 동월 사용량", f"{customer['전년동월사용량(kWh)']:,.1f} kWh")
    p4.metric("비용상 추천요금제", str(mon["추천요금제"]), f"전환 시 {mon['요금제전환예상절감액(원)']:,.0f}원")

    # 누적계량 차트
    daily_actual = actual[idx].reshape(DAYS_IN_MONTH, 24).sum(axis=1)
    daily_baseline = baseline[idx].reshape(DAYS_IN_MONTH, 24).sum(axis=1)
    elapsed_days_float = cutoff / 24.0
    day_cum_actual = np.cumsum(daily_actual)
    day_cum_baseline = np.cumsum(daily_baseline)
    observed_day_count = int(math.ceil(elapsed_days_float))
    cum_fig = go.Figure()
    cum_fig.add_trace(go.Scatter(x=np.arange(1, observed_day_count + 1), y=day_cum_actual[:observed_day_count], mode="lines+markers", name="실제 누적계량"))
    cum_fig.add_trace(go.Scatter(x=np.arange(1, DAYS_IN_MONTH + 1), y=day_cum_baseline, mode="lines", name="기준 사용패턴", line=dict(dash="dash")))
    cum_fig.add_hline(y=quota, line_dash="dot", annotation_text=f"{customer['현재요금제']} 제공량")
    cum_fig.update_layout(title="월 누적 총계량값", xaxis_title="일자", yaxis_title="누적 kWh", height=390)
    st.plotly_chart(cum_fig, use_container_width=True)

    # 현재일 시간별 계량곡선: 실제로는 현재시각까지만 관측 가능
    day_idx = int(result["current_day"]) - 1
    start = day_idx * 24
    day_actual = actual[idx, start:start + 24]
    day_base = baseline[idx, start:start + 24]
    observed_mask = HOURS <= int(result["current_hour"])
    observed_values = np.where(observed_mask, day_actual, np.nan)
    forecast_values = np.where(observed_mask, np.nan, day_base * float(mon["추세계수"]))
    day_fig = go.Figure()
    day_fig.add_trace(go.Scatter(x=HOURS, y=observed_values, mode="lines+markers", name="관측된 총계량값"))
    day_fig.add_trace(go.Scatter(x=HOURS, y=forecast_values, mode="lines+markers", name="남은 시간 예측", line=dict(dash="dash")))
    day_fig.add_trace(go.Scatter(x=HOURS, y=day_base, mode="lines", name="과거 기준패턴", line=dict(dash="dot")))
    if st.checkbox("시뮬레이션 검증용 미래 실제값 표시", value=False):
        day_fig.add_trace(go.Scatter(x=HOURS, y=day_actual, mode="lines", name="숨은 실제값(검증용)", opacity=0.45))
    day_fig.update_layout(title=f"{result['current_day']}일 시간별 단일 계량값", xaxis_title="시간", yaxis_title="kW(1시간 평균)", height=420)
    st.plotly_chart(day_fig, use_container_width=True)

    st.markdown("**총계량 패턴에서 관측된 특징**")
    insights = pattern_insights(actual[idx], baseline[idx], cutoff, season)
    st.dataframe(insights, use_container_width=True, hide_index=True)
    st.caption("위 내용은 총계량값에서 추정한 가능성으로, 특정 가전이 실제 사용됐다고 단정하지 않습니다.")

    st.subheader("5. 목표요금 달성을 위한 행동대안")
    goal_type = st.selectbox("목표", ["현재 요금제 제공량 이내", "전월과 같은 요금", "전년 동월과 같은 요금", "사용자 지정 목표요금"], index=0)
    custom_bill = st.number_input("사용자 지정 목표요금(원)", min_value=int(fee), value=int(max(fee, mon["월말예상요금(원)"] - 10_000)), step=1_000, disabled=goal_type != "사용자 지정 목표요금")
    target_usage, target_bill, target_label = goal_target(
        goal_type, customer, str(customer["현재요금제"]),
        basic_fee, basic_quota, premium_fee, premium_quota,
        overage_rate, float(custom_bill),
    )
    forecast_usage = float(mon["월말예상사용량(kWh)"])
    observed_usage = float(mon["현재누적사용량(kWh)"])
    remaining_days = float(mon["남은일수"])
    required_reduction = max(forecast_usage - target_usage, 0.0)
    unavoidable = max(observed_usage - target_usage, 0.0)
    future_forecast = max(forecast_usage - observed_usage, 0.0)
    target_future_allowance = max(target_usage - observed_usage, 0.0)
    daily_reduction = required_reduction / max(remaining_days, 1e-9)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("목표", target_label)
    t2.metric("목표 월사용량", f"{target_usage:,.1f} kWh")
    t3.metric("필요 감축량", f"{required_reduction:,.1f} kWh")
    t4.metric("남은 기간 일평균 감축", f"{daily_reduction:,.2f} kWh/일")

    if unavoidable > 0:
        st.error(
            f"이미 목표사용량을 {unavoidable:.1f}kWh 초과해 이번 달에는 목표요금 달성이 불가능합니다. "
            "현재 시점부터 사용량을 최소화한 예상 최저요금과 다음 달 요금제 전환을 함께 검토해야 합니다."
        )
    elif required_reduction <= 0.01:
        st.success("현재 예측상 별도의 감축 없이 목표를 달성할 수 있습니다. 남은 기간의 사용 추세를 유지하면 됩니다.")
    else:
        st.warning(
            f"현재 추세의 남은 사용량은 약 {future_forecast:.1f}kWh이며, 목표 달성을 위해 사용할 수 있는 남은 전력은 {target_future_allowance:.1f}kWh입니다. "
            f"따라서 남은 기간에 약 {required_reduction:.1f}kWh의 감축이 필요합니다."
        )

    intensity = forecast_usage / max(source_monthly_kwh(season, result["source_percent"] / 100.0), 1e-9)
    candidates = action_candidates(customer, season, remaining_days, intensity)
    package_tables: Dict[str, pd.DataFrame] = {}
    package_metrics_rows = []
    tabs = st.tabs(list(PACKAGE_STYLES.keys()))
    for tab, style_name in zip(tabs, PACKAGE_STYLES.keys()):
        table, metrics = optimize_action_package(required_reduction, candidates, style_name)
        package_tables[style_name] = table
        expected_realized = metrics["계획감축량(kWh)"] * float(customer["알림수용도"])
        projected_usage = max(forecast_usage - expected_realized, observed_usage)
        projected_bill = subscription_bill(projected_usage, fee, quota, overage_rate)
        metrics_row = {
            "대안유형": style_name,
            **metrics,
            "고객이행확률반영감축(kWh)": expected_realized,
            "이행확률반영월말사용량(kWh)": projected_usage,
            "이행확률반영예상요금(원)": projected_bill,
            "목표달성가능": "가능" if metrics["계획하한(kWh)"] >= required_reduction - 1e-6 else "불확실/곤란",
        }
        package_metrics_rows.append(metrics_row)
        with tab:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("계획 절감량", f"{metrics['계획감축량(kWh)']:,.1f} kWh")
            m2.metric("예상 범위", f"{metrics['계획하한(kWh)']:,.1f}~{metrics['계획상한(kWh)']:,.1f} kWh")
            m3.metric("이행확률 반영 예상요금", f"{projected_bill:,.0f}원")
            m4.metric("불편점수", f"{metrics['총불편점수']:,.0f}점")
            if table.empty:
                st.info("추가 행동대안이 필요하지 않거나 등록된 가전정보로 제시할 수 있는 대안이 없습니다.")
            else:
                st.dataframe(table.round(2), use_container_width=True, hide_index=True)
                if metrics["계획하한(kWh)"] < required_reduction:
                    st.warning("절감량 하한 기준으로는 목표 달성이 불확실합니다. 상위 요금제 전환 또는 목표요금 조정이 필요할 수 있습니다.")

    package_metrics = pd.DataFrame(package_metrics_rows)
    st.markdown("**대안 비교**")
    st.dataframe(package_metrics.round(2), use_container_width=True, hide_index=True)
    st.caption(
        "행동대안의 절감량은 개별 가전 실측값이 아니라 고객 등록정보, 총계량 사용강도, 최초 제공한 표준 가전자료를 이용한 추정 범위입니다. 실제 계량값으로 다음 시간부터 효과를 다시 추정해 추천치를 자동 보정해야 합니다."
    )

    st.markdown("**계통 피크 완화를 위한 별도 대안(요금 절감량과 구분)**")
    grid_actions = []
    for action in GRID_SHIFT_LIBRARY:
        if bool(customer.get(str(action["ownership"]), False)):
            grid_actions.append(action)
    st.dataframe(pd.DataFrame(grid_actions), use_container_width=True, hide_index=True)
    st.caption("시간 이동은 월 총사용량을 줄이지 않을 수 있으나 변압기·계통 피크를 낮추는 데 기여합니다.")

    st.subheader("6. 100가구 고객별 모니터링·요금제 추천")
    display_cols = [
        "고객ID", "계량패턴군집", "현재요금제", "현재누적사용량(kWh)", "남은정액사용량(kWh)",
        "월말예상사용량(kWh)", "예측하한(kWh)", "예측상한(kWh)", "월말예상요금(원)",
        "알림단계", "추천요금제", "요금제전환예상절감액(원)",
    ]
    monitoring_display = monitoring.sort_values(["알림점수", "월말예상사용량(kWh)"], ascending=False)[display_cols].copy()
    st.dataframe(monitoring_display.round(2), use_container_width=True, hide_index=True)

    st.subheader("7. 결과 다운로드")
    download_tables = {
        "100가구_모니터링": monitoring,
        "계량패턴_군집요약": result["cluster_summary"],
        "계통알림_선정가구": result["grid_selected"],
        "가구기본정보_등록가전": households,
        "일별총계량값": result["daily"],
        "선택고객_패턴추정": insights,
        "선택고객_대안비교": package_metrics,
    }
    for style_name, table in package_tables.items():
        download_tables[f"선택고객_{style_name}"] = table
    st.download_button(
        "결과자료 ZIP(CSV) 다운로드",
        data=make_zip(download_tables),
        file_name="100가구_단일계량값_전력예산_행동추천_v6.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.warning(
        "모형 한계: 가구별 시간단위 총계량값이 확보된다는 전제입니다. 아파트 단지 전체 계량값만 있고 세대별 시간대 계량값이 없다면 세대별 월말예측·개인화 알림은 불가능하며 단지 단위 서비스로 제한됩니다. 현재 행동대안의 절감량은 실증 전 추정값이므로 실제 AMI 반응자료로 지속 보정해야 합니다."
    )


if __name__ == "__main__":
    main()
