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

APP_VERSION = "2026-07-13-tariff-aware-v3.0"
HOURS = np.arange(24)
WEEKDAYS_PER_MONTH = 22
WEEKENDS_PER_MONTH = 8

# -----------------------------------------------------------------------------
# 원본 Excel(260713_4인가구_계절별_가전_부하곡선_시나리오_보정.xlsx)에서
# 직접 추출한 시간별 부하를 네 범주로 분해한 값(kWh/h = 1시간 평균 kW)
# fixed: 비제어 부하
# shift: 사용량은 보존하고 운전시간만 이동 가능한 부하
# behavior: 게임콘솔 등 사전 동의 하 사용시간 제한 가능한 부하
# hvac: 거실 에어컨·히트펌프 난방 등 출력조정 부하
# 네 범주의 합은 각 시트의 '시간별전력량(kWh)'과 정확히 일치함.
# -----------------------------------------------------------------------------
SOURCE_COMPONENTS: Dict[str, Dict[str, List[float]]] = {
    "봄가을_주중": {
        "fixed": [0.158,0.158,0.158,0.158,0.158,0.158,0.903,0.953,0.233,0.233,0.233,0.233,0.233,0.233,0.233,0.233,0.363,0.423,1.863,1.763,0.773,0.723,0.663,0.253],
        "shift": [0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0.06,0.06,0.08,0.08,0.25,0.85,0.63,0.26],
        "behavior": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.06,0.06,0,0],
        "hvac": [0]*24,
    },
    "봄가을_주말": {
        "fixed": [0.158,0.158,0.158,0.158,0.158,0.158,0.203,0.203,0.703,1.653,0.543,0.543,1.303,1.223,0.493,0.543,0.493,0.493,2.773,2.053,0.813,0.763,0.763,0.263],
        "shift": [0,0,0,0,0,0,0,0,0,0.08,0.43,0.83,0.68,0.48,0.08,0.28,0.78,0.88,0.08,0.08,0.81,0.98,0.08,0.06],
        "behavior": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.08,0.08,0.08,0.08,0.08,0],
        "hvac": [0]*24,
    },
    "여름_주중": {
        "fixed": [0.578,0.578,0.578,0.578,0.578,0.578,1.043,0.943,0.223,0.223,0.343,0.343,0.343,0.223,0.223,0.223,0.353,0.413,1.928,1.828,0.838,0.788,1.078,0.668],
        "shift": [0,0,0,0,0,0,0,0,0,0,0.05,0,0,0,0,0,0.06,0.06,0.08,0.08,0.25,0.85,0.63,0.26],
        "behavior": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.06,0.06,0,0],
        "hvac": [0.3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6,0.9,0.9,0.9,0.9,0.65],
    },
    "여름_주말": {
        "fixed": [0.578,0.578,0.578,0.578,0.578,0.578,0.603,0.193,0.693,1.643,0.593,0.793,1.553,1.473,0.743,0.593,0.543,0.543,2.958,2.118,0.878,0.828,1.178,0.678],
        "shift": [0,0,0,0,0,0,0,0,0,0.08,0.43,0.83,0.68,0.48,0.08,0.28,0.78,0.88,0.08,0.08,0.81,0.98,0.08,0.06],
        "behavior": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.08,0.08,0.08,0.08,0.08,0],
        "hvac": [0.3,0,0,0,0,0,0,0,0,0,0,0.45,0.8,0.8,0.8,0.8,0.8,0.8,0.9,0.9,0.9,0.9,0.9,0.65],
    },
    "겨울_주중": {
        "fixed": [0.378,0.378,0.378,0.378,0.378,0.378,1.328,1.448,0.228,0.228,0.228,0.228,0.228,0.228,0.228,0.228,0.358,0.538,1.938,1.798,1.058,1.068,1.288,0.498],
        "shift": [0.04,0,0,0,0,0.05,0.05,0.05,0.05,0,0.05,0,0,0,0,0,0.06,0.11,0.13,0.13,0.3,0.9,0.68,0.31],
        "behavior": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.06,0.06,0,0],
        "hvac": [0.25,0,0,0,0,0,0.7,0.7,0.3,0,0,0,0,0,0,0,0,0.3,0.8,0.8,0.8,0.8,0.8,0.5],
    },
    "겨울_주말": {
        "fixed": [0.378,0.378,0.378,0.378,0.378,0.378,0.418,0.338,0.858,1.978,0.538,0.538,1.298,1.218,0.488,0.538,0.588,0.628,2.848,2.068,1.028,1.038,1.338,0.498],
        "shift": [0.04,0,0,0,0,0,0,0.05,0.05,0.13,0.48,0.88,0.73,0.53,0.13,0.33,0.83,0.93,0.13,0.13,0.86,1.03,0.13,0.11],
        "behavior": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.08,0.08,0.08,0.08,0.08,0],
        "hvac": [0.25,0,0,0,0,0,0,0.7,0.7,0.7,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.8,0.8,0.8,0.8,0.8,0.8,0.5],
    },
}

# 서로 다른 100가구를 만들기 위한 유형. 최종 집단 평균은 아래 차이와 무관하게
# 원본 Excel 곡선에 시간대별로 재보정됨.
ARCHETYPES = [
    {"name": "절약형", "prob": 0.16, "scale": (0.66, 0.84), "day_bias": (-0.08, 0.02), "eve_bias": (-0.10, 0.03), "accept": (0.78, 0.96), "control": (0.78, 0.98)},
    {"name": "표준형", "prob": 0.34, "scale": (0.84, 1.04), "day_bias": (-0.04, 0.08), "eve_bias": (-0.04, 0.10), "accept": (0.64, 0.92), "control": (0.72, 0.98)},
    {"name": "재택형", "prob": 0.18, "scale": (0.88, 1.14), "day_bias": (0.12, 0.32), "eve_bias": (-0.05, 0.08), "accept": (0.52, 0.84), "control": (0.62, 0.94)},
    {"name": "저녁집중형", "prob": 0.20, "scale": (0.90, 1.20), "day_bias": (-0.10, 0.02), "eve_bias": (0.14, 0.34), "accept": (0.50, 0.84), "control": (0.62, 0.94)},
    {"name": "고사용량형", "prob": 0.12, "scale": (1.16, 1.46), "day_bias": (0.00, 0.16), "eve_bias": (0.06, 0.24), "accept": (0.44, 0.78), "control": (0.58, 0.90)},
]

MODE_CONFIG = {
    "편의 우선": {
        "hvac_reduction": 0.05,
        "behavior_reduction": 0.20,
        "shift_penalty": 24,
        "behavior_penalty": 150,
        "hvac_penalty": 120,
        "fairness_penalty": 35,
    },
    "균형": {
        "hvac_reduction": 0.10,
        "behavior_reduction": 0.40,
        "shift_penalty": 15,
        "behavior_penalty": 95,
        "hvac_penalty": 75,
        "fairness_penalty": 27,
    },
    "계통 안정 우선": {
        "hvac_reduction": 0.15,
        "behavior_reduction": 0.70,
        "shift_penalty": 9,
        "behavior_penalty": 55,
        "hvac_penalty": 42,
        "fairness_penalty": 20,
    },
}


PLAN_DEFAULTS = {
    "기본형": {"monthly_fee": 84_900.0, "included_kwh": 450.0},
    "프리미엄형": {"monthly_fee": 249_900.0, "included_kwh": 1_000.0},
}


def allocate_monthly_control_to_households(
    households: pd.DataFrame,
    population: Dict[str, Dict[str, np.ndarray]],
    cluster_results: pd.DataFrame,
) -> pd.DataFrame:
    """군집 최적화의 행동·냉난방 감축량을 군집 내 가구의 가용 유연성 비중으로 배분함."""
    result = households.copy()
    monthly_reduction = np.zeros(len(result), dtype=float)
    cluster_values = result["군집"].to_numpy(dtype=int)

    for day_type, day_count in [("주중", WEEKDAYS_PER_MONTH), ("주말", WEEKENDS_PER_MONTH)]:
        day_population = population[day_type]
        day_rows = cluster_results[cluster_results["대표일"] == day_type]
        for _, row in day_rows.iterrows():
            cluster_id = int(row["군집"])
            indexes = np.where(cluster_values == cluster_id)[0]
            if len(indexes) == 0:
                continue
            daily_reduction = float(row["행동부하감축(kWh)"]) + float(row["냉난방감축(kWh)"])
            if daily_reduction <= 1e-12:
                continue
            available = (
                day_population["available_behavior"][indexes].sum(axis=1)
                + day_population["available_hvac"][indexes].sum(axis=1)
            )
            available_sum = float(available.sum())
            if available_sum <= 1e-12:
                weights = np.full(len(indexes), 1.0 / len(indexes))
            else:
                weights = available / available_sum
            monthly_reduction[indexes] += daily_reduction * float(day_count) * weights

    baseline = result["월사용량(kWh)"].to_numpy(dtype=float)
    monthly_reduction = np.minimum(monthly_reduction, baseline)
    result["월감축량(kWh)"] = monthly_reduction
    result["제어후월사용량(kWh)"] = np.maximum(baseline - monthly_reduction, 0.0)
    result["월사용량감축률(%)"] = np.divide(
        monthly_reduction * 100.0,
        baseline,
        out=np.zeros_like(monthly_reduction),
        where=baseline > 1e-12,
    )
    return result


def subscription_bill(usage_kwh: float, monthly_fee: float, included_kwh: float, overage_rate: float) -> float:
    return float(monthly_fee + max(float(usage_kwh) - float(included_kwh), 0.0) * float(overage_rate))


def build_plan_comparison_tables(
    households: pd.DataFrame,
    basic_fee: float,
    basic_quota: float,
    premium_fee: float,
    premium_quota: float,
    overage_rates: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """모든 가구에 대해 기본형·프리미엄형 요금을 제어 전후로 비교하고 비용상 유리한 요금제를 추천함."""
    detail_frames: List[pd.DataFrame] = []
    usage_before = households["월사용량(kWh)"].to_numpy(dtype=float)
    usage_after = households["제어후월사용량(kWh)"].to_numpy(dtype=float)

    for scenario, rate in overage_rates.items():
        basic_before = basic_fee + np.maximum(usage_before - basic_quota, 0.0) * rate
        basic_after = basic_fee + np.maximum(usage_after - basic_quota, 0.0) * rate
        premium_before = premium_fee + np.maximum(usage_before - premium_quota, 0.0) * rate
        premium_after = premium_fee + np.maximum(usage_after - premium_quota, 0.0) * rate

        recommended = np.where(
            basic_after < premium_after - 0.5,
            "기본형",
            np.where(premium_after < basic_after - 0.5, "프리미엄형", "동일"),
        )
        recommended_bill = np.minimum(basic_after, premium_after)
        second_bill = np.maximum(basic_after, premium_after)
        usage_fit = np.where(
            usage_after <= basic_quota + 1e-9,
            "기본형",
            np.where(usage_after <= premium_quota + 1e-9, "프리미엄형", "프리미엄형+초과"),
        )

        frame = households[["고객ID", "생성유형", "군집", "월사용량(kWh)", "제어후월사용량(kWh)", "월감축량(kWh)"]].copy()
        frame["초과단가시나리오"] = scenario
        frame["초과단가(원/kWh)"] = rate
        frame["기본형요금_제어전(원)"] = basic_before
        frame["기본형요금_제어후(원)"] = basic_after
        frame["프리미엄형요금_제어전(원)"] = premium_before
        frame["프리미엄형요금_제어후(원)"] = premium_after
        frame["비용상추천요금제"] = recommended
        frame["추천요금(원)"] = recommended_bill
        frame["차선요금(원)"] = second_bill
        frame["추천에따른절감액(원)"] = second_bill - recommended_bill
        frame["사용량구간상요금제"] = usage_fit
        frame["제어로인한기본형요금절감(원)"] = basic_before - basic_after
        frame["제어로인한프리미엄형요금절감(원)"] = premium_before - premium_after
        detail_frames.append(frame)

    detail = pd.concat(detail_frames, ignore_index=True)
    summary_rows: List[Dict[str, object]] = []
    for scenario, group in detail.groupby("초과단가시나리오", sort=False):
        recommended_counts = group["비용상추천요금제"].value_counts()
        summary_rows.append({
            "초과단가시나리오": scenario,
            "초과단가(원/kWh)": float(group["초과단가(원/kWh)"].iloc[0]),
            "기본형추천가구수": int(recommended_counts.get("기본형", 0)),
            "프리미엄형추천가구수": int(recommended_counts.get("프리미엄형", 0)),
            "동일요금가구수": int(recommended_counts.get("동일", 0)),
            "가구당평균_기본형요금(원)": float(group["기본형요금_제어후(원)"].mean()),
            "가구당평균_프리미엄형요금(원)": float(group["프리미엄형요금_제어후(원)"].mean()),
            "가구당평균_추천요금(원)": float(group["추천요금(원)"].mean()),
            "100가구_전원기본형총액(원)": float(group["기본형요금_제어후(원)"].sum()),
            "100가구_전원프리미엄총액(원)": float(group["프리미엄형요금_제어후(원)"].sum()),
            "100가구_추천조합총액(원)": float(group["추천요금(원)"].sum()),
            "요금제추천으로절감되는총액(원)": float(
                np.minimum(
                    group["기본형요금_제어후(원)"].sum(),
                    group["프리미엄형요금_제어후(원)"].sum(),
                ) - group["추천요금(원)"].sum()
            ),
        })
    summary = pd.DataFrame(summary_rows)

    break_even_rows: List[Dict[str, object]] = []
    fee_gap = float(premium_fee - basic_fee)
    quota_gap = float(premium_quota - basic_quota)
    for scenario, rate in overage_rates.items():
        threshold = float("inf") if rate <= 0 else basic_quota + fee_gap / rate
        if rate <= 0:
            conclusion = "초과단가가 0원이므로 기본형이 항상 저렴함"
            premium_advantage_start = np.nan
        elif threshold < premium_quota - 1e-9:
            premium_advantage_start = threshold
            conclusion = f"월 {threshold:.1f}kWh 초과부터 프리미엄형이 저렴함"
        elif abs(threshold - premium_quota) <= 1e-9:
            premium_advantage_start = np.nan
            conclusion = f"월 {premium_quota:.0f}kWh부터 두 요금이 동일함"
        else:
            premium_advantage_start = np.nan
            conclusion = "현재 가격구조에서는 프리미엄형이 비용상 더 저렴해지는 구간이 없음"

        after_quota_difference = (basic_fee - premium_fee) + quota_gap * rate
        break_even_rows.append({
            "초과단가시나리오": scenario,
            "초과단가(원/kWh)": rate,
            "계산상교차사용량(kWh)": threshold if np.isfinite(threshold) else np.nan,
            "프리미엄유리시작사용량(kWh)": premium_advantage_start,
            "1000kWh초과시_기본형-프리미엄형(원)": after_quota_difference,
            "해석": conclusion,
        })
    break_even = pd.DataFrame(break_even_rows)
    return detail, summary, break_even


@dataclass
class DayClusterData:
    cluster_id: int
    count: int
    baseline: np.ndarray
    shiftable: np.ndarray
    behavior: np.ndarray
    hvac: np.ndarray
    acceptance: float
    daily_kwh_per_house: float


def source_components(season: str, day_type: str) -> Dict[str, np.ndarray]:
    key = f"{season}_{day_type}"
    values = SOURCE_COMPONENTS[key]
    data = {name: np.asarray(arr, dtype=float) for name, arr in values.items()}
    data["total"] = data["fixed"] + data["shift"] + data["behavior"] + data["hvac"]
    return data


def source_monthly_kwh(season: str, multiplier: float = 1.0) -> float:
    wd = source_components(season, "주중")["total"].sum()
    we = source_components(season, "주말")["total"].sum()
    return float((wd * WEEKDAYS_PER_MONTH + we * WEEKENDS_PER_MONTH) * multiplier)


def grid_weights(season: str, day_type: str) -> np.ndarray:
    weights = np.full(24, 2, dtype=int)
    weights[(HOURS >= 23) | (HOURS < 8)] = 1
    if season == "여름":
        weights[(HOURS >= 14) & (HOURS < 21)] = 4
    elif season == "겨울":
        weights[((HOURS >= 8) & (HOURS < 11)) | ((HOURS >= 17) & (HOURS < 21))] = 4
    else:
        weights[(HOURS >= 17) & (HOURS < 21)] = 4
    if day_type == "주말":
        weights[weights == 4] = 3
    return weights


def smooth_noise(rng: np.random.Generator, sigma: float = 0.055) -> np.ndarray:
    raw = rng.normal(1.0, sigma, 24)
    padded = np.r_[raw[-1], raw, raw[0]]
    smoothed = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4
    return np.clip(smoothed, 0.82, 1.18)


def generate_population(
    household_count: int,
    season: str,
    seed: int,
    source_multiplier: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    """100개 가구를 생성하되, 집단 평균을 원본 Excel 곡선에 시간대별로 정확히 맞춤."""
    rng = np.random.default_rng(seed)
    probs = np.asarray([a["prob"] for a in ARCHETYPES], dtype=float)
    probs /= probs.sum()
    archetype_indices = rng.choice(len(ARCHETYPES), size=household_count, p=probs)

    household_params: List[Dict[str, float | int | str]] = []
    for idx in archetype_indices:
        archetype = ARCHETYPES[int(idx)]
        household_params.append({
            "생성유형": archetype["name"],
            "scale": rng.uniform(*archetype["scale"]),
            "day_bias": rng.uniform(*archetype["day_bias"]),
            "eve_bias": rng.uniform(*archetype["eve_bias"]),
            "acceptance": rng.uniform(*archetype["accept"]),
            "control": rng.uniform(*archetype["control"]),
            "shift_hours": int(rng.choice([-2, -1, 0, 0, 0, 1, 2])),
            "shift_usage": rng.uniform(0.80, 1.20),
            "behavior_usage": rng.uniform(0.65, 1.35),
            "hvac_usage": rng.uniform(0.78, 1.24),
        })

    population: Dict[str, Dict[str, np.ndarray]] = {}
    calibration_rows: List[Dict[str, float | str]] = []

    for day_type in ["주중", "주말"]:
        source = source_components(season, day_type)
        raw = {name: np.zeros((household_count, 24), dtype=float) for name in ["fixed", "shift", "behavior", "hvac"]}

        for i, param in enumerate(household_params):
            scale = float(param["scale"])
            day_bias = float(param["day_bias"])
            eve_bias = float(param["eve_bias"])
            time_shift = int(param["shift_hours"])

            hour_bias = np.ones(24, dtype=float)
            hour_bias[9:17] *= 1.0 + day_bias
            hour_bias[18:24] *= 1.0 + eve_bias

            # 상시·필수부하는 시간 이동하지 않음. 이동형 부하만 고객별 생활시간 차이를 반영함.
            raw["fixed"][i] = source["fixed"] * scale * hour_bias * smooth_noise(rng, 0.040)
            raw["shift"][i] = np.roll(source["shift"], time_shift) * scale * float(param["shift_usage"]) * hour_bias * smooth_noise(rng, 0.060)
            raw["behavior"][i] = np.roll(source["behavior"], time_shift) * scale * float(param["behavior_usage"]) * hour_bias * smooth_noise(rng, 0.075)
            raw["hvac"][i] = source["hvac"] * scale * float(param["hvac_usage"]) * hour_bias * smooth_noise(rng, 0.050)

        raw_total = sum(raw.values())
        raw_aggregate = raw_total.sum(axis=0)
        target_aggregate = source["total"] * household_count * source_multiplier
        factor = np.divide(target_aggregate, raw_aggregate, out=np.ones(24), where=raw_aggregate > 1e-12)

        # 모든 구성부하에 같은 시간대별 보정계수를 적용하여 구성비를 유지하면서
        # 100가구 평균 곡선을 원본 Excel 곡선에 정확히 맞춤.
        for name in raw:
            raw[name] *= factor[None, :]
        total = sum(raw.values())
        available = np.asarray([float(p["control"]) for p in household_params], dtype=float)[:, None]

        population[day_type] = {
            "fixed": raw["fixed"],
            "shift": raw["shift"],
            "behavior": raw["behavior"],
            "hvac": raw["hvac"],
            "total": total,
            "available_shift": raw["shift"] * available,
            "available_behavior": raw["behavior"] * available,
            "available_hvac": raw["hvac"] * available,
        }

        generated_avg = total.mean(axis=0)
        for hour in range(24):
            target = float(source["total"][hour] * source_multiplier)
            actual = float(generated_avg[hour])
            calibration_rows.append({
                "대표일": day_type,
                "시간": hour,
                "원본Excel_가구당(kW)": target,
                "생성100가구_가구당평균(kW)": actual,
                "차이(kW)": actual - target,
                "차이율(%)": 0.0 if abs(target) < 1e-12 else (actual - target) / target * 100,
            })

    rows: List[Dict[str, object]] = []
    wd_total = population["주중"]["total"]
    we_total = population["주말"]["total"]
    monthly = wd_total.sum(axis=1) * WEEKDAYS_PER_MONTH + we_total.sum(axis=1) * WEEKENDS_PER_MONTH
    monthly_shift = population["주중"]["available_shift"].sum(axis=1) * WEEKDAYS_PER_MONTH + population["주말"]["available_shift"].sum(axis=1) * WEEKENDS_PER_MONTH
    monthly_behavior = population["주중"]["available_behavior"].sum(axis=1) * WEEKDAYS_PER_MONTH + population["주말"]["available_behavior"].sum(axis=1) * WEEKENDS_PER_MONTH
    monthly_hvac = population["주중"]["available_hvac"].sum(axis=1) * WEEKDAYS_PER_MONTH + population["주말"]["available_hvac"].sum(axis=1) * WEEKENDS_PER_MONTH

    for i, param in enumerate(household_params):
        peak_wd = float(wd_total[i].max())
        peak_we = float(we_total[i].max())
        total_month = float(monthly[i])
        evening_energy = (
            wd_total[i, 18:24].sum() * WEEKDAYS_PER_MONTH
            + we_total[i, 18:24].sum() * WEEKENDS_PER_MONTH
        )
        daytime_energy = (
            wd_total[i, 9:17].sum() * WEEKDAYS_PER_MONTH
            + we_total[i, 9:17].sum() * WEEKENDS_PER_MONTH
        )
        rows.append({
            "고객ID": f"H{i+1:03d}",
            "생성유형": param["생성유형"],
            "월사용량(kWh)": total_month,
            "주중일사용량(kWh)": float(wd_total[i].sum()),
            "주말일사용량(kWh)": float(we_total[i].sum()),
            "주중최대부하(kW)": peak_wd,
            "주말최대부하(kW)": peak_we,
            "주간비중(09-17)": daytime_energy / max(total_month, 1e-9),
            "저녁비중(18-24)": evening_energy / max(total_month, 1e-9),
            "이동가능비중": float(monthly_shift[i] / max(total_month, 1e-9)),
            "행동제어가능비중": float(monthly_behavior[i] / max(total_month, 1e-9)),
            "냉난방제어가능비중": float(monthly_hvac[i] / max(total_month, 1e-9)),
            "제어수용도": float(param["acceptance"]),
            "원격제어가용도": float(param["control"]),
            "사용량구간상_초기분류": "기본형" if total_month <= 450 else "프리미엄형",
        })

    households = pd.DataFrame(rows)
    calibration = pd.DataFrame(calibration_rows)
    return households, population, calibration


def simple_kmeans(features: np.ndarray, cluster_count: int, seed: int, max_iter: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed + 1009)
    x = np.asarray(features, dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std

    centers = [z[rng.integers(0, len(z))]]
    for _ in range(1, cluster_count):
        distance_squared = np.min([np.sum((z - center) ** 2, axis=1) for center in centers], axis=0)
        if float(distance_squared.sum()) <= 1e-12:
            centers.append(z[rng.integers(0, len(z))])
        else:
            centers.append(z[rng.choice(len(z), p=distance_squared / distance_squared.sum())])
    centers = np.asarray(centers)

    labels = np.zeros(len(z), dtype=int)
    for _ in range(max_iter):
        distances = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(cluster_count):
            members = z[labels == cluster]
            if len(members) == 0:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                centers[cluster] = z[farthest]
                labels[farthest] = cluster
            else:
                centers[cluster] = members.mean(axis=0)

    # 월사용량이 낮은 군집부터 1번 부여
    cluster_monthly = []
    for cluster in range(cluster_count):
        members = x[labels == cluster]
        cluster_monthly.append(float(members[:, 0].mean()) if len(members) else float("inf"))
    order = np.argsort(cluster_monthly)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    return np.asarray([remap[int(value)] for value in labels], dtype=int)


def cluster_population(
    households: pd.DataFrame,
    population: Dict[str, Dict[str, np.ndarray]],
    cluster_count: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, List[DayClusterData]]]:
    feature_columns = [
        "월사용량(kWh)", "주중최대부하(kW)", "주말최대부하(kW)",
        "주간비중(09-17)", "저녁비중(18-24)", "이동가능비중",
        "냉난방제어가능비중", "제어수용도",
    ]
    labels = simple_kmeans(households[feature_columns].to_numpy(dtype=float), cluster_count, seed)
    households = households.copy()
    households["군집"] = labels + 1

    cluster_days: Dict[str, List[DayClusterData]] = {"주중": [], "주말": []}
    for day_type in ["주중", "주말"]:
        day = population[day_type]
        for cluster in range(cluster_count):
            indexes = np.where(labels == cluster)[0]
            members = households.iloc[indexes]
            cluster_days[day_type].append(DayClusterData(
                cluster_id=cluster + 1,
                count=len(indexes),
                baseline=day["total"][indexes].sum(axis=0),
                shiftable=day["available_shift"][indexes].sum(axis=0),
                behavior=day["available_behavior"][indexes].sum(axis=0),
                hvac=day["available_hvac"][indexes].sum(axis=0),
                acceptance=float(members["제어수용도"].mean()),
                daily_kwh_per_house=float(day["total"][indexes].sum(axis=1).mean()),
            ))
    return households, cluster_days


def optimize_day(
    clusters: List[DayClusterData],
    season: str,
    day_type: str,
    control_mode: str,
    transformer_capacity_kw: float,
    solve_seconds: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    cfg = MODE_CONFIG[control_mode]
    unit = 1000  # kW(1시간 평균) -> W 정수화
    weights = grid_weights(season, day_type)
    model = cp_model.CpModel()

    cluster_load_variables: Dict[Tuple[int, int], cp_model.IntVar] = {}
    cluster_meta: Dict[int, Dict[str, object]] = {}
    burden_variables: List[cp_model.IntVar] = []
    objective_terms = []

    for cluster_index, cluster in enumerate(clusters):
        baseline = np.rint(cluster.baseline * unit).astype(int)
        shiftable = np.rint(cluster.shiftable * unit).astype(int)
        behavior = np.rint(cluster.behavior * unit).astype(int)
        hvac = np.rint(cluster.hvac * unit).astype(int)
        fixed = np.maximum(baseline - shiftable - behavior - hvac, 0)

        total_shiftable = int(shiftable.sum())
        average_shift = math.ceil(total_shiftable / 24) if total_shiftable else 0
        shift_abs_variables: List[cp_model.IntVar] = []
        behavior_reduction_variables: List[cp_model.IntVar] = []
        hvac_reduction_variables: List[cp_model.IntVar] = []
        shift_variables: List[cp_model.IntVar] = []

        discomfort_multiplier = max(1, int(round(1.35 / max(cluster.acceptance, 0.25))))

        for hour in range(24):
            # 원래 이동부하보다 과도하게 특정 시간으로 몰리지 않도록 상한 설정
            shift_capacity = max(int(shiftable[hour] * 2.0), int(average_shift * 2.2), 1)
            shifted_load = model.NewIntVar(0, shift_capacity, f"shift_c{cluster_index}_h{hour}")
            behavior_max = int(math.floor(behavior[hour] * cfg["behavior_reduction"]))
            hvac_max = int(math.floor(hvac[hour] * cfg["hvac_reduction"]))
            behavior_reduction = model.NewIntVar(0, max(behavior_max, 0), f"behavior_red_c{cluster_index}_h{hour}")
            hvac_reduction = model.NewIntVar(0, max(hvac_max, 0), f"hvac_red_c{cluster_index}_h{hour}")

            load_upper = int(fixed[hour] + behavior[hour] + hvac[hour] + shift_capacity)
            controlled_load = model.NewIntVar(0, max(load_upper, 1), f"load_c{cluster_index}_h{hour}")
            model.Add(
                controlled_load
                == int(fixed[hour] + behavior[hour] + hvac[hour])
                + shifted_load
                - behavior_reduction
                - hvac_reduction
            )

            difference = model.NewIntVar(-max(shift_capacity, int(shiftable[hour])), max(shift_capacity, int(shiftable[hour])), f"shift_diff_c{cluster_index}_h{hour}")
            model.Add(difference == shifted_load - int(shiftable[hour]))
            absolute_difference = model.NewIntVar(0, max(shift_capacity, int(shiftable[hour])), f"shift_abs_c{cluster_index}_h{hour}")
            model.AddAbsEquality(absolute_difference, difference)

            cluster_load_variables[(cluster_index, hour)] = controlled_load
            shift_variables.append(shifted_load)
            shift_abs_variables.append(absolute_difference)
            behavior_reduction_variables.append(behavior_reduction)
            hvac_reduction_variables.append(hvac_reduction)

            objective_terms.append(absolute_difference * int(cfg["shift_penalty"] * discomfort_multiplier))
            objective_terms.append(behavior_reduction * int(cfg["behavior_penalty"] * discomfort_multiplier))
            objective_terms.append(hvac_reduction * int(cfg["hvac_penalty"] * discomfort_multiplier))

        # 시간 이동형 가전은 일사용량을 줄이지 않고 총량 보존
        model.Add(sum(shift_variables) == total_shiftable)

        absolute_sum = model.NewIntVar(0, max(total_shiftable * 4, 1), f"shift_abs_sum_c{cluster_index}")
        model.Add(absolute_sum == sum(shift_abs_variables))
        shifted_energy = model.NewIntVar(0, max(total_shiftable * 2, 1), f"shifted_energy_c{cluster_index}")
        model.AddDivisionEquality(shifted_energy, absolute_sum, 2)

        behavior_reduction_max = int(sum(math.floor(value * cfg["behavior_reduction"]) for value in behavior))
        behavior_reduction_sum = model.NewIntVar(0, max(behavior_reduction_max, 1), f"behavior_red_sum_c{cluster_index}")
        model.Add(behavior_reduction_sum == sum(behavior_reduction_variables))

        hvac_reduction_max = int(sum(math.floor(value * cfg["hvac_reduction"]) for value in hvac))
        hvac_reduction_sum = model.NewIntVar(0, max(hvac_reduction_max, 1), f"hvac_red_sum_c{cluster_index}")
        model.Add(hvac_reduction_sum == sum(hvac_reduction_variables))

        # 감축은 이동보다 고객 부담이 크므로 공정성 계산에서 가중함
        intervention_max = max(total_shiftable * 2 + behavior_reduction_max * 3 + hvac_reduction_max * 3, 1)
        intervention = model.NewIntVar(0, intervention_max, f"intervention_c{cluster_index}")
        model.Add(intervention == shifted_energy + behavior_reduction_sum * 3 + hvac_reduction_sum * 3)
        burden_per_house = model.NewIntVar(0, intervention_max, f"burden_per_house_c{cluster_index}")
        model.AddDivisionEquality(burden_per_house, intervention, max(cluster.count, 1))
        burden_variables.append(burden_per_house)

        cluster_meta[cluster_index] = {
            "shifted_energy": shifted_energy,
            "behavior_reduction_sum": behavior_reduction_sum,
            "hvac_reduction_sum": hvac_reduction_sum,
            "burden_per_house": burden_per_house,
        }

    unmanaged = np.sum([cluster.baseline for cluster in clusters], axis=0)
    max_total = int(math.ceil(max(unmanaged.max() * unit * 3, 1)))
    capacity = int(round(transformer_capacity_kw * unit))
    total_load_variables: List[cp_model.IntVar] = []
    overload_variables: List[cp_model.IntVar] = []
    peak_variable = model.NewIntVar(0, max_total, "aggregate_peak")

    for hour in range(24):
        total_load = model.NewIntVar(0, max_total, f"aggregate_load_h{hour}")
        model.Add(total_load == sum(cluster_load_variables[(cluster_index, hour)] for cluster_index in range(len(clusters))))
        overload = model.NewIntVar(0, max_total, f"overload_h{hour}")
        model.Add(overload >= total_load - capacity)
        model.Add(peak_variable >= total_load)
        total_load_variables.append(total_load)
        overload_variables.append(overload)
        objective_terms.append(overload * 5500)
        objective_terms.append(total_load * int(weights[hour]))

    objective_terms.append(peak_variable * 50)

    if len(burden_variables) >= 2:
        maximum_burden = model.NewIntVar(0, max_total, "maximum_burden")
        minimum_burden = model.NewIntVar(0, max_total, "minimum_burden")
        burden_gap = model.NewIntVar(0, max_total, "burden_gap")
        model.AddMaxEquality(maximum_burden, burden_variables)
        model.AddMinEquality(minimum_burden, burden_variables)
        model.Add(burden_gap == maximum_burden - minimum_burden)
        objective_terms.append(burden_gap * int(cfg["fairness_penalty"]))

    model.Minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(solve_seconds)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 42
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("최적화 해를 찾지 못했습니다. 변압기 용량을 높이거나 제어조건을 완화해 주세요.")

    controlled = np.asarray([solver.Value(variable) / unit for variable in total_load_variables])
    hourly = pd.DataFrame({
        "대표일": day_type,
        "시간": HOURS,
        "제어전(kW)": unmanaged,
        "제어후(kW)": controlled,
        "변압기용량(kW)": transformer_capacity_kw,
        "제어전_초과(kW)": np.maximum(unmanaged - transformer_capacity_kw, 0),
        "제어후_초과(kW)": np.maximum(controlled - transformer_capacity_kw, 0),
        "계통가중치": weights,
    })

    cluster_rows: List[Dict[str, object]] = []
    total_shifted = 0.0
    total_behavior_reduced = 0.0
    total_hvac_reduced = 0.0
    weighted_comfort = 0.0
    total_houses = sum(cluster.count for cluster in clusters)
    burdens: List[float] = []

    for cluster_index, cluster in enumerate(clusters):
        meta = cluster_meta[cluster_index]
        controlled_profile = np.asarray([solver.Value(cluster_load_variables[(cluster_index, hour)]) / unit for hour in range(24)])
        shifted = solver.Value(meta["shifted_energy"]) / unit
        behavior_reduced = solver.Value(meta["behavior_reduction_sum"]) / unit
        hvac_reduced = solver.Value(meta["hvac_reduction_sum"]) / unit
        burden = solver.Value(meta["burden_per_house"]) / unit
        daily_per_house = max(cluster.daily_kwh_per_house, 0.1)

        discomfort = (
            shifted / max(cluster.count, 1) / daily_per_house * 18
            + behavior_reduced / max(cluster.count, 1) / daily_per_house * 145
            + hvac_reduced / max(cluster.count, 1) / daily_per_house * 115
        )
        comfort = max(0.0, min(100.0, 100.0 - discomfort))
        cluster_rows.append({
            "대표일": day_type,
            "군집": cluster.cluster_id,
            "가구수": cluster.count,
            "가구당_평균일사용량(kWh)": cluster.daily_kwh_per_house,
            "평균제어수용도": cluster.acceptance,
            "제어전_군집피크(kW)": float(cluster.baseline.max()),
            "제어후_군집피크(kW)": float(controlled_profile.max()),
            "이동전력량(kWh)": shifted,
            "행동부하감축(kWh)": behavior_reduced,
            "냉난방감축(kWh)": hvac_reduced,
            "가구당_제어부담점수": burden,
            "편의점수(100점)": comfort,
        })
        total_shifted += shifted
        total_behavior_reduced += behavior_reduced
        total_hvac_reduced += hvac_reduced
        weighted_comfort += comfort * cluster.count
        burdens.append(burden)

    cluster_result = pd.DataFrame(cluster_rows)
    metrics = {
        "대표일": day_type,
        "제어전피크(kW)": float(unmanaged.max()),
        "제어후피크(kW)": float(controlled.max()),
        "피크감축률(%)": max(0.0, (unmanaged.max() - controlled.max()) / max(unmanaged.max(), 1e-9) * 100),
        "제어전_용량초과시간": int(np.sum(unmanaged > transformer_capacity_kw + 1e-9)),
        "제어후_용량초과시간": int(np.sum(controlled > transformer_capacity_kw + 1e-9)),
        "제어전_초과전력량(kWh)": float(np.maximum(unmanaged - transformer_capacity_kw, 0).sum()),
        "제어후_초과전력량(kWh)": float(np.maximum(controlled - transformer_capacity_kw, 0).sum()),
        "이동전력량(kWh)": total_shifted,
        "행동부하감축(kWh)": total_behavior_reduced,
        "냉난방감축(kWh)": total_hvac_reduced,
        "총감축전력량(kWh)": total_behavior_reduced + total_hvac_reduced,
        "평균편의점수": weighted_comfort / max(total_houses, 1),
        "군집간_가구당부담격차": float(max(burdens) - min(burdens)) if burdens else 0.0,
        "해상태": "최적해" if status == cp_model.OPTIMAL else "제한시간 내 실행가능해",
    }
    return hourly, cluster_result, metrics


def build_cluster_summary(households: pd.DataFrame) -> pd.DataFrame:
    return (
        households.groupby("군집", as_index=False)
        .agg(
            가구수=("고객ID", "count"),
            평균월사용량_kWh=("월사용량(kWh)", "mean"),
            평균주중일사용량_kWh=("주중일사용량(kWh)", "mean"),
            평균주말일사용량_kWh=("주말일사용량(kWh)", "mean"),
            평균주중최대부하_kW=("주중최대부하(kW)", "mean"),
            평균주말최대부하_kW=("주말최대부하(kW)", "mean"),
            평균주간비중=("주간비중(09-17)", "mean"),
            평균저녁비중=("저녁비중(18-24)", "mean"),
            평균이동가능비중=("이동가능비중", "mean"),
            평균냉난방제어가능비중=("냉난방제어가능비중", "mean"),
            평균제어수용도=("제어수용도", "mean"),
        )
    )


def calibration_summary(
    households: pd.DataFrame,
    calibration: pd.DataFrame,
    season: str,
    source_multiplier: float,
) -> pd.DataFrame:
    source_weekday = float(source_components(season, "주중")["total"].sum() * source_multiplier)
    source_weekend = float(source_components(season, "주말")["total"].sum() * source_multiplier)
    source_monthly = source_weekday * WEEKDAYS_PER_MONTH + source_weekend * WEEKENDS_PER_MONTH
    generated_weekday = float(households["주중일사용량(kWh)"].mean())
    generated_weekend = float(households["주말일사용량(kWh)"].mean())
    generated_monthly = float(households["월사용량(kWh)"].mean())
    return pd.DataFrame([
        {"구분": "주중 1일", "원본Excel(kWh/가구)": source_weekday, "생성100가구평균(kWh/가구)": generated_weekday},
        {"구분": "주말 1일", "원본Excel(kWh/가구)": source_weekend, "생성100가구평균(kWh/가구)": generated_weekend},
        {"구분": "월간(주중22+주말8)", "원본Excel(kWh/가구)": source_monthly, "생성100가구평균(kWh/가구)": generated_monthly},
    ]).assign(
        **{
            "차이(kWh)": lambda frame: frame["생성100가구평균(kWh/가구)"] - frame["원본Excel(kWh/가구)"],
            "차이율(%)": lambda frame: np.where(
                frame["원본Excel(kWh/가구)"].abs() > 1e-12,
                frame["차이(kWh)"] / frame["원본Excel(kWh/가구)"] * 100,
                0,
            ),
        }
    )


def make_csv_zip(tables: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, table in tables.items():
            archive.writestr(f"{name}.csv", table.to_csv(index=False).encode("utf-8-sig"))
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="100가구 군집형 수요관리 v2", page_icon="🏘️", layout="wide")
    st.title("🏘️ 100가구 군집형 계층제어 시뮬레이터")
    st.caption(f"앱 버전 {APP_VERSION} · 원본 4인가구 Excel 시간대별·가전별 부하로 재보정")
    st.success(
        "보정 핵심: 100가구의 시간대별 평균 부하가 최초 제공된 4인가구 Excel 곡선과 정확히 일치하도록 생성함. "
        "월사용량은 주중 22일과 주말 8일을 결합해 계산하며, 제어 가능량도 Excel의 가전별 '수요 이전 가능 여부'를 기초로 산정함. 제어 후 가구별 사용량에 대해 기본형·프리미엄형 요금을 모두 계산해 비용상 유리한 요금제를 추천함."
    )

    with st.sidebar:
        st.header("시뮬레이션 설정")
        household_count = st.slider("가구 수", min_value=50, max_value=300, value=100, step=10)
        cluster_count = st.slider("군집 수", min_value=3, max_value=8, value=5, step=1)
        season = st.selectbox("계절", ["봄가을", "여름", "겨울"], index=1)
        control_mode = st.selectbox("제어 모드", ["편의 우선", "균형", "계통 안정 우선"], index=1)
        source_percent = st.slider(
            "원본 Excel 평균사용량 반영률",
            min_value=60,
            max_value=120,
            value=100,
            step=5,
            format="%d%%",
            help="100%는 최초 제공한 4인가구 사용량을 그대로 100가구 평균으로 사용함. 실제 AMI 수준이 더 낮으면 비율을 낮춰 민감도 분석 가능함.",
        )
        capacity_ratio = st.slider("변압기 용량 / 제어 전 최대피크", 70, 110, 90, 1, format="%d%%")
        random_seed = st.number_input("가상가구 생성번호", min_value=1, max_value=9999, value=42, step=1)

        st.divider()
        st.subheader("요금제 비교 설정")
        basic_fee = st.number_input("기본형 월 구독료(원)", min_value=0, value=84_900, step=1_000)
        basic_quota = st.number_input("기본형 제공량(kWh)", min_value=1, value=450, step=10)
        premium_fee = st.number_input("프리미엄형 월 구독료(원)", min_value=0, value=249_900, step=1_000)
        premium_quota = st.number_input("프리미엄형 제공량(kWh)", min_value=1, value=1_000, step=10)
        current_marginal_rate = st.number_input(
            "현행 한계단가 비교값(원/kWh)", min_value=0.0, value=307.3, step=0.1,
            help="첨부 주택용 저압 요금표의 최고 누진구간 전력량요금 307.3원/kWh를 초기값으로 사용함.",
        )
        recommendation_rate_label = st.selectbox(
            "요금제 추천에 적용할 초과단가",
            ["200원/kWh", "300원/kWh", "400원/kWh", "현행 한계단가"],
            index=1,
        )
        run = st.button("보정·제어·요금제 비교 실행", type="primary", use_container_width=True)

    if not run and "calibrated_results" not in st.session_state:
        st.subheader("기존 모형과 달라진 점")
        st.markdown(
            """
            1. **원본 곡선 고정**: 가상가구를 만든 후 100가구 평균을 원본 Excel의 24시간 곡선에 시간대별로 재보정함  
            2. **월사용량 산식 수정**: 대표일×30이 아니라 `주중 22일 + 주말 8일`로 계산함  
            3. **상시부하 시간 이동 제거**: 냉장고 등 고정부하는 시각을 옮기지 않고, 이동 가능 가전만 고객별 생활시간 차이를 반영함  
            4. **제어 가능량 근거 변경**: 총부하에 임의 비율을 곱하지 않고, 원본 Excel의 가전별 부하를 이동형·행동제어형·냉난방형으로 분해함  
            5. **검증표 제공**: 원본 사용량과 생성된 100가구 평균의 차이를 매 실행마다 표시함  
            6. **요금제 비교**: 각 가구의 제어 후 사용량에 기본형·프리미엄형을 각각 적용하고 초과단가별 유리한 요금제를 추천함
            """
        )
        return

    if run:
        with st.spinner("원본 Excel 기준 100가구 생성 → 시간대별 재보정 → 군집화 → 주중·주말 최적화 중입니다..."):
            source_multiplier = source_percent / 100.0
            households, population, calibration_detail = generate_population(
                int(household_count), season, int(random_seed), source_multiplier
            )
            households, cluster_days = cluster_population(
                households, population, int(cluster_count), int(random_seed)
            )

            unmanaged_weekday = population["주중"]["total"].sum(axis=0)
            unmanaged_weekend = population["주말"]["total"].sum(axis=0)
            baseline_peak = float(max(unmanaged_weekday.max(), unmanaged_weekend.max()))
            transformer_capacity = baseline_peak * capacity_ratio / 100.0

            hourly_weekday, cluster_weekday, metrics_weekday = optimize_day(
                cluster_days["주중"], season, "주중", control_mode, transformer_capacity
            )
            hourly_weekend, cluster_weekend, metrics_weekend = optimize_day(
                cluster_days["주말"], season, "주말", control_mode, transformer_capacity
            )

            calibration_table = calibration_summary(households, calibration_detail, season, source_multiplier)
            hourly_all = pd.concat([hourly_weekday, hourly_weekend], ignore_index=True)
            cluster_results = pd.concat([cluster_weekday, cluster_weekend], ignore_index=True)
            day_metrics = pd.DataFrame([metrics_weekday, metrics_weekend])

            households = allocate_monthly_control_to_households(households, population, cluster_results)
            overage_rates = {
                "200원/kWh": 200.0,
                "300원/kWh": 300.0,
                "400원/kWh": 400.0,
                "현행 한계단가": float(current_marginal_rate),
            }
            plan_detail, plan_summary, plan_break_even = build_plan_comparison_tables(
                households,
                float(basic_fee), float(basic_quota),
                float(premium_fee), float(premium_quota),
                overage_rates,
            )
            selected_plan = plan_detail[plan_detail["초과단가시나리오"] == recommendation_rate_label].copy()
            selected_columns = [
                "고객ID", "기본형요금_제어후(원)", "프리미엄형요금_제어후(원)",
                "비용상추천요금제", "추천요금(원)", "추천에따른절감액(원)", "사용량구간상요금제",
            ]
            households = households.merge(selected_plan[selected_columns], on="고객ID", how="left")
            cluster_summary = build_cluster_summary(households)
            plan_cluster_summary = (
                households.pivot_table(
                    index="군집", columns="비용상추천요금제", values="고객ID", aggfunc="count", fill_value=0
                )
                .reset_index()
            )
            for column in ["기본형", "프리미엄형", "동일"]:
                if column not in plan_cluster_summary.columns:
                    plan_cluster_summary[column] = 0
            plan_cluster_summary = plan_cluster_summary[["군집", "기본형", "프리미엄형", "동일"]]
            plan_cluster_summary.columns = ["군집", "기본형추천가구수", "프리미엄형추천가구수", "동일요금가구수"]

            monthly_baseline = float(
                hourly_weekday["제어전(kW)"].sum() * WEEKDAYS_PER_MONTH
                + hourly_weekend["제어전(kW)"].sum() * WEEKENDS_PER_MONTH
            )
            monthly_controlled = float(
                hourly_weekday["제어후(kW)"].sum() * WEEKDAYS_PER_MONTH
                + hourly_weekend["제어후(kW)"].sum() * WEEKENDS_PER_MONTH
            )
            monthly_shifted = float(
                metrics_weekday["이동전력량(kWh)"] * WEEKDAYS_PER_MONTH
                + metrics_weekend["이동전력량(kWh)"] * WEEKENDS_PER_MONTH
            )
            monthly_reduced = monthly_baseline - monthly_controlled
            monthly_overload_before = int(
                metrics_weekday["제어전_용량초과시간"] * WEEKDAYS_PER_MONTH
                + metrics_weekend["제어전_용량초과시간"] * WEEKENDS_PER_MONTH
            )
            monthly_overload_after = int(
                metrics_weekday["제어후_용량초과시간"] * WEEKDAYS_PER_MONTH
                + metrics_weekend["제어후_용량초과시간"] * WEEKENDS_PER_MONTH
            )
            average_comfort = (
                metrics_weekday["평균편의점수"] * WEEKDAYS_PER_MONTH
                + metrics_weekend["평균편의점수"] * WEEKENDS_PER_MONTH
            ) / (WEEKDAYS_PER_MONTH + WEEKENDS_PER_MONTH)

            overall_metrics = pd.DataFrame([{
                "가구수": household_count,
                "군집수": cluster_count,
                "계절": season,
                "제어모드": control_mode,
                "원본반영률(%)": source_percent,
                "변압기용량(kW)": transformer_capacity,
                "제어전최대피크(kW)": max(metrics_weekday["제어전피크(kW)"], metrics_weekend["제어전피크(kW)"]),
                "제어후최대피크(kW)": max(metrics_weekday["제어후피크(kW)"], metrics_weekend["제어후피크(kW)"]),
                "월간총사용량_제어전(kWh)": monthly_baseline,
                "월간총사용량_제어후(kWh)": monthly_controlled,
                "월간이동전력량(kWh)": monthly_shifted,
                "월간감축전력량(kWh)": monthly_reduced,
                "월간용량초과시간_제어전": monthly_overload_before,
                "월간용량초과시간_제어후": monthly_overload_after,
                "평균고객편의점수": average_comfort,
                "가구당평균월사용량(kWh)": households["월사용량(kWh)"].mean(),
            }])

            st.session_state["calibrated_results"] = {
                "households": households,
                "population": population,
                "calibration_detail": calibration_detail,
                "calibration_table": calibration_table,
                "cluster_summary": cluster_summary,
                "cluster_results": cluster_results,
                "hourly": hourly_all,
                "day_metrics": day_metrics,
                "overall_metrics": overall_metrics,
                "plan_detail": plan_detail,
                "plan_summary": plan_summary,
                "plan_break_even": plan_break_even,
                "plan_cluster_summary": plan_cluster_summary,
                "recommendation_rate_label": recommendation_rate_label,
                "settings": pd.DataFrame([{
                    "가구수": household_count,
                    "군집수": cluster_count,
                    "계절": season,
                    "제어모드": control_mode,
                    "원본Excel평균사용량반영률(%)": source_percent,
                    "변압기용량비율(%)": capacity_ratio,
                    "변압기용량(kW)": transformer_capacity,
                    "가상가구생성번호": random_seed,
                    "월주중일수": WEEKDAYS_PER_MONTH,
                    "월주말일수": WEEKENDS_PER_MONTH,
                    "기본형월구독료(원)": basic_fee,
                    "기본형제공량(kWh)": basic_quota,
                    "프리미엄형월구독료(원)": premium_fee,
                    "프리미엄형제공량(kWh)": premium_quota,
                    "현행한계단가비교값(원/kWh)": current_marginal_rate,
                    "요금제추천적용초과단가": recommendation_rate_label,
                }]),
            }

    result = st.session_state.get("calibrated_results")
    if not result:
        return

    calibration_table = result["calibration_table"]
    households = result["households"]
    hourly = result["hourly"]
    cluster_summary = result["cluster_summary"]
    cluster_results = result["cluster_results"]
    plan_summary = result["plan_summary"]
    plan_break_even = result["plan_break_even"]
    plan_cluster_summary = result["plan_cluster_summary"]
    recommendation_rate_label = result["recommendation_rate_label"]
    overall = result["overall_metrics"].iloc[0]

    st.subheader("1. 원본 Excel 보정 검증")
    display_calibration = calibration_table.copy()
    for column in display_calibration.select_dtypes(include=["float"]).columns:
        display_calibration[column] = display_calibration[column].round(6)
    st.dataframe(display_calibration, use_container_width=True, hide_index=True)
    maximum_hourly_error = float(result["calibration_detail"]["차이율(%)"].abs().max())
    st.caption(f"24시간대별 최대 보정오차: {maximum_hourly_error:.10f}% · 부동소수점 반올림 외에는 원본 곡선과 일치함")

    st.subheader("2. 핵심 결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("가구당 평균 월사용량", f"{overall['가구당평균월사용량(kWh)']:.1f} kWh")
    c2.metric("제어 전 최대피크", f"{overall['제어전최대피크(kW)']:.1f} kW")
    peak_reduction = max(0.0, (overall["제어전최대피크(kW)"] - overall["제어후최대피크(kW)"]) / max(overall["제어전최대피크(kW)"], 1e-9) * 100)
    c3.metric("제어 후 최대피크", f"{overall['제어후최대피크(kW)']:.1f} kW", f"-{peak_reduction:.1f}%")
    c4.metric("평균 고객편의", f"{overall['평균고객편의점수']:.1f}점")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("월간 이동전력량", f"{overall['월간이동전력량(kWh)']:.1f} kWh")
    c6.metric("월간 감축전력량", f"{overall['월간감축전력량(kWh)']:.1f} kWh")
    c7.metric("월 용량초과시간", f"{int(overall['월간용량초과시간_제어후'])}시간", f"제어 전 {int(overall['월간용량초과시간_제어전'])}시간")
    c8.metric("변압기 용량", f"{overall['변압기용량(kW)']:.1f} kW")

    st.subheader("3. 기본형·프리미엄형 요금 비교 및 추천")
    selected_summary = plan_summary[plan_summary["초과단가시나리오"] == recommendation_rate_label].iloc[0]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("추천 적용 초과단가", f"{selected_summary['초과단가(원/kWh)']:.1f}원/kWh")
    p2.metric("기본형 추천", f"{int(selected_summary['기본형추천가구수'])}가구")
    p3.metric("프리미엄형 추천", f"{int(selected_summary['프리미엄형추천가구수'])}가구")
    p4.metric("가구당 평균 추천요금", f"{selected_summary['가구당평균_추천요금(원)']:,.0f}원")

    summary_display = plan_summary.copy()
    money_columns = [column for column in summary_display.columns if "요금" in column or "총액" in column or "절감" in column]
    for column in money_columns:
        if column in summary_display.columns:
            summary_display[column] = summary_display[column].round(0).astype(int)
    st.dataframe(summary_display, use_container_width=True, hide_index=True)

    average_figure = go.Figure()
    average_figure.add_trace(go.Bar(
        x=plan_summary["초과단가시나리오"],
        y=plan_summary["가구당평균_기본형요금(원)"],
        name="기본형",
    ))
    average_figure.add_trace(go.Bar(
        x=plan_summary["초과단가시나리오"],
        y=plan_summary["가구당평균_프리미엄형요금(원)"],
        name="프리미엄형",
    ))
    average_figure.update_layout(
        barmode="group", xaxis_title="초과단가 시나리오", yaxis_title="가구당 평균 월요금(원)", height=390
    )
    st.plotly_chart(average_figure, use_container_width=True)

    st.markdown("**요금제 손익분기 사용량**")
    break_even_display = plan_break_even.copy()
    for column in ["계산상교차사용량(kWh)", "프리미엄유리시작사용량(kWh)", "1000kWh초과시_기본형-프리미엄형(원)"]:
        break_even_display[column] = break_even_display[column].round(1)
    st.dataframe(break_even_display, use_container_width=True, hide_index=True)

    selected_premium_count = int(selected_summary["프리미엄형추천가구수"])
    if selected_premium_count == 0:
        st.warning(
            "선택한 초과단가에서는 100가구 모두 기본형이 비용상 유리합니다. 이는 고객 사용량이 낮아서라기보다, "
            "프리미엄형 구독료 249,900원이 기본형 대비 높아 요금 역전구간이 형성되지 않거나 매우 높기 때문일 수 있습니다."
        )
    else:
        st.info(
            f"선택한 초과단가 기준으로 프리미엄형이 더 저렴한 고객은 {selected_premium_count}가구입니다. "
            "추천은 제어 후 월사용량과 초과요금을 기준으로 하며 자동전환은 수행하지 않습니다."
        )

    st.markdown("**군집별 비용상 추천 요금제**")
    st.dataframe(plan_cluster_summary, use_container_width=True, hide_index=True)

    st.subheader("4. 원본 곡선과 100가구 평균 비교")
    comparison = result["calibration_detail"].copy()
    for day_type in ["주중", "주말"]:
        day = comparison[comparison["대표일"] == day_type]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=day["시간"], y=day["원본Excel_가구당(kW)"], mode="lines+markers", name="원본 Excel"))
        figure.add_trace(go.Scatter(x=day["시간"], y=day["생성100가구_가구당평균(kW)"], mode="lines", name="100가구 평균", line=dict(dash="dash")))
        figure.update_layout(title=f"{day_type} 가구당 평균 부하", xaxis_title="시간", yaxis_title="kW", hovermode="x unified", height=340)
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("5. 변압기 총부하 제어")
    weekday_tab, weekend_tab = st.tabs(["주중", "주말"])
    for tab, day_type in [(weekday_tab, "주중"), (weekend_tab, "주말")]:
        with tab:
            day = hourly[hourly["대표일"] == day_type]
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=day["시간"], y=day["제어전(kW)"], mode="lines+markers", name="제어 전"))
            figure.add_trace(go.Scatter(x=day["시간"], y=day["제어후(kW)"], mode="lines+markers", name="제어 후"))
            figure.add_trace(go.Scatter(x=day["시간"], y=day["변압기용량(kW)"], mode="lines", name="변압기 용량", line=dict(dash="dash")))
            figure.update_layout(xaxis_title="시간", yaxis_title="전력(kW)", hovermode="x unified", height=430)
            st.plotly_chart(figure, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("6. 군집 구성")
        figure = go.Figure(go.Bar(
            x=cluster_summary["군집"].astype(str),
            y=cluster_summary["가구수"],
            text=cluster_summary["가구수"],
            textposition="auto",
        ))
        figure.update_layout(xaxis_title="군집", yaxis_title="가구 수", height=360)
        st.plotly_chart(figure, use_container_width=True)
    with right:
        st.subheader("7. 군집별 평균 월사용량")
        figure = go.Figure(go.Bar(
            x=cluster_summary["군집"].astype(str),
            y=cluster_summary["평균월사용량_kWh"],
            text=cluster_summary["평균월사용량_kWh"].round(1),
            textposition="auto",
        ))
        figure.update_layout(xaxis_title="군집", yaxis_title="가구당 월사용량(kWh)", height=360)
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("8. 군집별 제어 결과")
    display_cluster = cluster_results.copy()
    for column in display_cluster.select_dtypes(include=["float"]).columns:
        display_cluster[column] = display_cluster[column].round(3)
    st.dataframe(display_cluster, use_container_width=True, hide_index=True)

    with st.expander("100가구 생성자료와 군집 배정 확인"):
        display_households = households.copy()
        for column in display_households.select_dtypes(include=["float"]).columns:
            display_households[column] = display_households[column].round(4)
        st.dataframe(display_households, use_container_width=True, hide_index=True, height=500)

    st.subheader("9. 결과 내려받기")
    download = make_csv_zip({
        "시뮬레이션설정": result["settings"],
        "원본보정검증_요약": calibration_table,
        "원본보정검증_시간대별": result["calibration_detail"],
        "100가구_생성자료": households,
        "군집요약": cluster_summary,
        "군집별_제어결과": cluster_results,
        "시간대별_총부하": hourly,
        "대표일별_핵심지표": result["day_metrics"],
        "월간_핵심지표": result["overall_metrics"],
        "요금제비교_가구별전체시나리오": result["plan_detail"],
        "요금제비교_시나리오요약": result["plan_summary"],
        "요금제비교_손익분기": result["plan_break_even"],
        "요금제비교_군집별추천": result["plan_cluster_summary"],
    })
    st.download_button(
        "결과자료 ZIP(CSV) 다운로드",
        data=download,
        file_name="100가구_원본보정_계통제어_요금제비교_결과.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.warning(
        "해석상 유의: 최초 제공된 4인가구 곡선은 고사양·다가전 보유를 가정해 실제 평균가구보다 사용량이 높을 수 있음. "
        "따라서 기본값은 원본 반영률 100%로 두되, 실제 AMI 자료 확보 후 '원본 Excel 평균사용량 반영률'과 가구유형 분포를 보정해야 함. "
        "현재 모형은 전압·무효전력·통신지연을 반영하지 않은 개념검증용임."
    )


if __name__ == "__main__":
    main()
