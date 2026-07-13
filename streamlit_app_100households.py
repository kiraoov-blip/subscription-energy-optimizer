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

APP_VERSION = "2026-07-13-multi-v1.0"
HOURS = np.arange(24)

# 사용자가 제공한 4인 가구 계절·요일별 부하곡선의 시간별 합계(kWh/h)를 기초 프로파일로 사용
BASE_PROFILES: Dict[Tuple[str, str], List[float]] = {
    ("봄가을", "주중"): [0.158,0.158,0.158,0.158,0.158,0.158,0.903,0.953,0.233,0.233,0.283,0.233,0.233,0.233,0.233,0.233,0.423,0.483,1.943,1.843,1.083,1.633,1.293,0.513],
    ("봄가을", "주말"): [0.158,0.158,0.158,0.158,0.158,0.158,0.203,0.203,0.703,1.733,0.973,1.373,1.983,1.703,0.573,0.923,1.373,1.473,2.933,2.213,1.703,1.823,0.923,0.323],
    ("여름", "주중"): [0.878,0.578,0.578,0.578,0.578,0.578,1.043,0.943,0.223,0.223,0.393,0.343,0.343,0.223,0.223,0.223,0.413,0.473,2.608,2.808,2.048,2.598,2.608,1.578],
    ("여름", "주말"): [0.878,0.578,0.578,0.578,0.578,0.578,0.603,0.193,0.693,1.723,1.023,2.073,3.033,2.753,1.623,1.773,2.223,2.323,4.018,3.178,2.668,2.788,2.238,1.388],
    ("겨울", "주중"): [0.668,0.378,0.378,0.378,0.378,0.428,2.078,2.198,0.578,0.228,0.278,0.228,0.228,0.228,0.228,0.228,0.418,0.948,2.868,2.728,2.218,2.828,2.768,1.308],
    ("겨울", "주말"): [0.668,0.378,0.378,0.378,0.378,0.378,0.418,1.088,1.608,2.808,1.468,1.868,2.478,2.198,1.068,1.418,1.968,2.458,3.858,3.078,2.768,2.948,2.348,1.108],
}

# 실제 통계가 아니라 100가구 가상표본을 생성하기 위한 연구용 가정
ARCHETYPES = [
    {"name": "절약형", "prob": 0.18, "scale": (0.55, 0.78), "day_bias": (-0.08, 0.05), "eve_bias": (-0.08, 0.08), "flex": (0.18, 0.28), "reduce": (0.02, 0.05), "accept": (0.75, 0.95)},
    {"name": "표준형", "prob": 0.32, "scale": (0.78, 1.02), "day_bias": (-0.05, 0.08), "eve_bias": (-0.03, 0.12), "flex": (0.15, 0.25), "reduce": (0.025, 0.06), "accept": (0.60, 0.90)},
    {"name": "재택형", "prob": 0.18, "scale": (0.85, 1.12), "day_bias": (0.12, 0.35), "eve_bias": (-0.03, 0.10), "flex": (0.12, 0.22), "reduce": (0.02, 0.055), "accept": (0.50, 0.82)},
    {"name": "저녁집중형", "prob": 0.20, "scale": (0.88, 1.20), "day_bias": (-0.12, 0.02), "eve_bias": (0.16, 0.38), "flex": (0.16, 0.27), "reduce": (0.03, 0.07), "accept": (0.50, 0.84)},
    {"name": "고사용량형", "prob": 0.12, "scale": (1.18, 1.58), "day_bias": (0.00, 0.20), "eve_bias": (0.08, 0.28), "flex": (0.18, 0.30), "reduce": (0.04, 0.09), "accept": (0.42, 0.78)},
]

MODE_CONFIG = {
    "편의 우선": {"shift_multiplier": 1.55, "reduction_fraction": 0.25, "shift_penalty": 18, "reduction_penalty": 90, "fairness_penalty": 32},
    "균형": {"shift_multiplier": 2.00, "reduction_fraction": 0.50, "shift_penalty": 11, "reduction_penalty": 52, "fairness_penalty": 24},
    "계통 안정 우선": {"shift_multiplier": 2.60, "reduction_fraction": 0.85, "shift_penalty": 7, "reduction_penalty": 30, "fairness_penalty": 18},
}


@dataclass
class ClusterData:
    cluster_id: int
    count: int
    baseline: np.ndarray
    shiftable: np.ndarray
    reducible: np.ndarray
    acceptance: float
    daily_kwh_per_house: float


def grid_weights(season: str, day_type: str) -> np.ndarray:
    w = np.full(24, 2, dtype=int)
    w[(HOURS >= 23) | (HOURS < 8)] = 1
    if season == "여름":
        w[(HOURS >= 14) & (HOURS < 21)] = 4
    elif season == "겨울":
        w[((HOURS >= 8) & (HOURS < 11)) | ((HOURS >= 17) & (HOURS < 21))] = 4
    else:
        w[(HOURS >= 17) & (HOURS < 21)] = 4
    if day_type == "주말":
        w[w == 4] = 3
    return w


def generate_households(n: int, season: str, day_type: str, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = np.array(BASE_PROFILES[(season, day_type)], dtype=float)
    probs = np.array([a["prob"] for a in ARCHETYPES], dtype=float)
    probs /= probs.sum()
    archetype_idx = rng.choice(len(ARCHETYPES), size=n, p=probs)

    profiles = np.zeros((n, 24), dtype=float)
    rows: List[Dict[str, object]] = []

    for i, idx in enumerate(archetype_idx):
        a = ARCHETYPES[int(idx)]
        scale = rng.uniform(*a["scale"])
        day_bias = rng.uniform(*a["day_bias"])
        eve_bias = rng.uniform(*a["eve_bias"])
        flex_share = rng.uniform(*a["flex"])
        reducible_share = rng.uniform(*a["reduce"])
        acceptance = rng.uniform(*a["accept"])
        time_shift = int(rng.choice([-2, -1, 0, 0, 0, 1, 2]))

        p = np.roll(base, time_shift) * scale
        p[9:17] *= (1.0 + day_bias)
        p[18:24] *= (1.0 + eve_bias)
        p *= np.clip(rng.normal(1.0, 0.075, 24), 0.78, 1.25)
        p = np.maximum(p, 0.015)
        profiles[i] = p

        daily = float(p.sum())
        monthly = daily * (30 if day_type == "주말" else 30)  # 대표일 단순 환산
        peak_hour = int(np.argmax(p))
        plan = "기본형" if monthly <= 450 else "프리미엄형"
        rows.append({
            "고객ID": f"H{i+1:03d}",
            "생성유형": a["name"],
            "일사용량(kWh)": daily,
            "월환산사용량(kWh)": monthly,
            "최대부하(kW)": float(p.max()),
            "최대부하시각": peak_hour,
            "주간비중(09-17)": float(p[9:17].sum() / daily),
            "저녁비중(18-24)": float(p[18:24].sum() / daily),
            "이동가능비중": flex_share,
            "감축가능비중": reducible_share,
            "제어수용도": acceptance,
            "권장요금제": plan,
        })

    return pd.DataFrame(rows), profiles


def simple_kmeans(features: np.ndarray, k: int, seed: int, max_iter: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed + 1009)
    x = np.asarray(features, dtype=float)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    z = (x - mu) / sigma

    # k-means++에 가까운 초기화
    centers = [z[rng.integers(0, len(z))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((z - c) ** 2, axis=1) for c in centers], axis=0)
        if float(d2.sum()) <= 1e-12:
            centers.append(z[rng.integers(0, len(z))])
        else:
            centers.append(z[rng.choice(len(z), p=d2 / d2.sum())])
    centers = np.array(centers)

    labels = np.zeros(len(z), dtype=int)
    for _ in range(max_iter):
        dist = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = z[labels == c]
            if len(members) == 0:
                farthest = int(np.argmax(np.min(dist, axis=1)))
                centers[c] = z[farthest]
                labels[farthest] = c
            else:
                centers[c] = members.mean(axis=0)

    # 일사용량이 낮은 군집부터 번호 재부여
    cluster_daily = []
    for c in range(k):
        members = x[labels == c]
        cluster_daily.append(float(members[:, 0].mean()) if len(members) else float("inf"))
    order = np.argsort(cluster_daily)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    return np.array([remap[int(v)] for v in labels], dtype=int)


def cluster_households(households: pd.DataFrame, profiles: np.ndarray, k: int, seed: int) -> Tuple[pd.DataFrame, List[ClusterData]]:
    features = households[[
        "일사용량(kWh)", "최대부하(kW)", "주간비중(09-17)", "저녁비중(18-24)", "이동가능비중", "제어수용도"
    ]].to_numpy(dtype=float)
    labels = simple_kmeans(features, k, seed)
    households = households.copy()
    households["군집"] = labels + 1

    clusters: List[ClusterData] = []
    for c in range(k):
        mask = labels == c
        idx = np.where(mask)[0]
        h = households.loc[mask]
        p = profiles[idx]
        flex = h["이동가능비중"].to_numpy(dtype=float)[:, None]
        red = h["감축가능비중"].to_numpy(dtype=float)[:, None]
        clusters.append(ClusterData(
            cluster_id=c + 1,
            count=len(idx),
            baseline=p.sum(axis=0),
            shiftable=(p * flex).sum(axis=0),
            reducible=(p * red).sum(axis=0),
            acceptance=float(h["제어수용도"].mean()),
            daily_kwh_per_house=float(h["일사용량(kWh)"].mean()),
        ))
    return households, clusters


def optimize_clusters(
    clusters: List[ClusterData],
    season: str,
    day_type: str,
    control_mode: str,
    capacity_kw: float,
    solve_seconds: float = 8.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    cfg = MODE_CONFIG[control_mode]
    unit = 1000  # kWh/h -> Wh/h 정수화
    weights = grid_weights(season, day_type)
    model = cp_model.CpModel()

    cluster_load_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}
    cluster_shift_new: Dict[Tuple[int, int], cp_model.IntVar] = {}
    cluster_reduction: Dict[Tuple[int, int], cp_model.IntVar] = {}
    burden_vars: List[cp_model.IntVar] = []
    cluster_meta: Dict[int, Dict[str, object]] = {}
    objective_terms = []

    for ci, c in enumerate(clusters):
        baseline_wh = np.rint(c.baseline * unit).astype(int)
        shift_wh = np.rint(c.shiftable * unit).astype(int)
        reducible_wh = np.rint(c.reducible * unit).astype(int)
        fixed_wh = np.maximum(baseline_wh - shift_wh - reducible_wh, 0)
        total_shift = int(shift_wh.sum())
        avg_shift = math.ceil(total_shift / 24) if total_shift else 0
        abs_vars: List[cp_model.IntVar] = []
        reduction_vars: List[cp_model.IntVar] = []

        for t in range(24):
            cap = max(
                int(math.ceil(shift_wh[t] * cfg["shift_multiplier"])),
                int(avg_shift * 2),
                1,
            )
            shift_var = model.NewIntVar(0, cap, f"shift_c{ci}_t{t}")
            max_red = int(math.floor(reducible_wh[t] * cfg["reduction_fraction"]))
            red_var = model.NewIntVar(0, max(max_red, 0), f"reduce_c{ci}_t{t}")
            max_load = int(fixed_wh[t] + reducible_wh[t] + cap)
            load_var = model.NewIntVar(0, max(max_load, 1), f"load_c{ci}_t{t}")
            model.Add(load_var == int(fixed_wh[t] + reducible_wh[t]) + shift_var - red_var)

            diff = model.NewIntVar(-max(cap, int(shift_wh[t])), max(cap, int(shift_wh[t])), f"shift_diff_c{ci}_t{t}")
            model.Add(diff == shift_var - int(shift_wh[t]))
            abs_diff = model.NewIntVar(0, max(cap, int(shift_wh[t])), f"shift_abs_c{ci}_t{t}")
            model.AddAbsEquality(abs_diff, diff)

            cluster_load_vars[(ci, t)] = load_var
            cluster_shift_new[(ci, t)] = shift_var
            cluster_reduction[(ci, t)] = red_var
            abs_vars.append(abs_diff)
            reduction_vars.append(red_var)

            discomfort = max(1, int(round(1.3 / max(c.acceptance, 0.25))))
            objective_terms.append(abs_diff * int(cfg["shift_penalty"] * discomfort))
            objective_terms.append(red_var * int(cfg["reduction_penalty"] * discomfort))

        # 이동형 에너지는 하루 총량 보존
        model.Add(sum(cluster_shift_new[(ci, t)] for t in range(24)) == total_shift)

        abs_sum = model.NewIntVar(0, max(total_shift * 4, 1), f"abs_sum_c{ci}")
        model.Add(abs_sum == sum(abs_vars))
        shifted = model.NewIntVar(0, max(total_shift * 2, 1), f"shifted_c{ci}")
        model.AddDivisionEquality(shifted, abs_sum, 2)
        red_sum_max = int(sum(int(math.floor(v * cfg["reduction_fraction"])) for v in reducible_wh))
        red_sum = model.NewIntVar(0, max(red_sum_max, 1), f"red_sum_c{ci}")
        model.Add(red_sum == sum(reduction_vars))
        intervention = model.NewIntVar(0, max(total_shift * 2 + red_sum_max, 1), f"intervention_c{ci}")
        model.Add(intervention == shifted + red_sum)
        burden = model.NewIntVar(0, max(total_shift * 2 + red_sum_max, 1), f"burden_per_hh_c{ci}")
        model.AddDivisionEquality(burden, intervention, max(c.count, 1))
        burden_vars.append(burden)
        cluster_meta[ci] = {
            "baseline_wh": baseline_wh,
            "shift_wh": shift_wh,
            "reducible_wh": reducible_wh,
            "shifted_var": shifted,
            "red_sum_var": red_sum,
            "burden_var": burden,
        }

    unmanaged = np.sum([c.baseline for c in clusters], axis=0)
    max_total_wh = int(math.ceil(max(unmanaged.max() * unit * 3, 1)))
    capacity_wh = int(round(capacity_kw * unit))
    total_load_vars: List[cp_model.IntVar] = []
    overload_vars: List[cp_model.IntVar] = []
    peak_var = model.NewIntVar(0, max_total_wh, "aggregate_peak")

    for t in range(24):
        total = model.NewIntVar(0, max_total_wh, f"total_t{t}")
        model.Add(total == sum(cluster_load_vars[(ci, t)] for ci in range(len(clusters))))
        overload = model.NewIntVar(0, max_total_wh, f"overload_t{t}")
        model.Add(overload >= total - capacity_wh)
        model.Add(peak_var >= total)
        total_load_vars.append(total)
        overload_vars.append(overload)
        objective_terms.append(overload * 5000)
        objective_terms.append(total * int(weights[t]))

    objective_terms.append(peak_var * 45)

    if len(burden_vars) >= 2:
        max_burden = model.NewIntVar(0, max_total_wh, "max_burden")
        min_burden = model.NewIntVar(0, max_total_wh, "min_burden")
        gap = model.NewIntVar(0, max_total_wh, "fairness_gap")
        model.AddMaxEquality(max_burden, burden_vars)
        model.AddMinEquality(min_burden, burden_vars)
        model.Add(gap == max_burden - min_burden)
        objective_terms.append(gap * int(cfg["fairness_penalty"]))
    else:
        gap = None

    model.Minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(solve_seconds)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 42
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("최적화 해를 찾지 못했습니다. 변압기 용량 또는 제어조건을 완화해 주세요.")

    controlled = np.array([solver.Value(v) / unit for v in total_load_vars])
    overload = np.maximum(controlled - capacity_kw, 0.0)
    hourly = pd.DataFrame({
        "시간": HOURS,
        "제어전(kW)": unmanaged,
        "제어후(kW)": controlled,
        "변압기용량(kW)": capacity_kw,
        "제어전_초과(kW)": np.maximum(unmanaged - capacity_kw, 0.0),
        "제어후_초과(kW)": overload,
        "계통가중치": weights,
    })

    cluster_rows: List[Dict[str, object]] = []
    total_shifted = 0.0
    total_reduced = 0.0
    weighted_comfort = 0.0
    total_houses = sum(c.count for c in clusters)
    burdens = []

    for ci, c in enumerate(clusters):
        meta = cluster_meta[ci]
        ctrl_profile = np.array([solver.Value(cluster_load_vars[(ci, t)]) / unit for t in range(24)])
        shifted_kwh = solver.Value(meta["shifted_var"]) / unit
        reduced_kwh = solver.Value(meta["red_sum_var"]) / unit
        burden_per_hh = solver.Value(meta["burden_var"]) / unit
        daily_per_hh = max(c.daily_kwh_per_house, 0.1)
        discomfort = (shifted_kwh / max(c.count, 1) / daily_per_hh) * 22 + (reduced_kwh / max(c.count, 1) / daily_per_hh) * 135
        comfort_score = max(0.0, min(100.0, 100.0 - discomfort))
        cluster_rows.append({
            "군집": c.cluster_id,
            "가구수": c.count,
            "가구당_평균일사용량(kWh)": c.daily_kwh_per_house,
            "평균제어수용도": c.acceptance,
            "제어전_군집피크(kW)": float(c.baseline.max()),
            "제어후_군집피크(kW)": float(ctrl_profile.max()),
            "이동전력량(kWh)": shifted_kwh,
            "감축전력량(kWh)": reduced_kwh,
            "가구당_제어부담(kWh)": burden_per_hh,
            "편의점수(100점)": comfort_score,
        })
        total_shifted += shifted_kwh
        total_reduced += reduced_kwh
        weighted_comfort += comfort_score * c.count
        burdens.append(burden_per_hh)

    cluster_result = pd.DataFrame(cluster_rows)
    unmanaged_peak = float(unmanaged.max())
    controlled_peak = float(controlled.max())
    fairness_gap_kwh = float(max(burdens) - min(burdens)) if burdens else 0.0
    metrics = {
        "제어전피크(kW)": unmanaged_peak,
        "제어후피크(kW)": controlled_peak,
        "피크감축률(%)": max(0.0, (unmanaged_peak - controlled_peak) / max(unmanaged_peak, 1e-9) * 100),
        "제어전_용량초과시간": int(np.sum(unmanaged > capacity_kw + 1e-9)),
        "제어후_용량초과시간": int(np.sum(controlled > capacity_kw + 1e-9)),
        "제어전_초과전력량(kWh)": float(np.maximum(unmanaged - capacity_kw, 0).sum()),
        "제어후_초과전력량(kWh)": float(overload.sum()),
        "이동전력량(kWh)": total_shifted,
        "감축전력량(kWh)": total_reduced,
        "평균편의점수": weighted_comfort / max(total_houses, 1),
        "군집간_가구당부담격차(kWh)": fairness_gap_kwh,
        "해상태": "최적해" if status == cp_model.OPTIMAL else "제한시간 내 실행가능해",
    }
    return hourly, cluster_result, metrics


def build_cluster_summary(households: pd.DataFrame) -> pd.DataFrame:
    return (
        households.groupby("군집", as_index=False)
        .agg(
            가구수=("고객ID", "count"),
            평균일사용량_kWh=("일사용량(kWh)", "mean"),
            평균월환산사용량_kWh=("월환산사용량(kWh)", "mean"),
            평균최대부하_kW=("최대부하(kW)", "mean"),
            평균주간비중=("주간비중(09-17)", "mean"),
            평균저녁비중=("저녁비중(18-24)", "mean"),
            평균이동가능비중=("이동가능비중", "mean"),
            평균제어수용도=("제어수용도", "mean"),
        )
    )


def make_csv_zip(tables: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False).encode("utf-8-sig"))
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="100가구 군집형 수요관리", page_icon="🏘️", layout="wide")
    st.title("🏘️ 100가구 군집형 계층제어 시뮬레이터")
    st.caption(f"앱 버전 {APP_VERSION} · Google OR-Tools 기반 기초 연구모형")
    st.info(
        "100개의 서로 다른 가상 부하곡선을 생성한 뒤 유사한 가구를 군집화하고, "
        "상위 제어기가 변압기 용량과 계통 시간대 가중치를 고려해 군집별 전력 이동·감축량을 배분합니다. "
        "현재 버전은 실제 계통운영용이 아닌 개념검증용 기초모형입니다."
    )

    with st.sidebar:
        st.header("시뮬레이션 설정")
        household_count = st.slider("가구 수", min_value=50, max_value=300, value=100, step=10)
        cluster_count = st.slider("군집 수", min_value=3, max_value=8, value=5, step=1)
        season = st.selectbox("계절", ["봄가을", "여름", "겨울"], index=1)
        day_type = st.radio("대표일", ["주중", "주말"], horizontal=True)
        control_mode = st.selectbox("제어 모드", ["편의 우선", "균형", "계통 안정 우선"], index=1)
        capacity_ratio = st.slider("변압기 용량 / 제어 전 피크", 70, 110, 90, 1, format="%d%%")
        random_seed = st.number_input("가상가구 생성번호", min_value=1, max_value=9999, value=42, step=1)
        run = st.button("100가구 시뮬레이션 실행", type="primary", use_container_width=True)

    if not run and "multi_results" not in st.session_state:
        st.subheader("모형 구조")
        st.markdown(
            """
            1. **가구 생성 계층**: 기준 4인 가구 부하곡선에 사용량 규모, 재실패턴, 피크시각, 제어수용도의 차이를 부여해 고유한 가구를 생성함  
            2. **군집화 계층**: 일사용량·피크·주간/저녁 비중·유연성·수용도를 기준으로 유사 고객을 묶음  
            3. **상위 최적화 계층**: 변압기 용량초과, 계통부담, 고객불편, 군집 간 제어불균형을 동시에 최소화함  
            4. **하위 실행 계층**: 군집별 목표를 각 가구에 재배분하는 구조를 가정함
            """
        )
        return

    if run:
        with st.spinner("100개 가상가구 생성 → 군집화 → OR-Tools 최적화 중입니다..."):
            households, profiles = generate_households(int(household_count), season, day_type, int(random_seed))
            households, clusters = cluster_households(households, profiles, int(cluster_count), int(random_seed))
            unmanaged = profiles.sum(axis=0)
            capacity_kw = float(unmanaged.max() * capacity_ratio / 100.0)
            hourly, cluster_result, metrics = optimize_clusters(
                clusters, season, day_type, control_mode, capacity_kw
            )
            cluster_summary = build_cluster_summary(households)
            st.session_state["multi_results"] = {
                "households": households,
                "profiles": profiles,
                "cluster_summary": cluster_summary,
                "cluster_result": cluster_result,
                "hourly": hourly,
                "metrics": metrics,
                "settings": pd.DataFrame([{
                    "가구수": household_count,
                    "군집수": cluster_count,
                    "계절": season,
                    "대표일": day_type,
                    "제어모드": control_mode,
                    "변압기용량비율(%)": capacity_ratio,
                    "변압기용량(kW)": capacity_kw,
                    "가상가구생성번호": random_seed,
                }]),
            }

    r = st.session_state.get("multi_results")
    if not r:
        return

    metrics = r["metrics"]
    hourly = r["hourly"]
    households = r["households"]
    cluster_summary = r["cluster_summary"]
    cluster_result = r["cluster_result"]

    st.subheader("1. 핵심 결과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("제어 전 피크", f"{metrics['제어전피크(kW)']:.1f} kW")
    c2.metric("제어 후 피크", f"{metrics['제어후피크(kW)']:.1f} kW", f"-{metrics['피크감축률(%)']:.1f}%")
    c3.metric("용량초과 시간", f"{metrics['제어후_용량초과시간']}시간", f"제어 전 {metrics['제어전_용량초과시간']}시간")
    c4.metric("평균 고객편의", f"{metrics['평균편의점수']:.1f}점")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("이동 전력량", f"{metrics['이동전력량(kWh)']:.1f} kWh")
    c6.metric("감축 전력량", f"{metrics['감축전력량(kWh)']:.1f} kWh")
    c7.metric("제어 후 초과전력량", f"{metrics['제어후_초과전력량(kWh)']:.1f} kWh")
    c8.metric("군집 간 부담격차", f"{metrics['군집간_가구당부담격차(kWh)']:.3f} kWh/가구")

    st.subheader("2. 변압기 총부하 제어")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly["시간"], y=hourly["제어전(kW)"], mode="lines+markers", name="제어 전"))
    fig.add_trace(go.Scatter(x=hourly["시간"], y=hourly["제어후(kW)"], mode="lines+markers", name="제어 후"))
    fig.add_trace(go.Scatter(x=hourly["시간"], y=hourly["변압기용량(kW)"], mode="lines", name="변압기 용량", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="시간", yaxis_title="전력(kW)", hovermode="x unified", height=430)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("3. 군집 구성")
        fig2 = go.Figure(go.Bar(x=cluster_summary["군집"].astype(str), y=cluster_summary["가구수"], text=cluster_summary["가구수"], textposition="auto"))
        fig2.update_layout(xaxis_title="군집", yaxis_title="가구 수", height=350)
        st.plotly_chart(fig2, use_container_width=True)
    with right:
        st.subheader("4. 군집별 평균 사용량")
        fig3 = go.Figure(go.Bar(x=cluster_summary["군집"].astype(str), y=cluster_summary["평균일사용량_kWh"], text=cluster_summary["평균일사용량_kWh"].round(1), textposition="auto"))
        fig3.update_layout(xaxis_title="군집", yaxis_title="가구당 일사용량(kWh)", height=350)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("5. 군집별 제어 결과")
    display_cluster = cluster_result.copy()
    for col in display_cluster.select_dtypes(include=["float"]).columns:
        display_cluster[col] = display_cluster[col].round(3)
    st.dataframe(display_cluster, use_container_width=True, hide_index=True)

    with st.expander("100가구 생성자료와 군집 배정 확인"):
        display_households = households.copy()
        for col in display_households.select_dtypes(include=["float"]).columns:
            display_households[col] = display_households[col].round(4)
        st.dataframe(display_households, use_container_width=True, hide_index=True, height=480)

    st.subheader("6. 결과 내려받기")
    download = make_csv_zip({
        "시뮬레이션설정": r["settings"],
        "100가구_생성자료": households,
        "군집요약": cluster_summary,
        "군집별_제어결과": cluster_result,
        "시간대별_총부하": hourly,
        "핵심지표": pd.DataFrame([metrics]),
    })
    st.download_button(
        "결과자료 ZIP(CSV) 다운로드",
        data=download,
        file_name="100가구_군집형_수요관리_결과.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.warning(
        "해석상 유의: 가구별 프로파일과 유연성은 제공된 4인 가구 예시를 바탕으로 임의 생성한 값이며, "
        "배전선로 전압·무효전력·통신지연·실제 고객 수동해제는 아직 반영하지 않았습니다. "
        "따라서 본 결과는 계층형 제어 개념검증에만 사용해야 합니다."
    )


if __name__ == "__main__":
    main()
