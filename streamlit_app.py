from __future__ import annotations

import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from subscription_energy_optimizer import (
    CURRENT_RATE_LABEL,
    export_results,
    get_overage_rates,
    load_workbook_data,
    run_batch,
    solve_month,
)

DEFAULT_INPUT_NAME = "default_input.xlsx"
FALLBACK_INPUT_NAME = "구독형_전기요금_최적제어_입력자료.xlsx"


def _normalized_filename(name: str) -> str:
    """iPad/macOS와 Linux 간 한글 파일명 정규화 차이를 제거합니다."""
    return unicodedata.normalize("NFC", name).casefold()


def _find_default_input() -> Path | None:
    # 1순위: 영문 파일명(가장 안정적)
    english_path = APP_DIR / DEFAULT_INPUT_NAME
    if english_path.exists():
        return english_path

    # 2순위: 기존 한글 파일명
    korean_path = APP_DIR / FALLBACK_INPUT_NAME
    if korean_path.exists():
        return korean_path

    # 3순위: Unicode NFC/NFD 차이를 무시하여 같은 이름 탐색
    targets = {
        _normalized_filename(DEFAULT_INPUT_NAME),
        _normalized_filename(FALLBACK_INPUT_NAME),
    }
    xlsx_files = [p for p in APP_DIR.rglob("*.xlsx") if not p.name.startswith("~$")]
    for path in xlsx_files:
        if _normalized_filename(path.name) in targets:
            return path

    # 4순위: '입력자료'가 포함된 Excel 파일 자동 탐색
    for path in xlsx_files:
        normalized = _normalized_filename(path.name)
        if "입력자료" in normalized or "input" in normalized:
            return path

    # Excel 파일이 하나뿐이면 해당 파일 사용
    if len(xlsx_files) == 1:
        return xlsx_files[0]
    return None

st.set_page_config(
    page_title="구독형 전기요금 최적제어",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# iPad에서 조작하기 쉽도록 입력창과 버튼을 크게 표시합니다.
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1rem; padding-bottom: 3rem;}
    div.stButton > button, div.stDownloadButton > button {
        min-height: 3.2rem; font-size: 1.05rem; font-weight: 700; width: 100%;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        min-height: 3rem;
    }
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    [data-testid="stMetricLabel"] {font-size: 0.95rem;}
    .notice {
        padding: 0.85rem 1rem; border-radius: 0.65rem;
        background: #f3f7fb; border: 1px solid #c8d8e8; margin-bottom: 0.8rem;
    }
    @media (max-width: 900px) {
        .block-container {padding-left: 0.8rem; padding-right: 0.8rem;}
        h1 {font-size: 1.75rem !important;}
        [data-testid="stMetricValue"] {font-size: 1.35rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ 구독형 전기요금 최적제어 시뮬레이터")
st.caption("Google OR-Tools 기반 연구용 프로토타입 · iPad Safari에서 버튼 방식으로 실행")
st.markdown(
    """
    <div class="notice">
    아래에서 조건을 고른 뒤 <b>최적화 실행</b>만 누르면 됩니다. 코드를 수정할 필요가 없습니다.<br>
    공개 웹앱에는 실제 고객의 AMI·개인정보를 올리지 말고, 현재와 같은 가상·비식별 자료만 사용하십시오.
    </div>
    """,
    unsafe_allow_html=True,
)


def _won(value: float) -> str:
    return f"{value:,.0f}원"


def _kwh(value: float) -> str:
    return f"{value:,.1f}kWh"


def _kw(value: float) -> str:
    return f"{value:,.2f}kW"


def _load_input_file(uploaded_file) -> Tuple[str, tempfile.TemporaryDirectory | None]:
    if uploaded_file is None:
        default_input = _find_default_input()
        if default_input is None:
            available = ", ".join(sorted(str(p.relative_to(APP_DIR)) for p in APP_DIR.rglob("*.xlsx"))) or "없음"
            raise FileNotFoundError(
                "내장 입력 Excel 파일을 찾지 못했습니다. "
                f"저장소 최상위의 Excel 파일: {available}"
            )
        return str(default_input), None

    temp_dir = tempfile.TemporaryDirectory()
    input_path = Path(temp_dir.name) / "uploaded_input.xlsx"
    input_path.write_bytes(uploaded_file.getvalue())
    return str(input_path), temp_dir


def _apply_plan_settings(data, basic_fee: int, basic_limit: int, premium_fee: int, premium_limit: int) -> None:
    settings: Dict[str, Tuple[int, int]] = {
        "기본형": (basic_fee, basic_limit),
        "프리미엄형": (premium_fee, premium_limit),
    }
    for name, (fee, limit) in settings.items():
        mask = data.plans["요금제"] == name
        if mask.any():
            data.plans.loc[mask, "월 구독료(원)"] = float(fee)
            data.plans.loc[mask, "기본 제공량(kWh)"] = float(limit)


def _make_excel_bytes(summary_df: pd.DataFrame, hourly_df: pd.DataFrame, apps_df: pd.DataFrame, data) -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "구독형_전기요금_최적화_결과.xlsx"
        export_results(output_path, summary_df, hourly_df, apps_df, data)
        return output_path.read_bytes()


# 1. 입력자료
st.subheader("1. 입력자료")
source = st.radio(
    "사용할 자료",
    ["내장된 4인 가구 예시", "내 Excel 파일 업로드"],
    horizontal=True,
    help="처음에는 내장 예시를 사용하면 됩니다.",
)
uploaded = None
if source == "내 Excel 파일 업로드":
    uploaded = st.file_uploader(
        "입력 Excel 선택",
        type=["xlsx"],
        help="기존 입력자료와 동일한 시트 구조의 Excel 파일을 선택하십시오.",
    )
    if uploaded is None:
        st.info("Excel 파일을 선택하면 아래 설정과 실행 버튼이 활성화됩니다.")

input_ready = source == "내장된 4인 가구 예시" or uploaded is not None

if input_ready:
    try:
        input_path, temp_holder = _load_input_file(uploaded)
        data = load_workbook_data(input_path)
    except Exception as exc:
        st.error(f"입력자료를 읽지 못했습니다: {exc}")
        st.stop()

    # 2. 요금제 및 제어조건
    st.subheader("2. 분석조건")
    col1, col2, col3 = st.columns(3)
    with col1:
        season = st.selectbox("계절", ["봄가을", "여름", "겨울"], index=1)
        plan_name = st.selectbox("요금제", ["기본형", "프리미엄형"], index=0)
    with col2:
        mode = st.selectbox("냉난방 모드", ["편의 우선", "균형", "절약 우선"], index=1)
        customer_override = st.toggle(
            "고객 수동해제 적용",
            value=False,
            help="켜면 자동제어를 중단하고 기준 사용패턴을 유지합니다.",
        )
    with col3:
        compare_all_rates = st.toggle("초과단가 4개 모두 비교", value=True)
        if not compare_all_rates:
            rate_labels = [label for label, _ in get_overage_rates(data, season)] + ["직접 입력"]
            rate_label = st.selectbox("초과단가", rate_labels, index=1)
            if rate_label == "직접 입력":
                overage_rate = float(st.number_input("직접 입력 단가(원/kWh)", 0, 2000, 300, 10))
            else:
                rate_map = dict(get_overage_rates(data, season))
                overage_rate = float(rate_map[rate_label])
        else:
            rate_label = "300원"
            overage_rate = 300.0

    with st.expander("요금제 금액·제공량 변경", expanded=False):
        p1, p2 = st.columns(2)
        with p1:
            basic_fee = int(st.number_input("기본형 월 구독료(원)", 0, 1_000_000, 84_900, 1_000))
            basic_limit = int(st.number_input("기본형 제공량(kWh)", 0, 10_000, 450, 10))
        with p2:
            premium_fee = int(st.number_input("프리미엄형 월 구독료(원)", 0, 2_000_000, 249_900, 1_000))
            premium_limit = int(st.number_input("프리미엄형 제공량(kWh)", 0, 20_000, 1_000, 10))

    _apply_plan_settings(data, basic_fee, basic_limit, premium_fee, premium_limit)

    st.subheader("3. 실행")
    run_clicked = st.button("최적화 실행", type="primary", width="stretch")

    if run_clicked:
        try:
            with st.spinner("OR-Tools가 최적 운전계획을 계산하고 있습니다…"):
                if compare_all_rates:
                    summary_df, hourly_df, apps_df = run_batch(
                        data,
                        seasons=[season],
                        plans=[plan_name],
                        modes=[mode],
                        include_all_overage_rates=True,
                        customer_override=customer_override,
                        max_solve_seconds=8.0,
                        detail_selector=(season, plan_name, mode, 300),
                    )
                    representative = summary_df.loc[
                        summary_df["초과단가시나리오"] == "300원"
                    ].iloc[0].to_dict()
                else:
                    result = solve_month(
                        data,
                        season=season,
                        plan_name=plan_name,
                        mode=mode,
                        overage_label=rate_label,
                        overage_rate=overage_rate,
                        customer_override=customer_override,
                        max_solve_seconds=8.0,
                    )
                    summary_df = pd.DataFrame([result.summary])
                    hourly_df = result.hourly
                    apps_df = result.appliances
                    representative = result.summary

                excel_bytes = _make_excel_bytes(summary_df, hourly_df, apps_df, data)
                st.session_state["optimizer_result"] = {
                    "summary": summary_df,
                    "hourly": hourly_df,
                    "apps": apps_df,
                    "representative": representative,
                    "excel": excel_bytes,
                    "condition": f"{season} · {plan_name} · {mode}",
                    "compare_all": compare_all_rates,
                }
        except Exception as exc:
            st.error(f"최적화 중 오류가 발생했습니다: {exc}")

    # 4. 결과
    if "optimizer_result" in st.session_state:
        result = st.session_state["optimizer_result"]
        summary_df = result["summary"]
        hourly_df = result["hourly"]
        apps_df = result["apps"]
        s = result["representative"]

        st.divider()
        st.subheader(f"4. 결과 · {result['condition']}")
        if result["compare_all"]:
            st.caption("아래 핵심지표는 300원/kWh 초과단가 시나리오이며, 다른 단가는 비교표에서 확인합니다.")

        r1 = st.columns(4)
        r1[0].metric("제어 전 월 사용량", _kwh(float(s["기준월사용량(kWh)"])))
        r1[1].metric(
            "제어 후 월 사용량",
            _kwh(float(s["최적월사용량(kWh)"])),
            delta=f"-{float(s['월감축량(kWh)']):,.1f}kWh",
        )
        r1[2].metric("제공량 초과", _kwh(float(s["초과량(kWh)"])))
        r1[3].metric("고객 편의지수", f"{float(s['고객편의지수(100점)']):,.1f}점")

        r2 = st.columns(4)
        r2[0].metric("구독 최종 납부액", _won(float(s["구독최종납부액(원)"])))
        r2[1].metric("현행요금 추정", _won(float(s["현행요금_최적(원)"])))
        r2[2].metric("피크 감축", _kw(float(s["피크감축(kW)"])))
        r2[3].metric("월간 부하 이동량", _kwh(float(s["월이동량(kWh)"])))

        recommendation = str(s["요금제권고"])
        reason = str(s.get("권고사유", "") or "")
        if "전환" in recommendation:
            st.warning(f"요금제 권고: {recommendation}" + (f" — {reason}" if reason else ""))
        else:
            st.success(f"요금제 권고: {recommendation}" + (f" — {reason}" if reason else ""))

        tab_curve, tab_rates, tab_appliances = st.tabs(["부하곡선", "초과단가 비교", "가전별 변경"])
        with tab_curve:
            weekday_tab, weekend_tab = st.tabs(["주중", "주말"])
            for tab, day_type in [(weekday_tab, "주중"), (weekend_tab, "주말")]:
                with tab:
                    day_df = hourly_df.loc[hourly_df["요일"] == day_type].copy()
                    day_df = day_df.set_index("시간대")[["기준부하(kWh)", "최적부하(kWh)"]]
                    st.line_chart(day_df, height=340)
                    st.caption("가전 사용시간 이동으로 일부 저부하 시간에는 최적부하가 기준부하보다 높아질 수 있습니다.")

        with tab_rates:
            display_cols = [
                "초과단가시나리오",
                "초과단가(원/kWh)",
                "최적월사용량(kWh)",
                "초과량(kWh)",
                "구독최종납부액(원)",
                "현행요금_최적(원)",
                "요금제권고",
            ]
            visible = [c for c in display_cols if c in summary_df.columns]
            st.dataframe(
                summary_df[visible].style.format(
                    {
                        "초과단가(원/kWh)": "{:,.1f}",
                        "최적월사용량(kWh)": "{:,.1f}",
                        "초과량(kWh)": "{:,.1f}",
                        "구독최종납부액(원)": "{:,.0f}",
                        "현행요금_최적(원)": "{:,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        with tab_appliances:
            changed = apps_df.loc[
                (apps_df["이동량(kWh/일)"] > 1e-9)
                | (apps_df["감축량(kWh/일)"] > 1e-9)
                | (apps_df["기준사용시간"] != apps_df["최적사용시간"])
            ].copy()
            if changed.empty:
                st.info("현재 조건에서는 변경된 가전이 없습니다.")
            else:
                st.dataframe(changed, width="stretch", hide_index=True, height=430)

        st.download_button(
            "결과 Excel 다운로드",
            data=result["excel"],
            file_name="구독형_전기요금_최적화_결과.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            width="stretch",
        )

        with st.expander("결과 해석 시 유의사항"):
            st.markdown(
                """
                - 현재 버전은 1시간 단위의 연구용 시뮬레이션입니다.
                - 고객 편의지수는 시나리오 비교용 지표이며 실제 만족도 조사 결과가 아닙니다.
                - 냉난방은 건물 열모델 대신 모드별 최대 감축률을 사용합니다.
                - 실제 원격제어 전에는 기기 통신, 고객 동의, 수동해제, 고장 시 복귀 및 안전로직을 별도로 검증해야 합니다.
                """
            )

    if temp_holder is not None:
        # 현재 Streamlit 실행이 끝날 때까지 임시 입력파일을 유지합니다.
        st.session_state["_input_temp_holder"] = temp_holder
