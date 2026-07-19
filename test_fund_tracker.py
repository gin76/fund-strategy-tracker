"""基金定投策略核心逻辑的单元测试"""

import pytest
from fund_tracker import (
    calc_drawdown,
    calc_sell_point,
    get_tier_multiplier,
    init_fund_state,
    generate_signals,
    execute_signals,
    calc_ttm_dividend_yield,
    ensure_state_defaults,
    simulate_backtest,
    DEFAULT_CAPITAL,
    DEFAULT_BASE_RATIO,
    DEFAULT_TIER_STEP,
    DEFAULT_MULTIPLIER,
    DEFAULT_ENTRY_THRESHOLD,
)


# ── 辅助函数：创建一个用于测试的 fund 状态 ──────────────────────────

def _make_fund(**overrides):
    """创建一个 fund 状态字典，支持覆盖任意字段"""
    fund = init_fund_state(
        fund_code="000001",
        name="测试基金",
        capital=DEFAULT_CAPITAL,
        base_ratio=DEFAULT_BASE_RATIO,
        tier_step=DEFAULT_TIER_STEP,
        multiplier=DEFAULT_MULTIPLIER,
        entry_threshold=DEFAULT_ENTRY_THRESHOLD,
        sell_rule="momentum",
    )
    fund.update(overrides)
    return fund


# ══════════════════════════════════════════════════════════════════════
# calc_drawdown
# ══════════════════════════════════════════════════════════════════════

def test_calc_drawdown_normal():
    assert calc_drawdown(9.0, 10.0) == pytest.approx(10.0)
    assert calc_drawdown(8.5, 10.0) == pytest.approx(15.0)


def test_calc_drawdown_no_drawdown():
    assert calc_drawdown(10.0, 10.0) == 0.0


def test_calc_drawdown_negative():
    """上涨时回撤为负值"""
    assert calc_drawdown(11.0, 10.0) == pytest.approx(-10.0)


# ══════════════════════════════════════════════════════════════════════
# calc_sell_point
# ══════════════════════════════════════════════════════════════════════

def test_calc_sell_point():
    """趋势型卖出点：peak + (peak-trough)/2"""
    assert calc_sell_point(10.0, 8.0) == 11.0


def test_calc_sell_point_mean_revert():
    """均值回归型卖出点：就是 peak 本身"""
    assert calc_sell_point(10.0, 8.0, "mean_revert") == 10.0


def test_calc_sell_point_momentum_explicit():
    """显式指定 momentum 模式"""
    assert calc_sell_point(10.0, 8.0, "momentum") == 11.0


# ══════════════════════════════════════════════════════════════════════
# get_tier_multiplier
# ══════════════════════════════════════════════════════════════════════

def test_multiplier_below_threshold():
    """回撤未达入场阈值返回 0"""
    assert get_tier_multiplier(9.0, entry_threshold=10, tier_step=5, multiplier=1.2) == 0.0


def test_multiplier_first_tier():
    """第 0 档：entry ~ entry+step，乘数 = 1.0"""
    assert get_tier_multiplier(10.0, entry_threshold=10, tier_step=5, multiplier=1.2) == 1.0
    assert get_tier_multiplier(14.0, entry_threshold=10, tier_step=5, multiplier=1.2) == 1.0


def test_multiplier_second_tier():
    """第 1 档：乘数 = multiplier^1"""
    assert get_tier_multiplier(15.0, entry_threshold=10, tier_step=5, multiplier=1.2) == 1.2


def test_multiplier_third_tier():
    """第 2 档：乘数 = multiplier^2"""
    assert get_tier_multiplier(20.0, entry_threshold=10, tier_step=5, multiplier=1.2) == pytest.approx(1.44)


# ══════════════════════════════════════════════════════════════════════
# init_fund_state
# ══════════════════════════════════════════════════════════════════════

def test_init_fund_state_has_sell_signal_pending():
    fund = init_fund_state("000001", "测试", 100000, 0.03, 5, 1.2, 10)
    assert fund["sell_signal_pending"] is False


# ══════════════════════════════════════════════════════════════════════
# generate_signals
# ══════════════════════════════════════════════════════════════════════

def test_first_run_initializes_peak():
    """首次运行：peak 为 None，自动设为当前净值"""
    fund = _make_fund(peak_price=None)
    signals = generate_signals(fund, current_nav=10.0)

    assert fund["peak_price"] == 10.0
    assert len(signals) == 1
    assert signals[0]["type"] == "info"


def test_new_high_starts_new_cycle_but_retains_trough_for_sell_target():
    """创新高时开始新周期，但保留低点以继续计算卖点"""
    fund = _make_fund(
        peak_price=10.0,
        trough_price=8.0,
        last_trigger_level=15,
        mode="buying",
        total_invested=5000,
        total_shares=600,
        sell_signal_pending=True,
    )
    signals = generate_signals(fund, current_nav=11.0)

    assert fund["peak_price"] == 11.0
    assert fund["trough_price"] == 8.0
    assert fund["last_trigger_level"] is None
    assert fund["mode"] == "waiting"
    assert fund["sell_signal_pending"] is False


def test_no_signal_when_drawdown_below_threshold():
    """回撤不足阈值：无交易信号"""
    fund = _make_fund(peak_price=10.0, trough_price=None)
    signals = generate_signals(fund, current_nav=9.5)

    buy_sigs = [s for s in signals if s["type"] == "buy"]
    assert len(buy_sigs) == 0
    assert fund["mode"] == "waiting"


def test_buy_signal_single_level():
    """回撤刚刚跨过阈值，触发一次买入"""
    fund = _make_fund(peak_price=10.0, trough_price=None, last_trigger_level=None)

    # 回撤 11% → 跨过 10%（初始触发点）和 11%
    signals = generate_signals(fund, current_nav=8.9)

    buy_sigs = [s for s in signals if s["type"] == "buy"]
    assert len(buy_sigs) >= 1
    for sig in buy_sigs:
        assert sig["amount"] > 0


def test_buy_signal_not_repeated():
    """同一策略档位实际执行后不重复触发"""
    fund = _make_fund(peak_price=10.0, trough_price=None, last_trigger_level=None)

    # 首次触发：回撤 11%
    signals1 = generate_signals(fund, current_nav=8.9)
    buy_count1 = len([s for s in signals1 if s["type"] == "buy"])
    execute_signals(fund, signals1, "2026-01-10")

    # 再次触发同一级别
    signals2 = generate_signals(fund, current_nav=8.91)
    buy_count2 = len([s for s in signals2 if s["type"] == "buy"])

    assert buy_count2 == 0


def test_one_signal_per_strategy_tier_not_per_integer_percent():
    fund = _make_fund(
        peak_price=10.0, entry_threshold=10, tier_step=5,
        base_ratio=0.03, capital=100000,
    )
    signals = generate_signals(fund, current_nav=8.9)  # 11%回撤，仍是第1档
    buys = [s for s in signals if s["type"] == "buy"]
    assert len(buys) == 1
    assert buys[0]["tier_index"] == 0
    assert buys[0]["amount"] == 3000


def test_fund_budget_caps_generated_orders():
    fund = _make_fund(
        peak_price=10.0, entry_threshold=5, tier_step=3,
        multiplier=1.5, base_ratio=0.03, capital=10000,
        total_invested=9700,
    )
    signals = generate_signals(fund, current_nav=9.4)
    buys = [s for s in signals if s["type"] == "buy"]
    assert len(buys) == 1
    assert buys[0]["amount"] == 300


def test_state_migration_restores_independent_fund_decisions():
    fund = _make_fund(allow_new_buys=False)
    state = ensure_state_defaults({
        "funds": {"000001": fund},
        "portfolio": {"total_budget": 10000, "bucket_weights": {"other": 1.0}},
    })
    assert state["funds"]["000001"]["allow_new_buys"] is True
    assert "portfolio" not in state


def test_preview_buy_signal_does_not_advance_trigger_level():
    """只预览买入信号时，同一信号下次仍应出现"""
    fund = _make_fund(peak_price=10.0, last_trigger_level=None)

    first = generate_signals(fund, current_nav=8.9, advance_signals=False)
    second = generate_signals(fund, current_nav=8.9, advance_signals=False)

    assert any(s["type"] == "buy" for s in first)
    assert any(s["type"] == "buy" for s in second)
    assert fund["last_trigger_level"] is None


def test_preview_sell_signal_does_not_mark_it_pending():
    """只预览卖出信号时，未执行前应持续提醒"""
    fund = _make_fund(
        peak_price=10.0,
        trough_price=8.0,
        total_invested=5000,
        total_shares=600,
    )

    signals = generate_signals(fund, current_nav=11.0, advance_signals=False)

    assert any(s["type"] == "sell" for s in signals)
    assert fund["sell_signal_pending"] is False


def test_sell_signal_not_repeated():
    """卖出信号只生成一次，sell_signal_pending 阻止重复"""
    fund = _make_fund(
        peak_price=10.0,
        trough_price=8.0,
        total_invested=5000,
        total_shares=600,
        sell_signal_pending=False,
    )

    # 卖出点 = 10 + (10-8)/2 = 11
    signals1 = generate_signals(fund, current_nav=11.5)
    sell_count1 = len([s for s in signals1 if s["type"] == "sell"])
    assert sell_count1 == 1
    assert fund["sell_signal_pending"] is True

    # 再次触发：应被跳过
    signals2 = generate_signals(fund, current_nav=12.0)
    sell_count2 = len([s for s in signals2 if s["type"] == "sell"])
    assert sell_count2 == 0


def test_sell_signal_mean_revert():
    """均值回归模式：卖出点 = peak 本身"""
    fund = _make_fund(
        peak_price=10.0,
        trough_price=8.0,
        total_invested=5000,
        total_shares=600,
        sell_rule="mean_revert",
        sell_signal_pending=False,
    )

    # 价格回到前高 10.0，应触发卖出
    signals = generate_signals(fund, current_nav=10.0)
    sell_count = len([s for s in signals if s["type"] == "sell"])
    assert sell_count == 1

    # 价格略低于前高，不应触发
    fund2 = _make_fund(
        peak_price=10.0,
        trough_price=8.0,
        total_invested=5000,
        total_shares=600,
        sell_rule="mean_revert",
        sell_signal_pending=False,
    )
    signals2 = generate_signals(fund2, current_nav=9.99)
    sell_count2 = len([s for s in signals2 if s["type"] == "sell"])
    assert sell_count2 == 0


# ══════════════════════════════════════════════════════════════════════
# execute_signals
# ══════════════════════════════════════════════════════════════════════

def test_execute_buy():
    fund = _make_fund()
    buy_signal = [{"type": "buy", "nav": 8.0, "amount": 3000}]

    execute_signals(fund, buy_signal, "2026-01-15")

    assert fund["total_invested"] == 3000
    assert fund["total_shares"] == pytest.approx(375.0)  # 3000/8
    assert len(fund["positions"]) == 1
    assert fund["positions"][0]["nav"] == 8.0


def test_execute_sell_clears_position_and_resets_signal():
    fund = _make_fund(
        total_invested=5000,
        total_shares=600,
        sell_signal_pending=True,
        positions=[{"date": "2026-01-10", "nav": 8.0, "amount": 5000, "shares": 625}],
    )
    sell_signal = [{"type": "sell", "nav": 11.5, "amount": 6900}]

    execute_signals(fund, sell_signal, "2026-02-01")

    assert fund["total_invested"] == 0.0
    assert fund["total_shares"] == 0.0
    assert fund["positions"] == []
    assert fund["peak_price"] == 11.5
    assert fund["trough_price"] is None
    assert fund["mode"] == "waiting"
    assert fund["sell_signal_pending"] is False


def test_ttm_dividend_yield_uses_rolling_365_days():
    dividends = [
        ["2024-12-01", 0.10],
        ["2025-08-01", 0.02],
        ["2026-02-01", 0.03],
    ]
    result = calc_ttm_dividend_yield(1.0, dividends, "2026-07-18")
    assert result == pytest.approx(5.0)


def test_spread_buy_is_monthly_deduplicated_after_execution():
    fund = _make_fund(
        sell_rule="spread", fund_code="159307", asset_bucket="dividend_low_vol",
        peak_price=1.0, capital=50000, monthly_base=5000,
        spread_double_threshold=3.0, spread_clear_threshold=1.0,
        dividends=[["2025-09-01", 0.02], ["2026-01-01", 0.02], ["2026-04-01", 0.02]],
    )
    first = generate_signals(fund, 1.0, "2026-07-18", bond_yield=2.0)
    assert any(s["type"] == "buy" for s in first)
    execute_signals(fund, first, "2026-07-18")
    second = generate_signals(fund, 1.0, "2026-07-20", bond_yield=2.0)
    assert not any(s["type"] == "buy" for s in second)
    assert fund["last_spread_invest_month"] == "2026-07"


def test_backtest_reports_cash_risk_and_benchmarks():
    fund = _make_fund(
        capital=10000, peak_price=None, entry_threshold=5,
        tier_step=5, base_ratio=0.10, multiplier=1.2,
    )
    history = [
        {"date": "2025-01-02", "nav": 10.0},
        {"date": "2025-02-03", "nav": 9.0},
        {"date": "2025-03-03", "nav": 8.0},
        {"date": "2025-04-01", "nav": 9.5},
        {"date": "2025-05-02", "nav": 11.5},
    ]
    result = simulate_backtest(fund, history, 0.001, 0.001)
    assert result["initial_cash"] == 10000
    assert result["final_equity"] >= 0
    assert 0 <= result["max_drawdown"] <= 1
    assert "benchmark_buy_hold_return" in result
    assert "benchmark_monthly_dca_return" in result
    assert result["total_fees"] >= 0
