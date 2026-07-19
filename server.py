"""基金定投跟踪器 — FastAPI Web 后端"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fund_tracker import (
    BASE_DIR,
    DEFAULT_BASE_RATIO,
    DEFAULT_CAPITAL,
    DEFAULT_ENTRY_THRESHOLD,
    DEFAULT_MULTIPLIER,
    DEFAULT_SELL_RULE,
    DEFAULT_TIER_STEP,
    DEFAULT_SPREAD_MONTHLY_BASE,
    DEFAULT_SPREAD_DOUBLE_THRESHOLD,
    DEFAULT_SPREAD_CLEAR_THRESHOLD,
    STATE_FILE,
    calc_drawdown,
    calc_sell_point,
    execute_signals,
    fetch_bond_yield,
    fetch_bond_yield_history,
    fetch_etf_history,
    fetch_etf_price,
    fetch_latest_nav,
    fetch_nav_history,
    generate_signals,
    init_fund_state,
    load_state,
    replay_history,
    save_state,
    simulate_backtest,
)

# ── Pydantic models ────────────────────────────────────────────────────


class FundAddRequest(BaseModel):
    capital: float = DEFAULT_CAPITAL
    base_ratio: float = DEFAULT_BASE_RATIO
    tier_step: float = DEFAULT_TIER_STEP
    multiplier: float = DEFAULT_MULTIPLIER
    entry: float = DEFAULT_ENTRY_THRESHOLD
    sell_rule: str = DEFAULT_SELL_RULE
    # 现有持仓（选填，填了直接显示正确数据）
    market_value: Optional[float] = None
    pnl: Optional[float] = None
    shares: Optional[float] = None
    # 利差定投参数（仅 sell_rule="spread" 时使用）
    monthly_base: Optional[float] = None
    spread_double_threshold: Optional[float] = None
    spread_clear_threshold: Optional[float] = None
    dividends: Optional[list] = None


class FundConfigRequest(BaseModel):
    capital: Optional[float] = None
    base_ratio: Optional[float] = None
    entry: Optional[float] = None
    tier_step: Optional[float] = None
    multiplier: Optional[float] = None
    sell_rule: Optional[str] = None
    start_date: Optional[str] = None
    # 利差定投参数
    monthly_base: Optional[float] = None
    spread_double_threshold: Optional[float] = None
    spread_clear_threshold: Optional[float] = None
    dividends: Optional[list] = None


class BacktestRequest(BaseModel):
    days: int = 365
    entry_threshold: Optional[float] = None
    tier_step: Optional[float] = None
    multiplier: Optional[float] = None
    base_ratio: Optional[float] = None
    capital: Optional[float] = None
    sell_rule: Optional[str] = None
    buy_fee_rate: float = 0.001
    sell_fee_rate: float = 0.001


class ManualTradeRequest(BaseModel):
    amount: float  # 正数=加仓，负数=减仓
    date: str  # YYYY-MM-DD
    nav: Optional[float] = None  # 不填则自动获取当日净值


class EditHoldingsRequest(BaseModel):
    total_invested: float   # cost * shares
    total_shares: float

# ── FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(title="基金定投跟踪器", version="1.0.0")


# ── Helpers ────────────────────────────────────────────────────────────

FUND_CODE_PATTERN = re.compile(r"^\d{6}$")

def _validate_fund_code(code: str):
    if not FUND_CODE_PATTERN.match(code):
        raise HTTPException(status_code=400, detail="基金代码必须是6位数字")

def _fetch_price_for_fund(fund_code: str, sell_rule: str | None = None) -> dict | None:
    """根据策略类型选择数据源：利差 → 新浪财经，其他 → 天天基金（失败则降级到新浪）"""
    if sell_rule == "spread":
        return fetch_etf_price(fund_code)
    result = fetch_latest_nav(fund_code)
    if result is None:
        result = fetch_etf_price(fund_code)
    return result


def _fund_summary(fund: dict) -> dict:
    """从 fund state 生成列表页摘要（不调 API，用上次刷新存下的净值）"""
    peak = fund.get("peak_price")
    current_nav = fund.get("last_nav") or peak  # 最近净值
    drawdown = 0.0

    if peak and current_nav:
        drawdown = round(calc_drawdown(current_nav, peak), 2)

    trough = fund.get("trough_price")
    summary = {
        "fund_code": fund["fund_code"],
        "name": fund["name"],
        "current_nav": current_nav,
        "peak_price": peak,
        "peak_date": fund.get("peak_date"),
        "trough_price": trough,
        "drawdown": drawdown,
        "mode": fund.get("mode", "waiting"),
        "sell_rule": fund.get("sell_rule", "momentum"),
        "sell_signal_pending": fund.get("sell_signal_pending", False),
        "total_invested": round(fund.get("total_invested", 0), 2),
        "total_shares": round(fund.get("total_shares", 0), 4),
        "last_updated": fund.get("last_updated"),
        "entry_threshold": fund.get("entry_threshold", 10),
    }

    shares = fund.get("total_shares", 0)
    if shares > 0 and current_nav:
        avg_cost = fund["total_invested"] / shares
        market_value = current_nav * shares
        summary["avg_cost"] = round(avg_cost, 4)
        summary["market_value"] = round(market_value, 2)
        summary["pnl"] = round(market_value - fund["total_invested"], 2)
        if avg_cost > 0:
            summary["pnl_pct"] = round((current_nav - avg_cost) / avg_cost * 100, 2)

        # 当日涨跌幅 & 当日收益
        prev_nav = fund.get("prev_nav")
        if prev_nav and prev_nav > 0:
            daily_pct = round((current_nav - prev_nav) / prev_nav * 100, 2)
            summary["daily_change"] = daily_pct
            summary["daily_return"] = round((current_nav - prev_nav) * shares, 2)
        else:
            summary["daily_change"] = None
            summary["daily_return"] = None
    else:
        summary["daily_change"] = None
        summary["daily_return"] = None

    # 卖出参考信息
    if trough and fund.get("total_shares", 0) > 0 and peak and current_nav:
        sell_rule = fund.get("sell_rule", "momentum")
        sell_pt = calc_sell_point(peak, trough, sell_rule)
        if sell_pt is not None:
            summary["sell_point"] = round(sell_pt, 4)
            summary["pct_to_sell"] = round((sell_pt - current_nav) / current_nav * 100, 2)

    # 利差定投特有字段
    if fund.get("sell_rule") == "spread":
        summary["spread_signal"] = fund.get("spread_signal")
        summary["current_spread"] = fund.get("current_spread")
        summary["current_div_yield"] = fund.get("current_div_yield")
        summary["current_bond_yield"] = fund.get("current_bond_yield")

    return summary


def _fund_detail(fund: dict, current_nav: float, signals: list[dict]) -> dict:
    """生成基金详情响应"""
    peak = fund.get("peak_price") or current_nav
    trough = fund.get("trough_price")
    drawdown = round(calc_drawdown(current_nav, peak), 2)

    detail = {
        "fund_code": fund["fund_code"],
        "name": fund["name"],
        "current_nav": current_nav,
        "peak_price": peak,
        "peak_date": fund.get("peak_date"),
        "trough_price": trough,
        "drawdown": drawdown,
        "mode": fund.get("mode", "waiting"),
        "total_invested": round(fund.get("total_invested", 0), 2),
        "total_shares": round(fund.get("total_shares", 0), 4),
        "capital": fund.get("capital", DEFAULT_CAPITAL),
        "base_ratio": fund.get("base_ratio", DEFAULT_BASE_RATIO),
        "tier_step": fund.get("tier_step", DEFAULT_TIER_STEP),
        "multiplier": fund.get("multiplier", DEFAULT_MULTIPLIER),
        "entry_threshold": fund.get("entry_threshold", DEFAULT_ENTRY_THRESHOLD),
        "sell_rule": fund.get("sell_rule", DEFAULT_SELL_RULE),
        "sell_signal_pending": fund.get("sell_signal_pending", False),
        "last_updated": fund.get("last_updated"),
        "positions": fund.get("positions", []),
        "signals": [s for s in signals],
        "last_trigger_tier": fund.get("last_trigger_tier"),
    }

    if trough and fund.get("total_shares", 0) > 0:
        sell_pt = calc_sell_point(peak, trough, fund.get("sell_rule", "momentum"))
        if sell_pt is not None:
            detail["sell_point"] = round(sell_pt, 4)
            detail["pct_to_sell"] = round((sell_pt - current_nav) / current_nav * 100, 2)

    if fund.get("total_shares", 0) > 0:
        avg_cost = fund["total_invested"] / fund["total_shares"]
        market_value = current_nav * fund["total_shares"]
        detail["avg_cost"] = round(avg_cost, 4)
        detail["market_value"] = round(market_value, 2)
        detail["pnl"] = round(market_value - fund["total_invested"], 2)
        if avg_cost > 0:
            detail["pnl_pct"] = round((current_nav - avg_cost) / avg_cost * 100, 2)

        prev_nav = fund.get("prev_nav")
        if prev_nav and prev_nav > 0:
            detail["daily_change"] = round((current_nav - prev_nav) / prev_nav * 100, 2)
            detail["daily_return"] = round((current_nav - prev_nav) * fund["total_shares"], 2)

    # 利差定投特有字段
    if fund.get("sell_rule") == "spread":
        detail["spread_signal"] = fund.get("spread_signal")
        detail["current_spread"] = fund.get("current_spread")
        detail["current_div_yield"] = fund.get("current_div_yield")
        detail["current_bond_yield"] = fund.get("current_bond_yield")
        detail["monthly_base"] = fund.get("monthly_base", DEFAULT_SPREAD_MONTHLY_BASE)
        detail["spread_double_threshold"] = fund.get("spread_double_threshold", DEFAULT_SPREAD_DOUBLE_THRESHOLD)
        detail["spread_clear_threshold"] = fund.get("spread_clear_threshold", DEFAULT_SPREAD_CLEAR_THRESHOLD)
        detail["dividends"] = fund.get("dividends", [])
        detail["last_spread_invest_month"] = fund.get("last_spread_invest_month")

    return detail


# ── API Routes ─────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.get("/api/funds")
def list_funds():
    """获取所有基金的摘要列表和组合汇总"""
    state = load_state()
    funds = []
    fund_dict = state.get("funds", {})
    # 按 order 排序，没有 order 的放最后
    sorted_codes = sorted(fund_dict.keys(), key=lambda c: fund_dict[c].get("order", 9999))
    for code in sorted_codes:
        funds.append(_fund_summary(fund_dict[code]))

    # 组合汇总
    total_mv = sum(f.get("market_value", 0) or 0 for f in funds)
    total_invested = sum(f.get("total_invested", 0) or 0 for f in funds)
    total_pnl = sum(f.get("pnl", 0) or 0 for f in funds)
    total_daily = sum(f.get("daily_return", 0) or 0 for f in funds)
    total_pnl_pct = round(total_pnl / total_invested * 100, 2) if total_invested > 0 else 0
    total_daily_pct = round(total_daily / total_mv * 100, 2) if total_mv > 0 else 0

    # 取最近更新的日期作为组合日期
    dates = [f.get("last_updated", "") for f in funds if f.get("last_updated")]
    nav_date = max(dates) if dates else ""

    portfolio = {
        "total_market_value": round(total_mv, 2),
        "total_invested": round(total_invested, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": total_pnl_pct,
        "total_daily_return": round(total_daily, 2),
        "total_daily_pct": total_daily_pct,
        "nav_date": nav_date,
    }

    return {"funds": funds, "portfolio": portfolio}


@app.get("/api/funds/{fund_code}")
def get_fund(fund_code: str):
    """获取单只基金详情，实时获取净值并生成信号"""
    _validate_fund_code(fund_code)
    state = load_state()

    if fund_code not in state.get("funds", {}):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在跟踪列表中")

    fund = state["funds"][fund_code]
    sell_rule = fund.get("sell_rule")
    is_spread = sell_rule == "spread"

    info = _fetch_price_for_fund(fund_code, sell_rule)
    if info is None:
        raise HTTPException(status_code=502, detail=f"无法获取 {fund_code} 的净值数据")

    nav = info["nav"]
    nav_date = info["date"]
    # 用副本生成信号
    fund_copy = fund.copy()
    bond_yield = fetch_bond_yield() if is_spread else None
    signals = generate_signals(
        fund_copy, nav, nav_date, bond_yield, advance_signals=False
    )
    fund_copy["last_updated"] = nav_date

    # 同步行情状态回原 fund 并持久化，确保仪表盘与详情页一致
    sync_keys = (
        "last_trigger_level", "last_trigger_tier", "mode", "sell_signal_pending",
        "peak_price", "peak_date", "trough_price",
    )
    if is_spread:
        sync_keys += ("spread_signal", "current_spread", "current_div_yield", "current_bond_yield")
    for key in sync_keys:
        if key in fund_copy:
            fund[key] = fund_copy[key]
    fund["last_nav"] = nav
    fund["last_updated"] = nav_date
    save_state(state)

    return _fund_detail(fund_copy, nav, signals)


@app.post("/api/funds/{fund_code}")
def add_fund(fund_code: str, req: FundAddRequest = FundAddRequest()):
    """添加基金到跟踪列表"""
    _validate_fund_code(fund_code)
    state = load_state()

    if fund_code in state.get("funds", {}):
        raise HTTPException(status_code=409, detail="该基金已在跟踪列表中")

    is_spread = req.sell_rule == "spread"
    info = _fetch_price_for_fund(fund_code, req.sell_rule)
    if info is None:
        raise HTTPException(status_code=400, detail=f"无法获取基金 {fund_code} 的数据，请检查代码")

    fund = init_fund_state(
        fund_code=fund_code,
        name=info["name"],
        capital=req.capital,
        base_ratio=req.base_ratio,
        tier_step=req.tier_step,
        multiplier=req.multiplier,
        entry_threshold=req.entry,
        sell_rule=req.sell_rule,
        monthly_base=req.monthly_base or DEFAULT_SPREAD_MONTHLY_BASE,
        spread_double_threshold=req.spread_double_threshold or DEFAULT_SPREAD_DOUBLE_THRESHOLD,
        spread_clear_threshold=req.spread_clear_threshold or DEFAULT_SPREAD_CLEAR_THRESHOLD,
    )

    nav = info["nav"]
    fund["last_nav"] = nav

    # 自动分配排序位置
    max_order = max((f.get("order", 0) for f in state["funds"].values()), default=0)
    fund["order"] = max_order + 1

    # 用户填了现有持仓 → 直接初始化
    if req.shares and req.shares > 0:
        mv = req.market_value or (nav * req.shares)
        fund["total_shares"] = round(req.shares, 4)
        fund["total_invested"] = round(mv - (req.pnl or 0), 2)
        fund["last_nav"] = round(mv / req.shares, 4) if req.shares else nav

    # 从历史数据找 peak/trough + prev_nav（利差策略跳过）
    if not is_spread:
        long_history = fetch_nav_history(fund_code, days=365 * 3)
        if not long_history:
            long_history = fetch_etf_history(fund_code, days=365 * 3)
        if long_history:
            peak_entry = max(long_history, key=lambda x: x["nav"])
            fund["peak_price"] = peak_entry["nav"]
            fund["peak_date"] = peak_entry["date"]
            after_peak = [h for h in long_history if h["date"] >= peak_entry["date"]]
            if after_peak:
                trough_entry = min(after_peak, key=lambda x: x["nav"])
                if trough_entry["nav"] < peak_entry["nav"]:
                    fund["trough_price"] = trough_entry["nav"]
            if len(long_history) >= 2:
                fund["prev_nav"] = long_history[-2]["nav"]

    # 利差策略需要传入债券收益率
    bond_yield = fetch_bond_yield() if is_spread else None
    state["funds"][fund_code] = fund
    signals = generate_signals(fund, nav, info["date"], bond_yield)
    execute_signals(fund, signals, info["date"])
    save_state(state)

    return _fund_detail(fund, nav, signals)


@app.post("/api/funds/{fund_code}/refresh")
def refresh_fund(fund_code: str, execute: bool = Query(False)):
    """刷新基金净值并生成信号，可选执行买入"""
    import copy

    _validate_fund_code(fund_code)
    state = load_state()

    if fund_code not in state.get("funds", {}):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在跟踪列表中")

    fund = state["funds"][fund_code]
    sell_rule = fund.get("sell_rule")
    is_spread = sell_rule == "spread"

    info = _fetch_price_for_fund(fund_code, sell_rule)
    if info is None:
        raise HTTPException(status_code=502, detail=f"无法获取 {fund_code} 的净值数据")

    nav = info["nav"]
    nav_date = info["date"]

    # 在副本上生成信号，避免不执行买入时状态被意外推进
    fund_copy = copy.deepcopy(fund)
    bond_yield = fetch_bond_yield() if is_spread else None
    signals = generate_signals(
        fund_copy, nav, nav_date, bond_yield, advance_signals=execute
    )

    # 行情状态字段始终同步回原 fund，确保仪表盘数据准确
    sync_keys = (
        "last_trigger_level", "last_trigger_tier", "mode", "sell_signal_pending",
        "peak_price", "peak_date", "trough_price",
    )
    if is_spread:
        sync_keys += ("spread_signal", "current_spread", "current_div_yield", "current_bond_yield")
    for key in sync_keys:
        if key in fund_copy:
            fund[key] = fund_copy[key]

    if execute:
        execute_signals(fund, signals, nav_date)

    fund["last_nav"] = nav
    fund["last_updated"] = nav_date

    # 获取前一日净值，用于计算当日涨跌
    if is_spread:
        history = fetch_etf_history(fund_code, days=7)
    else:
        history = fetch_nav_history(fund_code, days=7)
        if not history:
            history = fetch_etf_history(fund_code, days=7)
    if len(history) >= 2:
        fund["prev_nav"] = history[-2]["nav"]

    state["funds"][fund_code] = fund
    save_state(state)

    return _fund_detail(fund, nav, signals)


@app.post("/api/funds/{fund_code}/consolidate")
def consolidate_positions(fund_code: str):
    """清除所有持仓记录，将当前持仓合并为一条记录"""
    _validate_fund_code(fund_code)
    state = load_state()

    if fund_code not in state.get("funds", {}):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在跟踪列表中")

    fund = state["funds"][fund_code]
    invested = fund.get("total_invested", 0)
    shares = fund.get("total_shares", 0)
    nav = fund.get("last_nav") or fund.get("peak_price") or 0

    if shares > 0 and nav > 0:
        fund["positions"] = [{
            "date": datetime.today().strftime("%Y-%m-%d"),
            "nav": round(nav, 4),
            "amount": round(invested, 2),
            "shares": round(shares, 4),
        }]
    else:
        fund["positions"] = []

    save_state(state)
    return {"message": "持仓已合并", "positions": fund["positions"]}


@app.post("/api/funds/{fund_code}/positions")
def sync_trade(fund_code: str, req: ManualTradeRequest):
    """同步加仓/减仓：正数加仓，负数减仓，按日期净值计算"""
    _validate_fund_code(fund_code)
    state = load_state()

    if fund_code not in state.get("funds", {}):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在跟踪列表中")

    fund = state["funds"][fund_code]
    nav = req.nav

    if nav is None:
        history = fetch_nav_history(fund_code, days=14)
        if not history:
            history = fetch_etf_history(fund_code, days=14)
        # 1. 精确匹配日期
        matching = [h for h in history if h["date"] == req.date]
        if matching:
            nav = matching[0]["nav"]
            nav_date = matching[0]["date"]
        else:
            # 2. 非交易日 → 找下一个交易日净值（如上周末买入按下周一算）
            future = [h for h in history if h["date"] > req.date]
            if future:
                nav = future[0]["nav"]
                nav_date = future[0]["date"]
            elif history:
                # 3. 没有未来数据 → 取最近一个
                nav = history[-1]["nav"]
                nav_date = history[-1]["date"]
            else:
                raise HTTPException(status_code=400, detail="无法获取净值，请手动填写")

    if nav <= 0:
        raise HTTPException(status_code=400, detail="净值必须大于0")

    shares = req.amount / nav
    new_invested = fund["total_invested"] + req.amount
    new_shares = fund["total_shares"] + shares

    if new_shares < 0:
        raise HTTPException(status_code=400, detail="减仓份额超过持仓份额")
    if new_invested < 0:
        new_invested = 0.0

    fund["positions"].append({
        "date": req.date,
        "nav": round(nav, 4),
        "amount": round(req.amount, 2),
        "shares": round(shares, 4),
    })
    fund["total_invested"] = round(new_invested, 2)
    fund["total_shares"] = round(new_shares, 4)

    info = fetch_latest_nav(fund_code)
    if info:
        fund["last_nav"] = info["nav"]
    save_state(state)
    current_nav = info["nav"] if info else (fund.get("peak_price") or nav)
    return _fund_detail(fund, current_nav, [])


@app.put("/api/funds/{fund_code}/holdings")
def edit_holdings(fund_code: str, req: EditHoldingsRequest):
    """直接修改持仓：填入投入本金和份额。"""
    _validate_fund_code(fund_code)
    state = load_state()

    if fund_code not in state.get("funds", {}):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在跟踪列表中")

    if req.total_shares < 0:
        raise HTTPException(status_code=400, detail="持仓份额不能为负数")

    fund = state["funds"][fund_code]

    if req.total_shares == 0:
        # 清空持仓
        fund["total_invested"] = 0.0
        fund["total_shares"] = 0.0
        fund["positions"] = []
        avg_cost = 0.0
    else:
        total_invested = round(req.total_invested, 2)
        fund["total_invested"] = total_invested
        fund["total_shares"] = round(req.total_shares, 4)
        avg_cost = total_invested / req.total_shares

    # 确保 last_nav 存在，让列表页能显示正确净值
    if not fund.get("last_nav"):
        info = fetch_latest_nav(fund_code)
        if info:
            fund["last_nav"] = info["nav"]

    invested = fund["total_invested"]
    shares = fund["total_shares"]
    save_state(state)
    return {"message": "持仓已更新" if shares > 0 else "持仓已清空",
            "total_invested": round(invested, 2),
            "total_shares": round(shares, 4),
            "avg_cost": round(avg_cost, 4)}

@app.delete("/api/funds/{fund_code}")
def remove_fund(fund_code: str):
    """从跟踪列表移除基金"""
    state = load_state()

    if fund_code not in state.get("funds", {}):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在跟踪列表中")

    name = state["funds"][fund_code]["name"]
    del state["funds"][fund_code]
    save_state(state)
    return {"message": f"已移除 {name}（{fund_code}）"}


@app.post("/api/funds/{fund_code}/move")
def move_fund(fund_code: str, direction: str = "up"):
    """调整基金在列表中的显示顺序（up=上移, down=下移）"""
    state = load_state()
    funds = state.get("funds", {})

    if fund_code not in funds:
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在跟踪列表中")

    # 按 order 排序
    sorted_codes = sorted(funds.keys(), key=lambda c: funds[c].get("order", 9999))
    idx = sorted_codes.index(fund_code)

    if direction == "up" and idx > 0:
        # 与上一个交换 order
        funds[fund_code]["order"], funds[sorted_codes[idx - 1]]["order"] = \
            funds[sorted_codes[idx - 1]]["order"], funds[fund_code]["order"]
    elif direction == "down" and idx < len(sorted_codes) - 1:
        funds[fund_code]["order"], funds[sorted_codes[idx + 1]]["order"] = \
            funds[sorted_codes[idx + 1]]["order"], funds[fund_code]["order"]

    save_state(state)
    return {"message": "顺序已更新"}


@app.put("/api/funds/{fund_code}")
def update_fund_config(fund_code: str, req: FundConfigRequest):
    """修改基金策略参数"""
    _validate_fund_code(fund_code)
    state = load_state()

    if fund_code not in state.get("funds", {}):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在跟踪列表中")

    fund = state["funds"][fund_code]
    changes = {}

    if req.capital is not None:
        fund["capital"] = req.capital
        changes["capital"] = req.capital
    if req.base_ratio is not None:
        fund["base_ratio"] = req.base_ratio
        changes["base_ratio"] = req.base_ratio
    if req.entry is not None:
        fund["entry_threshold"] = req.entry
        fund["last_trigger_level"] = None
        fund["last_trigger_tier"] = None
        changes["entry_threshold"] = req.entry
    if req.tier_step is not None:
        fund["tier_step"] = req.tier_step
        fund["last_trigger_level"] = None
        fund["last_trigger_tier"] = None
        changes["tier_step"] = req.tier_step
    if req.multiplier is not None:
        fund["multiplier"] = req.multiplier
        fund["last_trigger_level"] = None
        fund["last_trigger_tier"] = None
        changes["multiplier"] = req.multiplier
    if req.sell_rule is not None:
        fund["sell_rule"] = req.sell_rule
        changes["sell_rule"] = req.sell_rule
    if req.start_date is not None:
        fund["drawdown_start_date"] = req.start_date
        fund["peak_date"] = None
        changes["drawdown_start_date"] = req.start_date
    # 利差定投参数
    if req.monthly_base is not None:
        fund["monthly_base"] = req.monthly_base
        changes["monthly_base"] = req.monthly_base
    if req.spread_double_threshold is not None:
        fund["spread_double_threshold"] = req.spread_double_threshold
        changes["spread_double_threshold"] = req.spread_double_threshold
    if req.spread_clear_threshold is not None:
        fund["spread_clear_threshold"] = req.spread_clear_threshold
        changes["spread_clear_threshold"] = req.spread_clear_threshold
    if req.dividends is not None:
        fund["dividends"] = req.dividends
        changes["dividends"] = f"已更新（{len(req.dividends)}条）"
    save_state(state)
    return {"message": "配置已更新", "changes": changes}


@app.get("/api/funds/{fund_code}/history")
def get_fund_history(fund_code: str, days: int = 90):
    """获取基金近期净值历史，用于走势图"""
    _validate_fund_code(fund_code)
    state = load_state()
    fund = state.get("funds", {}).get(fund_code, {})
    if fund.get("sell_rule") == "spread":
        history = fetch_etf_history(fund_code, days=days)
    else:
        history = fetch_nav_history(fund_code, days=days)
        if not history:
            history = fetch_etf_history(fund_code, days=days)
    if not history:
        raise HTTPException(status_code=502, detail="无法获取历史数据")
    return {
        "fund_code": fund_code,
        "days": days,
        "data": [{"date": h["date"], "nav": h["nav"]} for h in history]
    }


@app.get("/api/funds/{fund_code}/signal-history")
def get_signal_history(fund_code: str, days: int = Query(30, ge=7, le=365)):
    """用当前策略配置回放历史净值，返回最近 N 天的交易信号"""
    _validate_fund_code(fund_code)
    state = load_state()

    if fund_code not in state.get("funds", {}):
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不在跟踪列表中")

    fund = state["funds"][fund_code]
    is_spread = fund.get("sell_rule") == "spread"

    # 获取历史数据：多取 60 天用于预热（让 peak 稳定），避免首日初始化信号出现在展示窗口
    history = fetch_etf_history(fund_code, days=days + 60) if is_spread else fetch_nav_history(fund_code, days=days + 60)
    if not history and not is_spread:
        history = fetch_etf_history(fund_code, days=days + 60)

    if not history:
        raise HTTPException(status_code=502, detail="无法获取历史数据")

    # 利差策略必须使用对应日期的历史国债收益率，拒绝用当前值回填历史。
    bond_yield = fetch_bond_yield_history(days + 90) if is_spread else None
    if is_spread and not bond_yield:
        raise HTTPException(status_code=502, detail="无法获取历史国债收益率，暂停利差历史回放")

    all_signals = replay_history(fund, history, bond_yield)

    # 仅返回最近 N 天
    cutoff = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent = [s for s in all_signals if s["date"] >= cutoff]

    return {
        "fund_code": fund_code,
        "days": days,
        "signals": recent,
    }


@app.post("/api/funds/{fund_code}/backtest")
def run_backtest(fund_code: str, req: BacktestRequest = BacktestRequest()):
    """历史回测：包含现金约束、费用、基准、回撤和样本外区间。"""
    state = load_state()
    fund_info = state.get("funds", {}).get(fund_code, {})
    is_spread = fund_info.get("sell_rule") == "spread"
    history = fetch_etf_history(fund_code, days=req.days) if is_spread else fetch_nav_history(fund_code, days=req.days)
    if not history:
        history = fetch_etf_history(fund_code, days=req.days)

    if not history:
        raise HTTPException(status_code=502, detail="无法获取历史数据")

    if fund_code in state.get("funds", {}):
        fund = json.loads(json.dumps(state["funds"][fund_code]))
    else:
        info = fetch_latest_nav(fund_code)
        name = info["name"] if info else fund_code
        fund = init_fund_state(
            fund_code, name, DEFAULT_CAPITAL,
            DEFAULT_BASE_RATIO, DEFAULT_TIER_STEP,
            DEFAULT_MULTIPLIER, DEFAULT_ENTRY_THRESHOLD, DEFAULT_SELL_RULE,
        )

    # 应用用户调整的参数（不保存到实际状态）
    if req.entry_threshold is not None:
        fund["entry_threshold"] = req.entry_threshold
    if req.tier_step is not None:
        fund["tier_step"] = req.tier_step
    if req.multiplier is not None:
        fund["multiplier"] = req.multiplier
    if req.base_ratio is not None:
        fund["base_ratio"] = req.base_ratio
    if req.capital is not None:
        fund["capital"] = req.capital
    if req.sell_rule is not None:
        fund["sell_rule"] = req.sell_rule
    bond_yields = fetch_bond_yield_history(req.days + 30) if is_spread else None
    if is_spread and not bond_yields:
        raise HTTPException(status_code=502, detail="无法获取历史国债收益率，利差回测已暂停")
    result = simulate_backtest(fund, history, req.buy_fee_rate, req.sell_fee_rate, bond_yields)
    split_index = max(1, int(len(history) * 0.7))
    out_of_sample_history = history[split_index:]
    out_of_sample = simulate_backtest(
        fund, out_of_sample_history, req.buy_fee_rate, req.sell_fee_rate, bond_yields
    ) if len(out_of_sample_history) >= 2 else {}

    # 小范围参数敏感性：避免只汇报一个历史最优点。
    sensitivity = []
    base_entry = float(fund.get("entry_threshold", 10))
    base_multiplier = float(fund.get("multiplier", 1.2))
    entry_values = [base_entry] if is_spread else sorted({max(0.5, base_entry - 2), base_entry, base_entry + 2})
    multiplier_values = [base_multiplier] if is_spread else sorted({max(1.0, base_multiplier - 0.2), base_multiplier, base_multiplier + 0.2})
    for entry_value in entry_values:
        for multiplier_value in multiplier_values:
            scenario = json.loads(json.dumps(fund))
            scenario["entry_threshold"] = entry_value
            scenario["multiplier"] = multiplier_value
            simulated = simulate_backtest(scenario, history, req.buy_fee_rate, req.sell_fee_rate, bond_yields)
            sensitivity.append({
                "entry_threshold": entry_value,
                "multiplier": round(multiplier_value, 2),
                "total_return": simulated.get("total_return", 0),
                "max_drawdown": simulated.get("max_drawdown", 0),
            })

    result.update({
        "fund_code": fund_code,
        "name": fund["name"],
        "params": {
            "entry_threshold": fund.get("entry_threshold", 10),
            "tier_step": fund.get("tier_step", 5),
            "multiplier": fund.get("multiplier", 1.2),
            "base_ratio": fund.get("base_ratio", 0.03),
            "capital": fund.get("capital", 100000),
            "sell_rule": fund.get("sell_rule", "momentum"),
            "buy_fee_rate": req.buy_fee_rate,
            "sell_fee_rate": req.sell_fee_rate,
        },
        "out_of_sample": out_of_sample,
        "sensitivity": sensitivity,
    })
    return result


# ── Static files (must be last) ────────────────────────────────────────

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
