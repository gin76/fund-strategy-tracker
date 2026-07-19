#!/usr/bin/env python3
"""
基金定投交易跟踪器 — 基于回撤比例的阶梯式定投策略

买入规则：
  回撤 < entry_threshold   不操作
  每跨越一个完整策略档位    仅触发一次买入
  每只标的只受自身资金上限和剩余资金限制

卖出规则：
  卖出点位 = 前期高点 + (前期高点 - 本轮最低点) / 2

数据来源：天天基金（fund.eastmoney.com）公开 API
"""

import copy
import json
import math
import os
import re
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import click
import requests

# ── 工作目录 & 状态文件 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "fund_state.json"

# ── 默认参数 ─────────────────────────────────────────────────────────
DEFAULT_CAPITAL = 100_000       # 默认总资金（元）
DEFAULT_BASE_RATIO = 0.03       # 基础买入比例（3%）
DEFAULT_TIER_STEP = 5           # 每档 5%
DEFAULT_MULTIPLIER = 1.2        # 每档乘数
DEFAULT_ENTRY_THRESHOLD = 10    # 回撤超过 10% 才开始买入
DEFAULT_SELL_RULE = "momentum"  # 卖出规则: "momentum" 或 "mean_revert"
# ── 利差定投策略参数 ────────────────────────────────────────────────
DEFAULT_SPREAD_MONTHLY_BASE = 5000      # 月定投基础金额
DEFAULT_SPREAD_DOUBLE_THRESHOLD = 3.0   # 双倍定投线（利差 > X%）
DEFAULT_SPREAD_CLEAR_THRESHOLD = 1.0    # 清仓线（利差 < X%）
SINA_PRICE_URL = "https://hq.sinajs.cn/list={market}{code}"
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn"}

# ── API 配置 ─────────────────────────────────────────────────────────
EASTMONEY_NAV_URL = "https://api.fund.eastmoney.com/f10/lsjz"
EASTMONEY_SEARCH_URL = "https://fundsuggest.eastmoney.com/FundSearch/api/FundSearchAPI.ashx"
API_HEADERS = {"Referer": "https://fundf10.eastmoney.com/"}


# ══════════════════════════════════════════════════════════════════════
# 数据获取（天天基金 API）
# ══════════════════════════════════════════════════════════════════════

def _api_get(url: str, params: dict, timeout: int = 15) -> dict | None:
    """带重试的 API 请求"""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=API_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(0.5)
    return None


def fetch_fund_name(fund_code: str) -> str | None:
    """通过天天基金搜索 API 获取基金名称"""
    try:
        data = _api_get(EASTMONEY_SEARCH_URL, {"m": "1", "key": fund_code})
        if data and data.get("Datas"):
            return str(data["Datas"][0]["NAME"])
    except Exception:
        pass
    return None


def fetch_latest_nav(fund_code: str) -> dict | None:
    """
    获取基金最新净值。
    返回 {"name": str, "nav": float, "date": str} 或 None。
    """
    name = fetch_fund_name(fund_code)
    today = datetime.today()
    end = today.strftime("%Y-%m-%d")
    start = (today - timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        data = _api_get(EASTMONEY_NAV_URL, {
            "fundCode": fund_code,
            "pageIndex": 1,
            "pageSize": 5,
            "startDate": start,
            "endDate": end,
        })
        items = data.get("Data", {}).get("LSJZList", [])
        if not items:
            return None

        latest = items[0]
        return {
            "name": name or fund_code,
            "nav": float(latest["DWJZ"]),
            "date": latest["FSRQ"],
        }
    except Exception:
        return None


def fetch_nav_history(fund_code: str, days: int = 365 * 5) -> list[dict]:
    """
    获取基金历史净值数据。
    返回按日期升序排列的 [{"date": str, "nav": float}, ...]。

    分页获取天天基金数据，最多取 days 个自然日范围。
    """
    today = datetime.today()
    end = today.strftime("%Y-%m-%d")
    start = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    all_items = []
    page = 1

    while True:
        try:
            data = _api_get(EASTMONEY_NAV_URL, {
                "fundCode": fund_code,
                "pageIndex": page,
                "pageSize": 20,
                "startDate": start,
                "endDate": end,
            })
        except Exception:
            break

        items = data.get("Data", {}).get("LSJZList", [])
        if not items:
            break

        for item in items:
            all_items.append({
                "date": item["FSRQ"],
                "nav": float(item["DWJZ"]),
            })

        total = data.get("TotalCount", 0)
        if page * 20 >= total:
            break
        page += 1
        time.sleep(0.15)  # 控制请求频率

    # 按日期升序返回
    all_items.sort(key=lambda x: x["date"])
    return all_items


def recalibrate_peak(fund_code: str, start_date: str) -> dict | None:
    """
    从 start_date 起拉取历史净值，重算该日期以来的前期高点和本轮最低点。

    返回 {"peak": float, "peak_date": str, "trough": float | None} 或 None。
    """
    days_needed = (datetime.today() - datetime.strptime(start_date, "%Y-%m-%d")).days + 10
    history = fetch_nav_history(fund_code, days=max(days_needed, 30))

    # 只保留 start_date 之后的数据
    filtered = [h for h in history if h["date"] >= start_date]
    if not filtered:
        return None

    # 前期高点 = start_date 以来的最高净值
    peak_entry = max(filtered, key=lambda x: x["nav"])
    peak = peak_entry["nav"]
    peak_date = peak_entry["date"]

    # 本轮最低点 = 高点日期之后的最低净值
    after_peak = [h for h in filtered if h["date"] >= peak_date]
    trough_entry = min(after_peak, key=lambda x: x["nav"])
    trough = trough_entry["nav"] if trough_entry["nav"] < peak else None

    return {"peak": peak, "peak_date": peak_date, "trough": trough}


def fetch_etf_price(fund_code: str) -> dict | None:
    """
    从新浪财经获取 ETF/股票实时价格。
    5xxxxx/6xxxxx = 上交所，0xxxxx/1xxxxx/3xxxxx = 深交所。
    返回 {"name": str, "nav": float, "date": str} 或 None。
    """
    first_digit = fund_code[0]
    if first_digit in ("5", "6"):
        market = "sh"
    elif first_digit in ("0", "1", "3"):
        market = "sz"
    else:
        return None

    url = SINA_PRICE_URL.format(market=market, code=fund_code)
    try:
        req = urllib.request.Request(url, headers=SINA_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        text = raw.decode("gb18030", errors="replace")
        match = re.search(r'"([^"]+)"', text)
        if not match:
            return None
        fields = match.group(1).split(",")
        if len(fields) < 32:
            return None
        price = float(fields[3])
        name = fields[0]
        return {"name": name, "nav": price, "date": datetime.today().strftime("%Y-%m-%d")}
    except Exception:
        return None


# ── 国债收益率缓存 ──────────────────────────────────────────────────
_bond_yield_cache = {"value": None, "ts": 0.0}


def fetch_bond_yield() -> float | None:
    """从 akshare 获取十年期国债收益率（1小时内存缓存）"""
    now = time.time()
    if now - _bond_yield_cache["ts"] < 3600 and _bond_yield_cache["value"] is not None:
        return _bond_yield_cache["value"]

    try:
        import akshare as ak
        df = ak.bond_zh_us_rate()
        if df.empty:
            return _bond_yield_cache["value"]
        latest = df.iloc[-1]
        val = latest["中国国债收益率10年"]
        result = float(val) if val and val == val else None
        if result is not None:
            _bond_yield_cache["value"] = result
            _bond_yield_cache["ts"] = now
        return result
    except Exception:
        return _bond_yield_cache["value"]


def fetch_bond_yield_history(days: int = 730) -> dict[str, float]:
    """获取中国十年期国债历史收益率，返回日期到收益率的映射。"""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate()
        if df.empty or "中国国债收益率10年" not in df.columns:
            return {}
        date_column = "日期" if "日期" in df.columns else df.columns[0]
        cutoff = datetime.today() - timedelta(days=days)
        result = {}
        for _, row in df.iterrows():
            try:
                date_value = row[date_column]
                parsed = date_value if isinstance(date_value, datetime) else datetime.fromisoformat(str(date_value)[:10])
                yield_value = float(row["中国国债收益率10年"])
            except (TypeError, ValueError):
                continue
            if parsed >= cutoff and yield_value == yield_value:
                result[parsed.strftime("%Y-%m-%d")] = yield_value
        return result
    except Exception:
        return {}


def fetch_etf_history(fund_code: str, days: int = 90) -> list[dict]:
    """从新浪财经获取 ETF 日 K 线历史数据"""
    first_digit = fund_code[0]
    if first_digit in ("5", "6"):
        symbol = f"sh{fund_code}"
    else:
        symbol = f"sz{fund_code}"

    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={min(days, 500)}"
    )
    try:
        req = urllib.request.Request(url, headers=SINA_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        text = raw.decode("gb18030", errors="replace")
        data = json.loads(text)
        return [{"date": d["day"], "nav": float(d["close"])} for d in data if d.get("close")]
    except Exception:
        return []
# ══════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    """加载所有基金的跟踪状态"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return ensure_state_defaults(json.load(f))
        except (json.JSONDecodeError, IOError):
            # 状态文件损坏，备份后重置
            backup = STATE_FILE.with_suffix(".json.bak")
            STATE_FILE.rename(backup)
            return ensure_state_defaults({"funds": {}})
    return ensure_state_defaults({"funds": {}})


def save_state(state: dict):
    """保存状态到文件"""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def infer_asset_bucket(fund: dict) -> str:
    """根据基金代码和名称推断组合风险桶；用户可在配置中覆盖。"""
    code = str(fund.get("fund_code", ""))
    name = str(fund.get("name", ""))
    if code in {"021000", "016452", "016453", "018043"} or "纳斯达克100" in name:
        return "nasdaq100"
    if code in {"008163", "008114", "159307"} or "红利低波" in name:
        return "dividend_low_vol"
    if "债" in name:
        return "bond"
    if "黄金" in name:
        return "gold"
    if "碳中和" in name:
        return "thematic"
    return "other"


def ensure_state_defaults(state: dict) -> dict:
    """为旧状态补齐单标的策略字段，保持向后兼容。"""
    funds = state.setdefault("funds", {})
    for fund in funds.values():
        fund["allow_new_buys"] = True
        fund.setdefault("last_trigger_tier", None)
        fund.setdefault("last_spread_invest_month", None)

        # 从旧的整数回撤点迁移到档位，避免升级后重复发出已执行信号。
        if fund.get("last_trigger_tier") is None and fund.get("last_trigger_level") is not None:
            entry = float(fund.get("entry_threshold", DEFAULT_ENTRY_THRESHOLD))
            step = max(float(fund.get("tier_step", DEFAULT_TIER_STEP)), 0.0001)
            fund["last_trigger_tier"] = max(0, int((float(fund["last_trigger_level"]) - entry) / step))

    state.pop("portfolio", None)
    return state


def init_fund_state(
    fund_code: str,
    name: str,
    capital: float,
    base_ratio: float,
    tier_step: float,
    multiplier: float,
    entry_threshold: float,
    sell_rule: str = "momentum",
    monthly_base: float = DEFAULT_SPREAD_MONTHLY_BASE,
    dividends: list | None = None,
    spread_double_threshold: float = DEFAULT_SPREAD_DOUBLE_THRESHOLD,
    spread_clear_threshold: float = DEFAULT_SPREAD_CLEAR_THRESHOLD,
    asset_bucket: str | None = None,
    allow_new_buys: bool = True,
) -> dict:
    """为一只新基金创建初始状态"""
    state = {
        "fund_code": fund_code,
        "name": name,
        "capital": capital,
        "base_ratio": base_ratio,
        "tier_step": tier_step,
        "multiplier": multiplier,
        "entry_threshold": entry_threshold,
        "sell_rule": sell_rule,
        # 行情状态
        "peak_price": None,          # 前期高点
        "peak_date": None,           # 前期高点对应的净值日期
        "trough_price": None,        # 本轮最低点
        "last_trigger_level": None,  # 上一次触发买入的回撤整数点位
        "last_trigger_tier": None,   # 上一次实际执行的策略档位
        "mode": "waiting",           # waiting | buying
        "drawdown_start_date": None, # 回撤计算起始日期（None = 从添加日开始）
        # 持仓
        "positions": [],             # [{date, nav, amount, shares}, ...]
        "total_invested": 0.0,
        "total_shares": 0.0,
        # 信号去重
        "sell_signal_pending": False,
        "asset_bucket": asset_bucket,
        "allow_new_buys": allow_new_buys,
        # 历史记录
        "last_updated": None,
    }

    if sell_rule == "spread":
        state["monthly_base"] = monthly_base
        state["spread_double_threshold"] = spread_double_threshold
        state["spread_clear_threshold"] = spread_clear_threshold
        state["dividends"] = dividends or []
        state["last_spread_invest_month"] = None

    if not state["asset_bucket"]:
        state["asset_bucket"] = infer_asset_bucket(state)

    return state


# ══════════════════════════════════════════════════════════════════════
# 策略计算
# ══════════════════════════════════════════════════════════════════════

def calc_drawdown(current_nav: float, peak: float) -> float:
    """计算当前回撤百分比 (0-100)"""
    return (peak - current_nav) / peak * 100


def calc_ttm_dividend_yield(price: float, dividends: list, as_of_date: str = "") -> float:
    """TTM股息率 = 截至净值日向前滚动 365 天内的现金分红 / 当前价格。"""
    if not dividends or price <= 0:
        return 0.0
    try:
        end = datetime.strptime(as_of_date, "%Y-%m-%d") if as_of_date else datetime.today()
    except ValueError:
        end = datetime.today()
    start = end - timedelta(days=365)
    recent = []
    for item in dividends:
        try:
            div_date = datetime.strptime(str(item[0]), "%Y-%m-%d")
            amount = float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if start < div_date <= end:
            recent.append(amount)
    total = sum(recent)
    return total / price * 100


def generate_spread_signals(
    fund: dict,
    current_price: float,
    bond_yield: float | None,
    nav_date: str = "",
) -> list[dict]:
    """
    利差策略信号生成。
    spread > double_threshold → 双倍定投
    spread 1.5~double_threshold → 正常定投
    spread < clear_threshold → 清仓
    """
    signals = []
    monthly_base = fund.get("monthly_base", DEFAULT_SPREAD_MONTHLY_BASE)
    double_threshold = fund.get("spread_double_threshold", DEFAULT_SPREAD_DOUBLE_THRESHOLD)
    clear_threshold = fund.get("spread_clear_threshold", DEFAULT_SPREAD_CLEAR_THRESHOLD)
    dividends = fund.get("dividends", [])

    if bond_yield is None:
        fund["spread_signal"] = "error"
        fund["current_spread"] = None
        fund["current_div_yield"] = None
        fund["current_bond_yield"] = None
        signals.append({
            "type": "info",
            "nav": current_price,
            "amount": 0,
            "reason": "无法获取国债收益率数据，信号计算暂停",
        })
        return signals

    div_yield = calc_ttm_dividend_yield(current_price, dividends, nav_date)
    spread = div_yield - bond_yield

    fund["current_spread"] = round(spread, 2)
    fund["current_div_yield"] = round(div_yield, 2)
    fund["current_bond_yield"] = round(bond_yield, 4)

    date_prefix = f"{nav_date} " if nav_date else ""
    note = ""

    if spread > double_threshold:
        fund["spread_signal"] = "double"
        multiplier = 2.0
        action = "双倍定投"
    elif spread < clear_threshold:
        fund["spread_signal"] = "clear"
        multiplier = 0.0
        action = "清仓"
    else:
        fund["spread_signal"] = "normal"
        multiplier = 1.0
        action = "正常定投"

    requested_amount = monthly_base * multiplier
    shares = int(requested_amount / current_price / 100) * 100 if current_price > 0 and multiplier > 0 else 0
    amount = round(shares * current_price, 2)
    signal_month = nav_date[:7] if nav_date else datetime.today().strftime("%Y-%m")

    if abs(spread - double_threshold) < 0.1:
        note = "⚠️ 利差贴近双倍线，国债利率波动可能导致信号变化。"
    if abs(spread - clear_threshold) < 0.1:
        note = "⚠️ 利差贴近清仓线，密切关注。"

    if fund["spread_signal"] == "clear":
        signals.append({
            "type": "sell",
            "nav": current_price,
            "amount": round(current_price * fund.get("total_shares", 0), 2),
            "reason": (
                f"{date_prefix}利差 {spread:.2f}%（股息率 {div_yield:.2f}% - 国债 {bond_yield:.2f}%），"
                f"触及清仓线（{clear_threshold}%），建议全部卖出"
            ),
        })
    elif fund.get("last_spread_invest_month") == signal_month:
        signals.append({
            "type": "info",
            "nav": current_price,
            "amount": 0,
            "reason": f"{signal_month} 已完成利差定投，本月不重复下单",
        })
    elif shares <= 0:
        signals.append({
            "type": "info",
            "nav": current_price,
            "amount": 0,
            "reason": "本月定投金额不足一手（100份），未生成买入订单",
        })
    elif fund["spread_signal"] == "double":
        signals.append({
            "type": "buy",
            "nav": current_price,
            "amount": round(amount, 2),
            "shares": shares,
            "strategy": "spread",
            "signal_period": signal_month,
            "reason": (
                f"{date_prefix}利差 {spread:.2f}%（股息率 {div_yield:.2f}% - 国债 {bond_yield:.2f}%），"
                f"突破双倍线（{double_threshold}%），{action}，买入 {amount:.0f} 元（{shares}份）"
            ),
        })
    else:
        signals.append({
            "type": "buy",
            "nav": current_price,
            "amount": round(amount, 2),
            "shares": shares,
            "strategy": "spread",
            "signal_period": signal_month,
            "reason": (
                f"{date_prefix}利差 {spread:.2f}%（股息率 {div_yield:.2f}% - 国债 {bond_yield:.2f}%），"
                f"正常定投区间，买入 {amount:.0f} 元（{shares}份）"
            ),
        })

    if note:
        signals.append({"type": "info", "nav": current_price, "amount": 0, "reason": note})

    return signals


def calc_sell_point(peak: float, trough: float, sell_rule: str = "momentum") -> float | None:
    """
    计算卖出点位。

    sell_rule:
      "momentum"    — 趋势型：peak + (peak - trough) / 2（适合纳斯达克等高波动品种）
      "mean_revert" — 均值回归型：peak（适合红利低波等低波动品种）
      "none"        — 长期持有，不计算卖出点
    """
    if sell_rule in ("none", "spread"):
        return None
    if sell_rule == "mean_revert":
        return peak
    return peak + (peak - trough) / 2


def get_tier_multiplier(drawdown: float, entry_threshold: float,
                        tier_step: float, multiplier: float) -> float:
    """
    根据当前回撤计算买入档位乘数。

    档位计算：
      tier 0: entry_threshold ~ entry_threshold + tier_step     → × 1.0
      tier 1: entry_threshold + tier_step ~ entry_threshold + 2×tier_step → × multiplier
      tier 2: ...                                                → × multiplier²
    """
    if drawdown < entry_threshold:
        return 0.0
    tier = int((drawdown - entry_threshold) / tier_step)
    return multiplier ** tier


def generate_signals(fund: dict, current_nav: float, nav_date: str = "",
                     bond_yield: float | None = None,
                     advance_signals: bool = True) -> list[dict]:
    """
    根据当前净值和基金状态生成交易信号。

    advance_signals=False 用于页面预览：仍更新高点/低点等行情状态，
    但不标记买卖信号为已处理，因此未执行的信号不会被“吞掉”。

    返回信号列表，每个信号为:
      {"type": "buy"|"sell"|"info", "nav": float, "amount": float, "reason": str}
    """
    signals = []
    peak = fund["peak_price"]
    trough = fund["trough_price"]

    # ── 利差定投策略 ──────────────────────────────────────────────
    if fund.get("sell_rule") == "spread":
        if bond_yield is None:
            bond_yield = fetch_bond_yield()
        # 首次运行：初始化 peak 用于显示
        if fund["peak_price"] is None:
            today_str = datetime.today().strftime("%Y-%m-%d")
            fund["peak_price"] = current_nav
            fund["peak_date"] = today_str
            fund["last_updated"] = today_str
        return generate_spread_signals(fund, current_nav, bond_yield, nav_date)

    # ── 长期持有策略：不生成任何买卖信号 ────────────────────────────
    if fund.get("sell_rule") == "none":
        if peak is None:
            today_str = datetime.today().strftime("%Y-%m-%d")
            fund["peak_price"] = current_nav
            fund["peak_date"] = today_str
            fund["last_updated"] = today_str
        fund["mode"] = "waiting"
        return signals

    capital = fund["capital"]
    base_ratio = fund["base_ratio"]
    entry_threshold = fund["entry_threshold"]
    tier_step = fund["tier_step"]
    mult = fund["multiplier"]

    # ── 首次运行：初始化 peak ──────────────────────────────────────
    if peak is None:
        today_str = datetime.today().strftime("%Y-%m-%d")
        fund["peak_price"] = current_nav
        fund["peak_date"] = today_str
        fund["last_updated"] = today_str
        signals.append({
            "type": "info",
            "nav": current_nav,
            "amount": 0,
            "reason": f"初始化 — 当前净值 {current_nav:.4f} 设为前期高点",
        })
        return signals

    drawdown = round(calc_drawdown(current_nav, peak), 6)

    # ── 更新 trough（本轮最低点，仅在低于前期高点时更新）─────────
    if current_nav < peak and (trough is None or current_nav < trough):
        fund["trough_price"] = current_nav

    # ── 卖出信号（必须在创新高判断之前，因为卖出点 > peak）───────
    # 卖点基于当前 peak 计算，当日提醒一次后自动推进，不再重复
    current_trough = fund["trough_price"]
    sell_rule = fund.get("sell_rule", "momentum")
    if current_trough is not None and fund["total_shares"] > 0 and not fund["sell_signal_pending"]:
        sell_pt = calc_sell_point(peak, current_trough, sell_rule)
        if sell_pt is not None and current_nav >= sell_pt:
            avg_cost = fund["total_invested"] / fund["total_shares"] if fund["total_shares"] > 0 else 0
            profit = (current_nav - avg_cost) * fund["total_shares"]
            if advance_signals:
                fund["sell_signal_pending"] = True
            # 同步更新 peak 到最新净值（如果已创新高），确保回撤显示正确
            if current_nav > peak:
                today_str = datetime.today().strftime("%Y-%m-%d")
                fund["peak_price"] = current_nav
                fund["peak_date"] = today_str
            date_prefix = f"{nav_date} " if nav_date else ""
            signals.append({
                "type": "sell",
                "nav": current_nav,
                "amount": round(current_nav * fund["total_shares"], 2),
                "reason": (
                    f"{date_prefix}触发卖出! 卖出点={sell_pt:.4f}，当前净值={current_nav:.4f}，"
                    f"预估盈利 {profit:.2f} 元"
                ),
            })
            return signals

    # ── 价格回落，清除卖出标记（允许重新触发）─────────────────────
    if sell_rule != "none" and current_trough is not None:
        sell_pt = calc_sell_point(peak, current_trough, sell_rule)
        if sell_pt is not None and current_nav < sell_pt:
            fund["sell_signal_pending"] = False

    # ── 更新 peak（创新高）─────────────────────────────────────────
    # 保留 trough 确保卖出点随新高上移；重置 last_trigger_level 因为
    # 回撤百分比在 peak 变化后需要重新映射。同时清除卖出标记，新高=新周期
    if current_nav > peak:
        old_peak = peak
        today_str = datetime.today().strftime("%Y-%m-%d")
        fund["peak_price"] = current_nav
        fund["peak_date"] = today_str
        fund["last_trigger_level"] = None
        fund["last_trigger_tier"] = None
        fund["sell_signal_pending"] = False
        fund["mode"] = "waiting"
        signals.append({
            "type": "info",
            "nav": current_nav,
            "amount": 0,
            "reason": f"创新高! {old_peak:.4f} → {current_nav:.4f}",
        })
        return signals

    # ── 回撤不足阈值：不出手 ──────────────────────────────────────
    if drawdown < entry_threshold:
        fund["mode"] = "waiting"
        return signals

    # ── 回撤达标，进入买入模式 ──────────────────────────────────────
    fund["mode"] = "buying"

    current_tier = int((drawdown - entry_threshold) / tier_step)
    last_tier = fund.get("last_trigger_tier")
    last_tier = -1 if last_tier is None else int(last_tier)
    remaining_fund_budget = max(0.0, capital - float(fund.get("total_invested", 0) or 0))

    # 每个完整策略档位只触发一次；跳空会补齐跨过的档位，但总额受剩余预算限制。
    if current_tier > last_tier:
        for tier_idx in range(last_tier + 1, current_tier + 1):
            tier_mult = mult ** tier_idx
            buy_amount = capital * base_ratio * tier_mult
            buy_amount = min(buy_amount, remaining_fund_budget)
            if buy_amount <= 0.01:
                break
            trigger_drawdown = entry_threshold + tier_idx * tier_step
            date_prefix = f"{nav_date} " if nav_date else ""
            signals.append({
                "type": "buy",
                "nav": current_nav,
                "amount": round(buy_amount, 2),
                "tier_index": tier_idx,
                "reason": (
                    f"{date_prefix}回撤达到 {trigger_drawdown:g}%（第{tier_idx + 1}档，乘数×{tier_mult:.2f}），"
                    f"买入 {buy_amount:.0f} 元"
                ),
            })
            remaining_fund_budget -= buy_amount

    return signals


def execute_signals(fund: dict, signals: list[dict], nav_date: str):
    """根据信号更新持仓状态"""
    executed = []
    for sig in signals:
        if sig["type"] == "buy":
            nav = sig["nav"]
            remaining = max(0.0, float(fund.get("capital", 0)) - float(fund.get("total_invested", 0)))
            amount = min(float(sig["amount"]), remaining)
            if amount <= 0.01 or nav <= 0:
                continue
            if sig.get("strategy") == "spread":
                shares = int(amount / nav / 100) * 100
                amount = round(shares * nav, 2)
            else:
                shares = amount / nav
            if amount <= 0.01 or shares <= 0:
                continue
            fund["positions"].append({
                "date": nav_date,
                "nav": round(nav, 4),
                "amount": round(amount, 2),
                "shares": round(shares, 4),
            })
            fund["total_invested"] += amount
            fund["total_shares"] += shares
            if "tier_index" in sig:
                fund["last_trigger_tier"] = max(
                    int(fund.get("last_trigger_tier") if fund.get("last_trigger_tier") is not None else -1),
                    int(sig["tier_index"]),
                )
                fund["last_trigger_level"] = int(
                    fund["entry_threshold"] + int(sig["tier_index"]) * fund["tier_step"]
                )
            if sig.get("strategy") == "spread":
                fund["last_spread_invest_month"] = sig.get("signal_period") or nav_date[:7]
            executed.append({**sig, "amount": round(amount, 2), "shares": round(shares, 4)})

        elif sig["type"] == "sell":
            # 清仓
            fund["positions"] = []
            fund["total_invested"] = 0.0
            fund["total_shares"] = 0.0
            # 卖出后：以当前价格作为新的前期高点，重置本轮最低点
            fund["peak_price"] = sig["nav"]
            fund["peak_date"] = nav_date
            fund["trough_price"] = None
            fund["last_trigger_level"] = None
            fund["last_trigger_tier"] = None
            fund["mode"] = "waiting"
            fund["sell_signal_pending"] = False
            executed.append(sig)

    fund["last_updated"] = nav_date
    return executed


def simulate_backtest(
    fund_config: dict,
    history: list[dict],
    buy_fee_rate: float = 0.001,
    sell_fee_rate: float = 0.001,
    bond_yields: dict[str, float] | None = None,
) -> dict:
    """带现金约束、费用、基准和风险指标的单基金历史模拟。"""
    if not history:
        return {}
    fund = copy.deepcopy(fund_config)
    fund.update({
        "positions": [], "total_invested": 0.0, "total_shares": 0.0,
        "peak_price": None, "peak_date": None, "trough_price": None,
        "last_trigger_level": None, "last_trigger_tier": None,
        "mode": "waiting", "sell_signal_pending": False,
        "last_updated": None, "last_spread_invest_month": None,
        "allow_new_buys": True,
    })
    initial_cash = max(float(fund.get("capital", DEFAULT_CAPITAL)), 0.0)
    cash = initial_cash
    total_fees = 0.0
    total_buy_amount = 0.0
    total_buys = 0
    total_sells = 0
    realized_pnl = 0.0
    events = []
    equity_curve = []

    for entry in history:
        nav = float(entry["nav"])
        nav_date = entry.get("date", "")
        bond_yield = None
        if bond_yields:
            eligible_dates = [date for date in bond_yields if date <= nav_date]
            if eligible_dates:
                bond_yield = bond_yields[max(eligible_dates)]
        signals = generate_signals(fund, nav, nav_date, bond_yield)
        executable = []
        for signal in signals:
            if signal["type"] == "buy":
                max_amount = cash / (1 + buy_fee_rate) if buy_fee_rate >= 0 else cash
                amount = min(float(signal["amount"]), max_amount)
                if amount <= 0.01:
                    continue
                fee = amount * buy_fee_rate
                cash -= amount + fee
                total_fees += fee
                total_buy_amount += amount
                total_buys += 1
                clipped = {**signal, "amount": round(amount, 2)}
                executable.append(clipped)
                events.append({
                    "date": nav_date, "type": "buy", "nav": nav,
                    "amount": round(amount, 2), "fee": round(fee, 2),
                    "reason": signal["reason"],
                })
            elif signal["type"] == "sell" and fund.get("total_shares", 0) > 0:
                proceeds = nav * fund["total_shares"]
                fee = proceeds * sell_fee_rate
                avg_cost = fund["total_invested"] / fund["total_shares"]
                profit = (nav - avg_cost) * fund["total_shares"] - fee
                cash += proceeds - fee
                total_fees += fee
                realized_pnl += profit
                total_sells += 1
                executable.append(signal)
                events.append({
                    "date": nav_date, "type": "sell", "nav": nav,
                    "amount": round(proceeds, 2), "fee": round(fee, 2),
                    "pnl": round(profit, 2), "reason": signal["reason"],
                })
        execute_signals(fund, executable, nav_date)
        equity_curve.append({
            "date": nav_date,
            "equity": cash + nav * fund.get("total_shares", 0),
            "cash": cash,
        })

    final_nav = float(history[-1]["nav"])
    final_market_value = final_nav * fund.get("total_shares", 0)
    final_equity = cash + final_market_value
    total_return = (final_equity / initial_cash - 1) if initial_cash > 0 else 0.0

    peak_equity = 0.0
    max_drawdown = 0.0
    daily_returns = []
    previous_equity = None
    for point in equity_curve:
        equity = point["equity"]
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
        if previous_equity and previous_equity > 0:
            daily_returns.append(equity / previous_equity - 1)
        previous_equity = equity
    if daily_returns:
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((value - mean_return) ** 2 for value in daily_returns) / len(daily_returns)
        annual_volatility = math.sqrt(variance) * math.sqrt(252)
    else:
        annual_volatility = 0.0

    start = datetime.strptime(history[0]["date"], "%Y-%m-%d")
    end = datetime.strptime(history[-1]["date"], "%Y-%m-%d")
    years = max((end - start).days / 365.25, 1 / 365.25)
    cagr = (final_equity / initial_cash) ** (1 / years) - 1 if initial_cash > 0 and final_equity > 0 else -1.0

    # 同样初始资金的买入并持有基准。
    first_nav = float(history[0]["nav"])
    benchmark_amount = initial_cash / (1 + buy_fee_rate)
    benchmark_shares = benchmark_amount / first_nav if first_nav > 0 else 0
    benchmark_final = benchmark_shares * final_nav * (1 - sell_fee_rate)
    benchmark_return = benchmark_final / initial_cash - 1 if initial_cash > 0 else 0

    # 每月首个交易日等额定投基准，闲置现金按0收益处理。
    monthly_entries = []
    seen_months = set()
    for entry in history:
        month = entry["date"][:7]
        if month not in seen_months:
            seen_months.add(month)
            monthly_entries.append(entry)
    monthly_cash = initial_cash
    monthly_shares = 0.0
    monthly_budget = initial_cash / len(monthly_entries) if monthly_entries else 0
    for entry in monthly_entries:
        amount = min(monthly_budget / (1 + buy_fee_rate), monthly_cash / (1 + buy_fee_rate))
        monthly_cash -= amount * (1 + buy_fee_rate)
        monthly_shares += amount / float(entry["nav"])
    monthly_final = monthly_cash + monthly_shares * final_nav * (1 - sell_fee_rate)
    monthly_return = monthly_final / initial_cash - 1 if initial_cash > 0 else 0

    return {
        "start_date": history[0]["date"], "end_date": history[-1]["date"],
        "trading_days": len(history), "initial_cash": round(initial_cash, 2),
        "final_cash": round(cash, 2), "final_market_value": round(final_market_value, 2),
        "final_equity": round(final_equity, 2), "total_return": round(total_return, 6),
        "cagr": round(cagr, 6), "max_drawdown": round(max_drawdown, 6),
        "annual_volatility": round(annual_volatility, 6), "total_fees": round(total_fees, 2),
        "total_buys": total_buys, "total_sells": total_sells,
        "total_buy_amount": round(total_buy_amount, 2), "realized_pnl": round(realized_pnl, 2),
        "benchmark_buy_hold_return": round(benchmark_return, 6),
        "benchmark_monthly_dca_return": round(monthly_return, 6),
        "events": events, "equity_curve": equity_curve,
    }


def replay_history(fund_config: dict, history: list[dict],
                   bond_yield: float | dict[str, float] | None = None) -> list[dict]:
    """
    用当前策略配置回放历史净值，收集每一天会产生的交易信号。

    不修改原 fund_config——在深拷贝上运行，模拟完整的 signal → execute 流程。
    返回 [{date, type, nav, amount, reason}, ...]，按日期升序。
    """
    fund = copy.deepcopy(fund_config)

    # 重置动态行情状态，保留策略配置参数
    fund["positions"] = []
    fund["total_invested"] = 0.0
    fund["total_shares"] = 0.0
    fund["peak_price"] = None
    fund["peak_date"] = None
    fund["trough_price"] = None
    fund["last_trigger_level"] = None
    fund["last_trigger_tier"] = None
    fund["mode"] = "waiting"
    fund["sell_signal_pending"] = False
    fund["last_updated"] = None
    fund["last_spread_invest_month"] = None

    all_signals = []

    for entry in history:
        entry_date = entry.get("date", "")
        entry_bond_yield = bond_yield
        if isinstance(bond_yield, dict):
            eligible_dates = [date for date in bond_yield if date <= entry_date]
            entry_bond_yield = bond_yield[max(eligible_dates)] if eligible_dates else None
        signals = generate_signals(fund, entry["nav"], entry_date, entry_bond_yield)
        for sig in signals:
            all_signals.append({
                "date": entry["date"],
                "type": sig["type"],
                "nav": sig["nav"],
                "amount": sig["amount"],
                "reason": sig["reason"],
            })
        execute_signals(fund, signals, entry["date"])

    return all_signals


# ══════════════════════════════════════════════════════════════════════
# 展示
# ══════════════════════════════════════════════════════════════════════

def print_status(fund: dict, current_nav: float, signals: list[dict]):
    """打印基金当前状态"""
    peak = fund["peak_price"] or current_nav
    trough = fund["trough_price"]
    drawdown = calc_drawdown(current_nav, peak) if peak else 0

    click.echo()
    click.secho("━" * 58, fg="bright_black")
    click.secho(f"  {fund['name']}（{fund['fund_code']}）", fg="cyan", bold=True)
    click.secho("━" * 58, fg="bright_black")

    # 基本信息
    click.secho(f"  最新净值:  {current_nav:.4f}", fg="white")
    click.secho(f"  前期高点:  {peak:.4f}", fg="yellow")
    if trough:
        click.secho(f"  本轮最低:  {trough:.4f}", fg="red")
    click.secho(f"  当前回撤:  {drawdown:.2f}%",
                fg="red" if drawdown >= fund["entry_threshold"] else "green")

    # 卖出参考线
    if trough and fund["total_shares"] > 0:
        sell_pt = calc_sell_point(peak, trough, fund.get("sell_rule", "momentum"))
        pct_to_sell = (sell_pt - current_nav) / current_nav * 100
        click.secho(f"  卖出点位:  {sell_pt:.4f}（还需涨 {pct_to_sell:.2f}%）", fg="magenta")

    # 持仓
    click.secho(f"  持仓份额:  {fund['total_shares']:.2f}", fg="white")
    click.secho(f"  投入总额:  ¥{fund['total_invested']:,.2f}", fg="white")
    if fund["total_shares"] > 0:
        avg_cost = fund["total_invested"] / fund["total_shares"]
        market_value = current_nav * fund["total_shares"]
        pnl = market_value - fund["total_invested"]
        pnl_pct = (current_nav - avg_cost) / avg_cost * 100
        color = "green" if pnl >= 0 else "red"
        click.secho(f"  持仓市值:  ¥{market_value:,.2f}  |  浮动盈亏: {pnl:+,.2f} ({pnl_pct:+.2f}%)",
                    fg=color)

    # 下一档买入提示
    if drawdown >= fund["entry_threshold"]:
        next_level = (fund["last_trigger_level"] or int(fund["entry_threshold"])) + 1
        tier_mult = get_tier_multiplier(next_level, fund["entry_threshold"],
                                        fund["tier_step"], fund["multiplier"])
        trigger_nav = peak * (1 - next_level / 100)
        tier_idx = int((next_level - fund["entry_threshold"]) / fund["tier_step"])
        click.secho(
            f"  下一买入触发: 净值 ≤ {trigger_nav:.4f}（回撤 {next_level}%，"
            f"第{tier_idx + 1}档 ×{tier_mult:.2f}，买入 ¥{fund['capital'] * fund['base_ratio'] * tier_mult:,.0f}）",
            fg="blue"
        )

    # 交易信号
    buy_sigs = [s for s in signals if s["type"] == "buy"]
    sell_sigs = [s for s in signals if s["type"] == "sell"]
    info_sigs = [s for s in signals if s["type"] == "info"]

    for s in info_sigs:
        click.secho(f"  ℹ {s['reason']}", fg="bright_black")

    if buy_sigs:
        click.secho(f"\n  📈 买入信号（{len(buy_sigs)} 条）:", fg="green", bold=True)
        for s in buy_sigs:
            click.secho(f"     {s['reason']}", fg="green")

    if sell_sigs:
        click.secho(f"\n  📉 卖出信号:", fg="magenta", bold=True)
        for s in sell_sigs:
            click.secho(f"     {s['reason']}", fg="magenta")

    if not buy_sigs and not sell_sigs and not info_sigs:
        if drawdown < fund["entry_threshold"]:
            click.secho(f"  状态: 观望中（回撤 {drawdown:.1f}% < {fund['entry_threshold']}% 阈值）",
                        fg="bright_black")
        else:
            click.secho(f"  状态: 持有中，无新触发（当前回撤 {drawdown:.1f}%）",
                        fg="bright_black")

    click.secho("━" * 58, fg="bright_black")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

@click.group()
def cli():
    """基金定投交易跟踪器 — 基于回撤比例的阶梯式定投策略

    数据来源：天天基金（fund.eastmoney.com）

    示例：
      python fund_tracker.py add 270042 -c 100000 -r 0.03
      python fund_tracker.py check
      python fund_tracker.py backtest 270042 -d 365
    """


@cli.command()
@click.argument("fund_code")
@click.option("--capital", "-c", type=float, default=DEFAULT_CAPITAL,
              help=f"总资金（元），默认 {DEFAULT_CAPITAL:,}")
@click.option("--base-ratio", "-r", type=float, default=DEFAULT_BASE_RATIO,
              help=f"基础买入比例，默认 {DEFAULT_BASE_RATIO*100}%")
@click.option("--tier-step", "-s", type=float, default=DEFAULT_TIER_STEP,
              help=f"每档回撤幅度（%），默认 {DEFAULT_TIER_STEP}%")
@click.option("--multiplier", "-m", type=float, default=DEFAULT_MULTIPLIER,
              help=f"档位乘数，默认 {DEFAULT_MULTIPLIER}")
@click.option("--entry", "-e", type=float, default=DEFAULT_ENTRY_THRESHOLD,
              help=f"入场回撤阈值（%），默认 {DEFAULT_ENTRY_THRESHOLD}%")
@click.option("--sell-rule", type=click.Choice(["momentum", "mean_revert", "none", "spread"]),
              default=DEFAULT_SELL_RULE,
              help=f"卖出规则: momentum=趋势型, mean_revert=均值回归型（默认 {DEFAULT_SELL_RULE}）")
def add(fund_code, capital, base_ratio, tier_step, multiplier, entry, sell_rule):
    """添加一只基金到跟踪列表

    FUND_CODE: 基金代码（6位数字），如 270042 为广发纳斯达克100ETF联接
    """
    state = load_state()

    if fund_code in state["funds"]:
        click.secho(f"基金 {fund_code} 已在跟踪列表中", fg="yellow")
        return

    click.echo(f"正在获取 {fund_code} 的基金信息...")
    info = fetch_etf_price(fund_code) if sell_rule == "spread" else fetch_latest_nav(fund_code)

    if info is None:
        click.secho(f"无法获取基金 {fund_code} 的数据，请检查代码是否正确", fg="red")
        return

    fund = init_fund_state(
        fund_code=fund_code,
        name=info["name"],
        capital=capital,
        base_ratio=base_ratio,
        tier_step=tier_step,
        multiplier=multiplier,
        entry_threshold=entry,
        sell_rule=sell_rule,
    )

    current_nav = info["nav"]
    state["funds"][fund_code] = fund
    bond_yield = fetch_bond_yield() if sell_rule == "spread" else None
    signals = generate_signals(fund, current_nav, info["date"], bond_yield)
    execute_signals(fund, signals, info["date"])

    save_state(state)

    click.secho(f"✓ 已添加 {info['name']}（{fund_code}）", fg="green")
    print_status(fund, current_nav, signals)


@cli.command()
@click.argument("fund_code", required=False)
@click.option("--execute/--no-execute", default=False,
              help="是否将买入信号记录到持仓中")
def check(fund_code, execute):
    """查看基金状态和交易信号。不指定代码则查看全部。"""
    state = load_state()

    if not state["funds"]:
        click.secho("暂无跟踪基金，请先使用 add 命令添加", fg="yellow")
        return

    codes = [fund_code] if fund_code else list(state["funds"].keys())

    for code in codes:
        if code not in state["funds"]:
            click.secho(f"基金 {code} 不在跟踪列表中", fg="red")
            continue

        fund = state["funds"][code]
        click.echo(f"\n正在更新 {fund['name']}（{code}）...")

        # 如果设置了回撤起始日期且 peak 早于该日期，先校准
        start_date = fund.get("drawdown_start_date")
        peak_date = fund.get("peak_date")
        if start_date and (not peak_date or peak_date < start_date):
            click.secho(f"  从 {start_date} 重新计算前期高点...", fg="bright_black")
            recalc = recalibrate_peak(code, start_date)
            if recalc:
                fund["peak_price"] = recalc["peak"]
                fund["peak_date"] = recalc["peak_date"]
                fund["trough_price"] = recalc["trough"]
                fund["last_trigger_level"] = None
                fund["last_trigger_tier"] = None
                click.secho(f"  校准后前期高点: {recalc['peak']:.4f}（{recalc['peak_date']}）", fg="bright_black")

        is_spread = fund.get("sell_rule") == "spread"
        info = fetch_etf_price(code) if is_spread else fetch_latest_nav(code)

        if info is None:
            click.secho(f"  无法获取 {code} 的最新数据，使用缓存状态", fg="yellow")
            nav = fund.get("peak_price", 0)
            nav_date = fund.get("last_updated", "")
        else:
            nav = info["nav"]
            nav_date = info["date"]

        bond_yield = fetch_bond_yield() if is_spread else None
        signals = generate_signals(fund, nav, nav_date, bond_yield, advance_signals=execute)
        if execute:
            execute_signals(fund, signals, nav_date)

        state["funds"][code] = fund
        print_status(fund, nav, signals)

    save_state(state)


@cli.command()
@click.argument("fund_code", required=False)
def remove(fund_code):
    """从跟踪列表中移除基金。不指定代码则移除全部。"""
    state = load_state()

    if fund_code:
        if fund_code in state["funds"]:
            name = state["funds"][fund_code]["name"]
            del state["funds"][fund_code]
            save_state(state)
            click.secho(f"✓ 已移除 {name}（{fund_code}）", fg="green")
        else:
            click.secho(f"基金 {fund_code} 不在跟踪列表中", fg="yellow")
    else:
        count = len(state["funds"])
        state["funds"] = {}
        save_state(state)
        click.secho(f"✓ 已移除全部 {count} 只基金", fg="green")


@cli.command()
@click.argument("fund_code")
@click.option("--capital", "-c", type=float, help="总资金（元）")
@click.option("--base-ratio", "-r", type=float, help="基础买入比例")
@click.option("--entry", "-e", type=float, help="入场回撤阈值（%）")
@click.option("--tier-step", "-s", type=float, help="每档回撤幅度（%）")
@click.option("--sell-rule", type=click.Choice(["momentum", "mean_revert", "none", "spread"]),
              help="卖出规则: momentum=趋势型, mean_revert=均值回归型")
@click.option("--start-date", "-d", type=str, help="回撤计算起始日期，如 2026-04-01")
def config(fund_code, capital, base_ratio, entry, tier_step, sell_rule, start_date):
    """修改基金的策略参数"""
    state = load_state()

    if fund_code not in state["funds"]:
        click.secho(f"基金 {fund_code} 不在跟踪列表中", fg="red")
        return

    fund = state["funds"][fund_code]
    if capital is not None:
        fund["capital"] = capital
        click.secho(f"  总资金 → {capital:,}", fg="green")
    if base_ratio is not None:
        fund["base_ratio"] = base_ratio
        click.secho(f"  基础买入比例 → {base_ratio*100}%", fg="green")
    if entry is not None:
        fund["entry_threshold"] = entry
        fund["last_trigger_level"] = None
        click.secho(f"  入场回撤阈值 → {entry}%", fg="green")
    if tier_step is not None:
        fund["tier_step"] = tier_step
        fund["last_trigger_level"] = None
        click.secho(f"  每档回撤幅度 → {tier_step}%", fg="green")
    if sell_rule is not None:
        fund["sell_rule"] = sell_rule
        click.secho(f"  卖出规则 → {sell_rule}", fg="green")
    if start_date is not None:
        fund["drawdown_start_date"] = start_date
        # 清除 peak_date 以触发下次 check 时的重新校准
        fund["peak_date"] = None
        click.secho(f"  回撤起始日期 → {start_date}（下次 check 时将自动校准）", fg="green")

    save_state(state)


@cli.command()
@click.argument("fund_code")
@click.option("--days", "-d", type=int, default=365,
              help="回测天数，默认 365 天")
def backtest(fund_code, days):
    """
    历史回测：用历史数据模拟策略表现。不修改当前跟踪状态。

    示例：python fund_tracker.py backtest 270042 -d 365
    """
    click.echo(f"正在获取 {fund_code} 历史数据...")
    state = load_state()
    existing = state.get("funds", {}).get(fund_code, {})
    is_spread = existing.get("sell_rule") == "spread"
    history = fetch_etf_history(fund_code, days=days) if is_spread else fetch_nav_history(fund_code, days=days)

    if not history:
        click.secho("无法获取历史数据", fg="red")
        return

    # 取最近 N 个交易日（覆盖约 days 个自然日）
    if fund_code in state["funds"]:
        fund = state["funds"][fund_code].copy()
    else:
        info = fetch_latest_nav(fund_code)
        name = info["name"] if info else fund_code
        fund = init_fund_state(fund_code, name, DEFAULT_CAPITAL,
                               DEFAULT_BASE_RATIO, DEFAULT_TIER_STEP,
                               DEFAULT_MULTIPLIER, DEFAULT_ENTRY_THRESHOLD,
                               DEFAULT_SELL_RULE)

    click.secho(f"\n回测 {fund['name']}（{fund_code}），"
                f"{history[0]['date']} ~ {history[-1]['date']}，共 {len(history)} 个交易日",
                fg="cyan", bold=True)

    bond_yields = fetch_bond_yield_history(days + 30) if is_spread else None
    if is_spread and not bond_yields:
        click.secho("无法获取历史国债收益率，已停止利差回测", fg="red")
        return
    result = simulate_backtest(fund, history, bond_yields=bond_yields)
    for event in result["events"][:10]:
        color = "green" if event["type"] == "buy" else "magenta"
        click.secho(f"  [{event['date']}] {event['type']} ¥{event['amount']:,.2f}", fg=color)

    # 回测总结
    click.secho(f"\n{'─' * 50}", fg="bright_black")
    click.secho("  回测结果", fg="cyan", bold=True)
    click.secho(f"  买入次数: {result['total_buys']}  |  卖出次数: {result['total_sells']}", fg="white")
    click.secho(f"  期末权益: ¥{result['final_equity']:,.2f}  |  费用: ¥{result['total_fees']:,.2f}", fg="white")
    click.secho(f"  总收益: {result['total_return'] * 100:+.2f}%  |  最大回撤: {result['max_drawdown'] * 100:.2f}%", fg="white")
    click.secho(f"  买入持有基准: {result['benchmark_buy_hold_return'] * 100:+.2f}%  |  月度定投基准: {result['benchmark_monthly_dca_return'] * 100:+.2f}%", fg="white")

    click.secho(f"{'─' * 50}", fg="bright_black")


if __name__ == "__main__":
    cli()
