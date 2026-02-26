from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_DEFAULT_MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
_OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

_MEMORY_MAX_TURNS = int(os.getenv("AI_MEMORY_MAX_TURNS", "10"))
_MEMORY_TTL_SECONDS = int(os.getenv("AI_MEMORY_TTL_SECONDS", "21600"))  # 6 hours
_RANKING_MAX_ITEMS = int(os.getenv("AI_RANKING_MAX_ITEMS", "25"))

# session_id -> {"updated_at": float, "messages": [{"role": "user"|"assistant", "content": str}]}
_MEMORY: Dict[str, Dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _prune_memory() -> None:
    if not _MEMORY:
        return
    cutoff = _now() - _MEMORY_TTL_SECONDS
    expired = [sid for sid, v in _MEMORY.items() if float(v.get("updated_at", 0.0)) < cutoff]
    for sid in expired:
        _MEMORY.pop(sid, None)


def _get_or_create_session_id(session_id: Optional[str]) -> str:
    if session_id:
        return session_id
    return str(uuid.uuid4())


def _append_memory(session_id: str, role: str, content: str) -> None:
    _prune_memory()
    rec = _MEMORY.get(session_id)
    if rec is None:
        rec = {"updated_at": _now(), "messages": []}
        _MEMORY[session_id] = rec
    rec["updated_at"] = _now()
    rec["messages"].append({"role": role, "content": content})
    rec["messages"] = rec["messages"][-(2 * _MEMORY_MAX_TURNS) :]


def _get_memory(session_id: str) -> List[Dict[str, str]]:
    _prune_memory()
    rec = _MEMORY.get(session_id)
    if not rec:
        return []
    msgs = rec.get("messages", [])
    return [m for m in msgs if isinstance(m, dict) and "role" in m and "content" in m]


def detect_intent(question: str) -> Set[str]:
    q = (question or "").lower()
    intents: Set[str] = set()

    if any(k in q for k in ("revenue", "mrr", "arr", "growth", "expansion", "contraction")):
        intents.add("revenue")
    if any(k in q for k in ("churn", "retention", "cancel", "downgrade", "renewal")):
        intents.add("churn")
    if any(k in q for k in ("plan", "pricing", "tier")):
        intents.add("plan")
    if any(k in q for k in ("industry", "vertical", "segment")):
        intents.add("industry")
    if any(k in q for k in ("region", "geo", "geography", "country", "market")):
        intents.add("region")
    if any(k in q for k in ("support", "ticket", "escalation", "csat", "nps")):
        intents.add("support")

    # Executive / general business update queries
    if any(k in q for k in ("executive", "summary", "overview", "board", "so what", "health", "how are we doing")):
        intents.add("general")

    if not intents:
        intents.add("general")
    # If the user asked a specific question (churn/revenue/etc), don't treat it as a general overview.
    if "general" in intents and len(intents) > 1:
        intents.discard("general")
    return intents


def detect_ranking_intent(question: str) -> bool:
    q = (question or "").lower()
    ranking_keywords = (
        "second",
        "third",
        "rank",
        "ranking",
        "top",
        "highest",
        "lowest",
        "compare",
        "comparison",
        "distribution",
        "most",
        "least",
    )
    return any(k in q for k in ranking_keywords)


def _infer_ranking_needs(question: str, intents: Set[str]) -> Dict[str, bool]:
    """
    Decide which ranked distributions are needed for ranking/comparison questions.
    """
    q = (question or "").lower()

    has_plan = any(k in q for k in ("plan", "pricing", "tier"))
    has_region = any(k in q for k in ("region", "geo", "geography", "country", "market"))
    has_industry = any(k in q for k in ("industry", "vertical", "segment"))

    churn_like = ("churn" in intents) or any(k in q for k in ("churn", "retention", "cancel", "downgrade", "renewal"))
    revenue_like = ("revenue" in intents) or any(k in q for k in ("revenue", "mrr", "arr"))

    needs = {
        "revenue_by_plan_ranked": False,
        "revenue_by_region_ranked": False,
        "churn_by_plan_ranked": False,
        "churn_by_industry_ranked": False,
    }

    if has_region:
        needs["revenue_by_region_ranked"] = True

    if has_industry:
        needs["churn_by_industry_ranked"] = True

    if has_plan:
        # If user is explicitly asking about churn + plan, use churn distribution.
        # Otherwise default to revenue plan distribution.
        if churn_like and not revenue_like:
            needs["churn_by_plan_ranked"] = True
        else:
            needs["revenue_by_plan_ranked"] = True

    # If the question is ranking-like but doesn't specify the dimension,
    # provide the most relevant distributions based on intent.
    if not (has_plan or has_region or has_industry):
        if churn_like:
            needs["churn_by_plan_ranked"] = True
            needs["churn_by_industry_ranked"] = True
        if revenue_like:
            needs["revenue_by_plan_ranked"] = True
            needs["revenue_by_region_ranked"] = True

    return needs


def _pick_endpoints(intents: Set[str]) -> List[str]:
    endpoints: Set[str] = set()

    # Minimal set for an executive-grade answer.
    if "general" in intents:
        endpoints.update(
            {
                "/revenue/monthly",
                "/revenue/by-plan",
                "/revenue/by-region",
                "/churn/by-plan",
                "/churn/by-industry",
                "/churn/support-correlation",
            }
        )

    if "revenue" in intents:
        endpoints.add("/revenue/monthly")
        endpoints.add("/revenue/by-plan")
    if "region" in intents:
        endpoints.add("/revenue/by-region")

    if "churn" in intents:
        endpoints.add("/churn/by-plan")
        endpoints.add("/churn/by-industry")
        endpoints.add("/churn/support-correlation")

    if "plan" in intents:
        endpoints.add("/churn/by-plan")
        endpoints.add("/revenue/by-plan")
    if "industry" in intents:
        endpoints.add("/churn/by-industry")
    if "support" in intents:
        endpoints.add("/churn/support-correlation")

    return sorted(endpoints)


def _http_get_json(path: str, *, base_url: str, timeout_s: float = 15.0) -> Any:
    url = f"{base_url}{path}"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        text = raw.decode("utf-8") if raw else ""
        return json.loads(text) if text else None
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = ""
        raise RuntimeError(f"MCP endpoint error {e.code} for {path}: {detail}".strip()) from e
    except URLError as e:
        raise RuntimeError(f"Failed to reach MCP endpoint {path}: {e}") from e


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_by_month(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    if "month" not in rows[0]:
        return rows
    return sorted(rows, key=lambda r: str(r.get("month", "")))


def _pct_change(prev: Optional[float], curr: Optional[float]) -> Optional[float]:
    if prev is None or curr is None or prev == 0:
        return None
    return ((curr - prev) / prev) * 100.0


def _max_row(
    rows: List[Dict[str, Any]],
    *,
    value_keys: Sequence[str],
    label_keys: Sequence[str],
) -> Optional[Dict[str, Any]]:
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for r in rows:
        v = None
        for k in value_keys:
            if k in r:
                v = _safe_float(r.get(k))
                if v is not None:
                    break
        if v is None:
            continue
        if best is None or v > best[0]:
            best = (v, r)
    if best is None:
        return None

    row = best[1]
    label = None
    for lk in label_keys:
        if lk in row and row.get(lk) is not None:
            label = str(row.get(lk))
            break
    return {"label": label, "value": best[0], "raw_keys": list(row.keys())}


def _best_named_metric(
    rows: List[Dict[str, Any]],
    *,
    label_out_key: str,
    label_in_keys: Sequence[str],
    value_out_key: str,
    value_in_keys: Sequence[str],
) -> Optional[Dict[str, Any]]:
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for r in rows:
        v = None
        for k in value_in_keys:
            if k in r:
                v = _safe_float(r.get(k))
                if v is not None:
                    break
        if v is None:
            continue
        if best is None or v > best[0]:
            best = (v, r)

    if best is None:
        return None

    row = best[1]
    label = None
    for lk in label_in_keys:
        if lk in row and row.get(lk) is not None:
            label = str(row.get(lk))
            break

    return {label_out_key: label, value_out_key: best[0]}


def _summarize_latest_mrr_and_growth(revenue_monthly: Any) -> Dict[str, Any]:
    rows = revenue_monthly if isinstance(revenue_monthly, list) else []
    rows = [r for r in rows if isinstance(r, dict)]
    rows = _sort_by_month(rows)

    mrr_key_candidates = ("total_mrr", "mrr", "monthly_mrr", "revenue", "total_revenue")
    month = str(rows[-1].get("month")) if rows else None

    def pick_mrr(row: Dict[str, Any]) -> Optional[float]:
        for k in mrr_key_candidates:
            if k in row:
                v = _safe_float(row.get(k))
                if v is not None:
                    return v
        return None

    latest_mrr = pick_mrr(rows[-1]) if rows else None

    # 3-month growth: compare latest vs value ~3 periods back if available.
    base_mrr = None
    if len(rows) >= 4:
        base_mrr = pick_mrr(rows[-4])
    elif len(rows) >= 2:
        base_mrr = pick_mrr(rows[0])

    growth_3m_pct = _pct_change(base_mrr, latest_mrr) if base_mrr is not None else None

    return {
        "latest_month": month,
        "latest_mrr": latest_mrr,
        "mrr_growth_3m_pct": growth_3m_pct,
    }


def _summarize_revenue_concentration_by_region(revenue_by_region: Any) -> Dict[str, Any]:
    rows = revenue_by_region if isinstance(revenue_by_region, list) else []
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return {"top_regions": [], "top_region_share_pct": None}

    region_keys = ("region", "geo_region", "country", "market")
    value_keys = ("revenue", "total_revenue", "mrr", "total_mrr", "arr", "total_arr")

    parsed: List[Tuple[str, float]] = []
    for r in rows:
        region = None
        for rk in region_keys:
            if rk in r and r.get(rk) is not None:
                region = str(r.get(rk))
                break
        if region is None:
            continue
        value = None
        for vk in value_keys:
            if vk in r:
                value = _safe_float(r.get(vk))
                if value is not None:
                    break
        if value is None:
            continue
        parsed.append((region, value))

    if not parsed:
        return {"top_regions": [], "top_region_share_pct": None}

    parsed.sort(key=lambda t: t[1], reverse=True)
    total = sum(v for _, v in parsed) or 0.0
    top_regions = [{"region": reg, "revenue": val} for reg, val in parsed[:3]]
    top_share = (parsed[0][1] / total * 100.0) if total > 0 else None
    return {"top_regions": top_regions, "top_region_share_pct": top_share}


def _summarize_top_revenue_region(revenue_by_region: Any) -> Optional[Dict[str, Any]]:
    ranked = _rank_named_distribution(
        revenue_by_region,
        label_in_keys=("region", "geo_region", "country", "market"),
        value_in_keys=("mrr", "total_mrr", "revenue", "total_revenue", "arr", "total_arr"),
        label_out_key="region",
        value_out_key="mrr",
    )
    if not ranked:
        return None
    return ranked[0]


def _summarize_top_revenue_plan(revenue_by_plan: Any) -> Optional[Dict[str, Any]]:
    ranked = _rank_named_distribution(
        revenue_by_plan,
        label_in_keys=("plan", "plan_name", "subscription_plan", "tier"),
        value_in_keys=("mrr", "total_mrr", "revenue", "total_revenue", "arr", "total_arr"),
        label_out_key="plan_name",
        value_out_key="mrr",
    )
    if not ranked:
        return None
    return ranked[0]


def _rank_named_distribution(
    rows_any: Any,
    *,
    label_in_keys: Sequence[str],
    value_in_keys: Sequence[str],
    label_out_key: str,
    value_out_key: str,
) -> List[Dict[str, Any]]:
    rows = rows_any if isinstance(rows_any, list) else []
    rows = [r for r in rows if isinstance(r, dict)]
    parsed: List[Tuple[str, float]] = []

    def infer_label_key() -> Optional[str]:
        candidates = []
        for k in rows[0].keys() if rows else []:
            lk = k.lower()
            if any(s in lk for s in ("plan", "tier", "region", "country", "market", "industry", "vertical", "segment")):
                candidates.append(k)
        scan_keys = candidates + [k for k in (rows[0].keys() if rows else []) if k not in candidates]
        for k in scan_keys:
            for r in rows[:10]:
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    return k
        return None

    def infer_value_key() -> Optional[str]:
        prefer = []
        for k in rows[0].keys() if rows else []:
            kk = k.lower()
            if any(s in kk for s in ("mrr", "revenue", "arr", "churn")):
                prefer.append(k)
        scan_keys = prefer + [k for k in (rows[0].keys() if rows else []) if k not in prefer]
        best_key = None
        best_score = None
        for k in scan_keys:
            values: List[float] = []
            for r in rows[:20]:
                fv = _safe_float(r.get(k))
                if fv is not None:
                    values.append(abs(fv))
            if not values:
                continue
            score = (len(values), max(values))
            if best_score is None or score > best_score:
                best_score = score
                best_key = k
        return best_key

    for r in rows:
        label = None
        for lk in label_in_keys:
            if lk in r and r.get(lk) is not None:
                label = str(r.get(lk))
                break
        if not label:
            continue

        value = None
        for vk in value_in_keys:
            if vk in r:
                value = _safe_float(r.get(vk))
                if value is not None:
                    break
        if value is None:
            continue

        parsed.append((label, value))

    if not parsed and rows:
        inferred_label_key = infer_label_key()
        inferred_value_key = infer_value_key()
        if inferred_label_key and inferred_value_key:
            for r in rows:
                label_val = r.get(inferred_label_key)
                value_val = _safe_float(r.get(inferred_value_key))
                if isinstance(label_val, str) and label_val.strip() and value_val is not None:
                    parsed.append((label_val.strip(), value_val))

    parsed.sort(key=lambda t: t[1], reverse=True)

    if not parsed:
        return []

    limit = min(len(parsed), _RANKING_MAX_ITEMS)
    limit = max(limit, min(5, len(parsed)))
    out: List[Dict[str, Any]] = []
    for label, value in parsed[:limit]:
        out.append({label_out_key: label, value_out_key: value})
    return out


def _summarize_support_correlation(support_corr: Any) -> Dict[str, Any]:
    rows = support_corr if isinstance(support_corr, list) else []
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return {"correlation": None, "interpretation": None}

    r0 = rows[0]
    corr_key_candidates = ("correlation", "corr", "correlation_coefficient", "pearson_r")
    corr = None
    for k in corr_key_candidates:
        if k in r0:
            corr = _safe_float(r0.get(k))
            if corr is not None:
                break

    interpretation = None
    if corr is not None:
        if corr >= 0.5:
            interpretation = "Strong positive correlation (higher support escalations associated with higher churn)."
        elif corr >= 0.2:
            interpretation = "Moderate positive correlation."
        elif corr <= -0.5:
            interpretation = "Strong negative correlation."
        elif corr <= -0.2:
            interpretation = "Moderate negative correlation."
        else:
            interpretation = "Weak or no clear correlation."

    return {"correlation": corr, "interpretation": interpretation}


def _build_structured_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    revenue_monthly = raw.get("/revenue/monthly")
    revenue_by_plan = raw.get("/revenue/by-plan")
    revenue_by_region = raw.get("/revenue/by-region")
    churn_by_plan = raw.get("/churn/by-plan")
    churn_by_industry = raw.get("/churn/by-industry")
    support_corr = raw.get("/churn/support-correlation")

    mrr_metrics = _summarize_latest_mrr_and_growth(revenue_monthly)
    region_metrics = _summarize_revenue_concentration_by_region(revenue_by_region)
    top_revenue_region = _summarize_top_revenue_region(revenue_by_region)
    top_revenue_plan = _summarize_top_revenue_plan(revenue_by_plan)
    support_metrics = _summarize_support_correlation(support_corr)

    churn_plan_rows = churn_by_plan if isinstance(churn_by_plan, list) else []
    churn_plan_rows = [r for r in churn_plan_rows if isinstance(r, dict)]
    churn_ind_rows = churn_by_industry if isinstance(churn_by_industry, list) else []
    churn_ind_rows = [r for r in churn_ind_rows if isinstance(r, dict)]

    highest_churn_plan = _best_named_metric(
        churn_plan_rows,
        label_out_key="plan_name",
        label_in_keys=("plan", "plan_name", "subscription_plan", "tier"),
        value_out_key="churn_rate",
        value_in_keys=("churn_rate", "churn_pct", "churn_percentage"),
    )
    highest_churn_industry = _best_named_metric(
        churn_ind_rows,
        label_out_key="industry",
        label_in_keys=("industry", "industry_name", "vertical", "segment"),
        value_out_key="churn_rate",
        value_in_keys=("churn_rate", "churn_pct", "churn_percentage"),
    )

    # Required structured context (explicit labels + numeric values).
    return {
        "latest_mrr": mrr_metrics.get("latest_mrr"),
        "mrr_growth_last_3_months": mrr_metrics.get("mrr_growth_3m_pct"),
        "top_revenue_plan": top_revenue_plan,
        "top_revenue_region": top_revenue_region,
        "highest_churn_plan": highest_churn_plan,
        "highest_churn_industry": highest_churn_industry,
        "support_escalation_correlation": support_metrics.get("correlation"),
        # Extra (allowed): useful for deeper analysis/debugging.
        "latest_mrr_month": mrr_metrics.get("latest_month"),
        "revenue_concentration_by_region": region_metrics,
    }


def _drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            v2 = _drop_nones(v)
            if v2 is None:
                continue
            if isinstance(v2, dict) and not v2:
                continue
            if isinstance(v2, list) and not v2:
                continue
            out[k] = v2
        return out
    if isinstance(value, list):
        cleaned = [_drop_nones(v) for v in value]
        return [v for v in cleaned if v is not None]
    return value


def _filter_structured_context(
    structured_metrics: Dict[str, Any],
    intents: Set[str],
    *,
    ranking_needs: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Ensure we only provide relevant metrics to the LLM so answers stay focused.
    """
    is_churn = "churn" in intents or "industry" in intents or "plan" in intents or "support" in intents
    is_revenue = "revenue" in intents or "region" in intents
    is_general = "general" in intents

    ctx: Dict[str, Any] = {}

    if is_general or is_revenue:
        for k in ("latest_mrr", "mrr_growth_last_3_months", "top_revenue_plan", "top_revenue_region"):
            if k in structured_metrics:
                ctx[k] = structured_metrics.get(k)

    if is_general or is_churn:
        for k in ("highest_churn_plan", "highest_churn_industry", "support_escalation_correlation"):
            if k in structured_metrics:
                ctx[k] = structured_metrics.get(k)

    # Always allow the model to ground to the month if available (but only if present).
    if structured_metrics.get("latest_mrr_month") is not None:
        ctx["latest_mrr_month"] = structured_metrics.get("latest_mrr_month")

    if ranking_needs:
        for key, needed in ranking_needs.items():
            if needed and key in structured_metrics:
                ctx[key] = structured_metrics.get(key)

    return _drop_nones(ctx)


def _openai_chat_completion(*, messages: List[Dict[str, str]]) -> str:
    if not _OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    payload = {
        "model": _OPENAI_MODEL,
        "temperature": _OPENAI_TEMPERATURE,
        "messages": messages,
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{_OPENAI_API_BASE}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {_OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=30.0) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8")) if raw else {}
        return data["choices"][0]["message"]["content"]
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = ""
        raise RuntimeError(f"OpenAI error {e.code}: {detail}".strip()) from e
    except URLError as e:
        raise RuntimeError(f"Failed to reach OpenAI: {e}") from e


def chat(
    question: str,
    *,
    session_id: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Production constraints (per architecture):
    - AI layer must call MCP endpoints only (no direct Postgres access).
    - LLM receives ONLY structured metrics (not raw JSON, not SQL).
    """
    if not question or not question.strip():
        raise ValueError("question is required")

    session_id = _get_or_create_session_id(session_id)
    base_url = (base_url or _DEFAULT_MCP_BASE_URL).rstrip("/")

    intents = detect_intent(question)
    is_ranking = detect_ranking_intent(question)
    ranking_needs = _infer_ranking_needs(question, intents) if is_ranking else None

    # Even for non-ranking questions, include ranked distributions when the question
    # requires selecting an item (e.g., "which plan contributes most MRR?").
    ql = (question or "").lower()
    soft_needs = {
        "revenue_by_plan_ranked": any(k in ql for k in ("plan", "tier", "pricing")) and any(k in ql for k in ("revenue", "mrr", "arr")),
        "revenue_by_region_ranked": any(k in ql for k in ("region", "geo", "geography", "country", "market")) and any(k in ql for k in ("revenue", "mrr", "arr")),
        "churn_by_plan_ranked": any(k in ql for k in ("plan", "tier")) and any(k in ql for k in ("churn", "retention", "cancel", "downgrade", "renewal")),
        "churn_by_industry_ranked": any(k in ql for k in ("industry", "vertical", "segment")) and any(k in ql for k in ("churn", "retention", "cancel", "downgrade", "renewal")),
    }
    distribution_needs = ranking_needs or {k: v for k, v in soft_needs.items() if v}

    endpoints = set(_pick_endpoints(intents))
    if distribution_needs:
        if distribution_needs.get("revenue_by_plan_ranked"):
            endpoints.add("/revenue/by-plan")
        if distribution_needs.get("revenue_by_region_ranked"):
            endpoints.add("/revenue/by-region")
        if distribution_needs.get("churn_by_plan_ranked"):
            endpoints.add("/churn/by-plan")
        if distribution_needs.get("churn_by_industry_ranked"):
            endpoints.add("/churn/by-industry")
    endpoints = sorted(endpoints)

    raw: Dict[str, Any] = {}
    for ep in endpoints:
        raw[ep] = _http_get_json(ep, base_url=base_url)

    structured_metrics = _build_structured_metrics(raw)
    if distribution_needs:
        if distribution_needs.get("revenue_by_plan_ranked"):
            structured_metrics["revenue_by_plan_ranked"] = _rank_named_distribution(
                raw.get("/revenue/by-plan"),
                label_in_keys=("plan", "plan_name", "subscription_plan", "tier"),
                value_in_keys=("mrr", "total_mrr", "revenue", "total_revenue", "arr", "total_arr"),
                label_out_key="plan_name",
                value_out_key="mrr",
            )
        if distribution_needs.get("revenue_by_region_ranked"):
            structured_metrics["revenue_by_region_ranked"] = _rank_named_distribution(
                raw.get("/revenue/by-region"),
                label_in_keys=("region", "geo_region", "country", "market"),
                value_in_keys=("mrr", "total_mrr", "revenue", "total_revenue", "arr", "total_arr"),
                label_out_key="region",
                value_out_key="mrr",
            )
        if distribution_needs.get("churn_by_plan_ranked"):
            structured_metrics["churn_by_plan_ranked"] = _rank_named_distribution(
                raw.get("/churn/by-plan"),
                label_in_keys=("plan", "plan_name", "subscription_plan", "tier"),
                value_in_keys=("churn_rate", "churn_pct", "churn_percentage"),
                label_out_key="plan_name",
                value_out_key="churn_rate",
            )
        if distribution_needs.get("churn_by_industry_ranked"):
            structured_metrics["churn_by_industry_ranked"] = _rank_named_distribution(
                raw.get("/churn/by-industry"),
                label_in_keys=("industry", "industry_name", "vertical", "segment"),
                value_in_keys=("churn_rate", "churn_pct", "churn_percentage"),
                label_out_key="industry",
                value_out_key="churn_rate",
            )

    structured_context = _filter_structured_context(structured_metrics, intents, ranking_needs=distribution_needs)

    history = _get_memory(session_id)
    _append_memory(session_id, "user", question)
    user_history = [m for m in history if m.get("role") == "user"][-6:]

    system = (
        "You are a SaaS CFO assistant.\n"
        "\n"
        "You must:\n"
        "- Reference exact plan names and industries from the structured context.\n"
        "- Never use vague phrases like 'highest plan' or 'highest industry'. Always use the labels provided in the structured context.\n"
        "- Always mention numeric values when referencing any metric.\n"
        "- Only use data provided.\n"
        "- If data is missing, state it clearly.\n"
        "- Do not hallucinate.\n"
        "- Answer ONLY what the user asked. Do not provide an overall company overview unless the user explicitly requests it.\n"
        "- Conversation history is for continuity only; never treat it as a source of facts or numbers.\n"
        "- You are provided ranked lists when available. You must use list positions to answer ranking questions.\n"
        "- If the user asks for second or third highest, select the correct index from the ranked list.\n"
        "- Do not claim data is missing if a ranked list is present.\n"
        "\n"
        "Provide:\n"
        "- Trend analysis\n"
        "- Risk evaluation\n"
        "- Business recommendation\n"
        "- Optional follow-up question (only if it would materially change the recommendation)\n"
        "\n"
        "Tone: executive, specific, concise.\n"
    )

    user_content = json.dumps(
        {
            "user_question": question,
            "detected_intents": sorted(intents),
            "structured_context": structured_context,
            "conversation_history": user_history,
            "constraints": {
                "no_hallucination": True,
                "use_only_structured_metrics": True,
            },
        },
        ensure_ascii=False,
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    try:
        answer = _openai_chat_completion(messages=messages)
    except RuntimeError as e:
        # Safe fallback: still return structured metrics, but make the missing OpenAI setup obvious.
        answer = (
            "AI reasoning is currently unavailable because OpenAI is not configured.\n"
            f"Error: {str(e)}\n\n"
            "Structured metrics are included in this response so you can validate the data flow through MCP endpoints."
        )

    _append_memory(session_id, "assistant", answer)

    return {
        "session_id": session_id,
        "used_endpoints": endpoints,
        "structured_metrics": structured_context,
        "answer": answer,
    }

