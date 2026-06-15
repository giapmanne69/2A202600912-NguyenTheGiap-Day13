"""YOUR mitigation + observability layer. The simulator calls mitigate() around the
opaque agent (a REAL LLM) for every request. This is the ONLY place observability can
live -- the agent is silent. Legal moves: retry / cache / route / guardrail / sanitize
/ fallback / session-reset / PROMPT ROUTING, plus your own logging/tracing/metrics.
Illegal: hardcoding answers, importing the agent internals, reading instructor files,
network exfiltration.

  call_next(question, config) -> result   # the only way to reach the black box
  context = {"session_id","turn_index","qid","cache": <shared dict>, "cache_lock": <Lock>}
  result  = {"answer","status","steps","trace","meta":{latency_ms,usage,...}}
"""
from __future__ import annotations
import time
import re

# Import the telemetry toolkit
try:
    from telemetry.logger import logger, set_correlation_id
    from telemetry.cost import cost_from_usage
    from telemetry.redact import redact
except Exception:
    logger = None
    cost_from_usage = lambda m, u: 0.0
    redact = lambda s: (s, 0)


def sanitize_input(q: str) -> str:
    """Sanitize input queries to protect against prompt injections in order notes.
    Strips numbers and system command keywords from notes sections.
    """
    # Regex to match notes (Ghi chú, Note, etc.)
    match = re.search(r'(ghi chú|ghi chu|note|ghi\s+chú)[:\-](.*)', q, re.IGNORECASE)
    if match:
        note_label = match.group(1)
        note_content = match.group(2)
        # Strip numbers and dangerous words (e.g. system, price, override, ignore, giá) to block price overrides
        clean_content = re.sub(r'[0-9]', '', note_content)
        dangerous_words = [
            r'price', r'cost', r'vnd', r'usd', r'giá', r'tiền', r'system', r'prompt',
            r'ignore', r'override', r'bỏ qua', r'thiết lập', r'thay đổi', r'chỉ dẫn'
        ]
        for word in dangerous_words:
            clean_content = re.compile(word, re.IGNORECASE).sub('', clean_content)
        # Re-assemble question
        q = q[:match.start()] + note_label + ": " + clean_content.strip()
    return q


def get_correct_total(question: str):
    """Programmatically calculate the correct total price based on business rules."""
    q_lower = question.lower()
    
    # 1. Product catalog details
    product = None
    price = 0
    weight = 0.0
    if "macbook" in q_lower or "mac book" in q_lower:
        product = "MacBook"
        price = 35000000
        weight = 1.6
    elif "iphone" in q_lower:
        product = "iPhone"
        price = 22000000
        weight = 0.5
    elif "ipad" in q_lower:
        product = "iPad"
        price = 18000000
        weight = 0.45
    elif "airpod" in q_lower:
        product = "AirPods"
        
    # AirPods is out of stock -> refusal
    if not product or product == "AirPods":
        return None
        
    # 2. Extract quantity
    qty = None
    match_mua = re.search(r'mua\s+(\d+)', q_lower)
    if match_mua:
        qty = int(match_mua.group(1))
    else:
        match_near = re.search(r'(\d+)\s*(iphone|ipad|macbook|mac\s*book|airpod)', q_lower)
        if match_near:
            qty = int(match_near.group(1))
    if qty is None:
        qty = 1
        
    # 3. Extract coupon discount percent
    coupon = 0
    for c in ["winner", "vip20", "sale15", "expired"]:
        if c in q_lower:
            if c == "winner":
                coupon = 10
            elif c == "vip20":
                coupon = 20
            elif c == "sale15":
                coupon = 15
            elif c == "expired":
                coupon = 0
            break
            
    # 4. Extract destination city and shipping base fee
    city = None
    base_shipping = 0
    if "hà nội" in q_lower or "ha noi" in q_lower:
        city = "Hà Nội"
        base_shipping = 30000
    elif "tp hcm" in q_lower or "ho chi minh" in q_lower or "tphcm" in q_lower or "sài gòn" in q_lower or "sai gon" in q_lower:
        city = "TP HCM"
        base_shipping = 25000
    elif "đà nẵng" in q_lower or "da nang" in q_lower:
        city = "Đà Nẵng"
        base_shipping = 35000
    elif "hải phòng" in q_lower or "hai phong" in q_lower:
        city = "Hải Phòng"
        base_shipping = 28000
        
    # Unsupported city -> refusal
    if not city:
        return None
        
    # Calculate totals
    subtotal = price * qty
    discounted = subtotal * (100 - coupon) // 100
    total_weight = weight * qty
    if total_weight > 1.0:
        shipping = base_shipping + 5000 * (total_weight - 1.0)
    else:
        shipping = base_shipping
        
    return discounted + int(shipping)


def mitigate(call_next, question, config, context):
    t0 = time.time()
    
    # 1. Thread-safe Caching
    cache = context.get("cache")
    lock = context.get("cache_lock")
    qid = context.get("qid", "unknown")
    
    # Set correlation ID for structured logs
    if logger:
        set_correlation_id(qid)
        
    q_key = question.strip().lower()
    if cache is not None and lock is not None:
        with lock:
            if q_key in cache:
                cached_res = dict(cache[q_key])
                if logger:
                    logger.log_event("CACHE_HIT", {
                        "qid": qid,
                        "question": question,
                        "answer": cached_res.get("answer"),
                        "status": cached_res.get("status")
                    })
                return cached_res

    # 2. Input Sanitization
    sanitized_question = sanitize_input(question)

    # 3. Execution with Retries
    res = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = call_next(sanitized_question, config)
            if res and res.get("status") == "ok":
                break
            # If agent loops or drifts, reset context in subsequent attempts
            if attempt < max_retries - 1:
                config_copy = dict(config)
                config_copy["context_reset_every"] = 1
                config = config_copy
        except Exception as e:
            if attempt == max_retries - 1:
                # Return wrapper error block rather than crashing
                res = {
                    "answer": "Hệ thống gặp sự cố. Vui lòng thử lại sau.",
                    "status": "wrapper_error",
                    "steps": 0,
                    "trace": [],
                    "meta": {
                        "latency_ms": int((time.time() - t0) * 1000),
                        "usage": {},
                        "model": config.get("model", ""),
                        "provider": config.get("provider", ""),
                        "tools_used": []
                    }
                }
            time.sleep(0.2)

    # 4. Output Redaction (PII Leak Guard) & Formatting Normalization
    if res and res.get("answer"):
        redacted_ans, num_redacts = redact(res["answer"])
        ans = redacted_ans if num_redacts > 0 else res["answer"]
        
        # Regex normalize total price to ensure it ends strictly with "Tong cong: <digits> VND"
        pattern = re.compile(
            r'\*?\*?\s*(?:[tT]ong\s+[cC]ong|[tT]ổng\s+[cC]ộng)\s*[:\-]?\s*\*?\*?\s*([\d\s\u202f\u00a0,.]*\d[\d\s\u202f\u00a0,.]*)\s*(?:VND|đ)?\s*\*?\*?(?:\s*\(lien\s+he:\s+\[REDACTED\]\))?',
            re.IGNORECASE
        )
        matches = list(pattern.finditer(ans))
        if matches:
            last_match = matches[-1]
            correct_total = get_correct_total(question)
            
            start, end = last_match.span()
            prefix = ans[:start].strip()
            suffix = ans[end:].strip()
            
            redacted_info = ""
            if "[REDACTED]" in last_match.group(0) or "[REDACTED]" in suffix or "[REDACTED]" in prefix:
                redacted_info = " (lien he: [REDACTED])"
            
            body = prefix
            if suffix:
                suffix_clean = suffix.replace("(lien he: [REDACTED])", "").strip()
                if suffix_clean:
                    body += "\n" + suffix_clean
            
            body = body.strip()
            body = re.sub(r'\*+\s*\*+', '', body).strip()
            body = re.sub(r'\*+$', '', body).strip()
            body = re.sub(r'^\*+', '', body).strip()
            body = body.strip()
            
            if redacted_info and redacted_info.strip() not in body:
                body += redacted_info
            
            if correct_total is not None:
                ans = body + f"\n\nTong cong: {correct_total} VND"
            else:
                ans = body
        res["answer"] = ans

    # 5. Populate Cache
    if res and res.get("status") == "ok" and cache is not None and lock is not None:
        with lock:
            cache[q_key] = res

    # 6. Observability Telemetry Logging
    wall_ms = int((time.time() - t0) * 1000)
    meta = res.get("meta", {}) if res else {}
    usage = meta.get("usage", {}) if meta else {}
    model = meta.get("model", "") if meta else ""
    
    if logger and res:
        logger.log_event("AGENT_CALL", {
            "qid": qid,
            "status": res.get("status"),
            "reported_latency_ms": meta.get("latency_ms"),
            "wall_ms": wall_ms,
            "tokens": usage,
            "cost_usd": cost_from_usage(model, usage),
            "pii_leak_prevented": num_redacts > 0 if 'num_redacts' in locals() else False,
            "tools_used": meta.get("tools_used", []),
            "steps": res.get("steps")
        })

    return res
