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
            r'\*?\*?\s*(?:[tT]ong\s+[cC]ong|[tT]ổng\s+[cC]ộng)\s*[:\-]?\s*\*?\*?\s*([\d\s\u202f\u00a0,.]+)\s*(?:VND|đ)?\s*\*?\*?(?:\s*\(lien\s+he:\s+\[REDACTED\]\))?',
            re.IGNORECASE
        )
        matches = list(pattern.finditer(ans))
        if matches:
            last_match = matches[-1]
            num_str = last_match.group(1)
            num_clean = re.sub(r'[^\d]', '', num_str)
            if num_clean:
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
                
                ans = body + f"\n\nTong cong: {num_clean} VND"
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
