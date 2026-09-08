"""AgentDiet-style trajectory reduction for Hermes (Agent_Bench port).

AB_HARNESS_HERMES_PATCH_AGENTDIET_BEGIN

Port of the reflection module from the AgentDiet artifact (Xiao et al., "Reducing
Cost of LLM Agents with Trajectory Reduction", FSE 2026; figshare
10.6084/m9.figshare.30073654, ``code/trae_agent/agents/traj_analyzer.py``).

Mechanism, kept as in the artifact:

* After every tool round the step ``ctx_after`` rounds back (default 2) is the
  TARGET.  A step is one assistant message plus the ``role: tool`` messages
  that answer its tool_calls.
* If the serialized target is at least ``theta`` tokens (default 500) a cheap
  reflector LLM sees the window ``[target-ctx_before .. target+ctx_after]``
  serialized as ``<step id=..><think/><call tool=../><result/></step>`` and is
  asked to rewrite ONLY the target, replacing redundant text with
  ``"... (short takeaway)"``.  The assistant turn of the reflector request is
  prefilled with ``<step id="N">`` and generation stops at ``</step>``.
* The rewrite is accepted when ``old - new >= min_save`` (400) OR
  ``new < ratio * old`` (0.8) -- the artifact's rule, not the paper's theta.

What differs from the artifact (deliberately, see the report):

* ``writeback=structured`` (default) keeps the assistant message and its
  tool_calls and rewrites only the tool ``content`` fields and long tool-call
  argument strings, so every ``tool_call_id`` still has its result -- Hermes's
  ``_sanitize_tool_pairs`` and Agent_Bench's trace normalizer both key on it.
  ``writeback=faithful`` reproduces the artifact: the whole step collapses
  into one assistant message carrying the "(System reminder: ...)" marker.
* The reflector is called through the OpenAI client directly with an explicit
  base_url/model, so it can be a SEPARATE vLLM from the server under test and
  its tokens are accounted separately (event stream + ``agent._agentdiet``).

Inert unless ``HERMES_AGENTDIET=1``.  Every decision is appended as one JSON
line to ``HERMES_AGENTDIET_EVENTS`` (when set) so the replay side can model the
in-prefix edit; reflector failures are recorded there and logged, never raised
into the agent loop.  Programming errors DO raise: a silently inert arm is the
failure this patch exists to avoid.

Env knobs (all optional):
  HERMES_AGENTDIET            1 to enable
  HERMES_AGENTDIET_MODE       ours | delete | skip            (ours)
  HERMES_AGENTDIET_THETA      min target tokens to analyse    (500)
  HERMES_AGENTDIET_CTX_BEFORE steps before the target shown   (1)
  HERMES_AGENTDIET_CTX_AFTER  delay = steps after the target  (2)
  HERMES_AGENTDIET_MIN_SAVE   accept if old-new >= this       (400)
  HERMES_AGENTDIET_RATIO      accept if new < ratio*old       (0.8)
  HERMES_AGENTDIET_WRITEBACK  structured | faithful           (structured)
  HERMES_AGENTDIET_MODEL      reflector model id              (HERMES_INFERENCE_MODEL)
  HERMES_AGENTDIET_BASE_URL   reflector endpoint              (OPENAI_BASE_URL)
  HERMES_AGENTDIET_API_KEY    reflector key                   (OPENAI_API_KEY / EMPTY)
  HERMES_AGENTDIET_MAX_TOKENS reflector max_tokens            (8192)
  HERMES_AGENTDIET_TIMEOUT    reflector timeout seconds       (180)
  HERMES_AGENTDIET_EXTRA_BODY JSON merged into the request    ({"chat_template_kwargs":{"enable_thinking":false}})
  HERMES_AGENTDIET_PREFILL    1 = assistant prefill via continue_final_message (1)
  HERMES_AGENTDIET_TOKENIZER  server | tiktoken | rough          (server = vLLM /tokenize of the
                              reflector endpoint, i.e. the served model's own tokenizer; falls back
                              to rough 4-chars/token if the endpoint has no /tokenize)
  HERMES_AGENTDIET_CAP        1 = cap reflector max_tokens at the largest ACCEPTABLE size
                              (max(old-min_save, ratio*old) + CAP_MARGIN); finish_reason=length
                              is then a cheap reject instead of a verbatim re-emit          (1)
  HERMES_AGENTDIET_CAP_MARGIN tokens of slack on that cap                                  (64)
  HERMES_AGENTDIET_EVENTS     jsonl path for per-decision events
  HERMES_AGENTDIET_EVENTS_FULL 1 = accepted events carry the ORIGINAL and NEW text of every edited
                              field (message index + field), so a replay can rebuild the pre-edit
                              prompt of every earlier turn                                 (1)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REMINDER = "(System reminder: compressed for better efficiency) "
DELETED = "(System reminder: long content deleted for better efficiency)"

# Verbatim from the artifact (traj_analyzer.py SYS_PROMPT).
SYS_PROMPT = """
You will analyze and compress a given step in a trajectory of an AI agent solving a software bug.

In the trajectory, each step is marked in <step id="..."></step>.
The agent will think in <think>, call external tools as marked in <call tool="..."></call>. Its result is marked in <result></result> within the <step> tag.

Your job is to compress the text within the given id to avoid harming efficiency, typically shortening it to 20%-50% of the original length.
Meanwhile, keep the compressed text useful such that you are able to continue the trajectory as close as the original path.

- You should ONLY remove redundant texts, which are either irrelevant to future steps or duplicated by other texts in the trajectory.
- Replace the text to remove to "..." and a short takeaway, e.g. "... (same as the content below)".
- You should keep the original structure unchanged, e.g., XML tags, Python indentation and line numbers.
- Again, keep useful details in the original content unchanged, e.g., XML tags, Python indentation and line numbers.

Typical examples:
- If the step opens a huge file but only one part is necessary for future steps, replace other parts to "... (unrelated function XXX, YYY)".
- If the step runs a verbose test script and everything goes fine, replace the verbose part to "... (expected output)".
- If the step uses str_replace_editor to modify a file and the content can be inferred by the content after it, replace the tool call argument to "... (see results below)".

You should only process the text within the <step> tag with the given id. STOP OUTPUT IMMEDIATELY AFTER </step>.
""".strip()


# --------------------------------------------------------------------------- config
def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


class Config:
    def __init__(self) -> None:
        self.mode = _env("HERMES_AGENTDIET_MODE", "ours").strip().lower()
        self.theta = int(_env("HERMES_AGENTDIET_THETA", "500"))
        self.ctx_before = int(_env("HERMES_AGENTDIET_CTX_BEFORE", "1"))
        self.ctx_after = int(_env("HERMES_AGENTDIET_CTX_AFTER", "2"))
        self.min_save = int(_env("HERMES_AGENTDIET_MIN_SAVE", "400"))
        self.ratio = float(_env("HERMES_AGENTDIET_RATIO", "0.8"))
        self.writeback = _env("HERMES_AGENTDIET_WRITEBACK", "structured").strip().lower()
        self.model = _env("HERMES_AGENTDIET_MODEL", os.environ.get("HERMES_INFERENCE_MODEL", ""))
        self.base_url = _env("HERMES_AGENTDIET_BASE_URL", os.environ.get("OPENAI_BASE_URL", ""))
        self.api_key = _env("HERMES_AGENTDIET_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.max_tokens = int(_env("HERMES_AGENTDIET_MAX_TOKENS", "8192"))
        self.timeout = float(_env("HERMES_AGENTDIET_TIMEOUT", "180"))
        self.prefill = _env("HERMES_AGENTDIET_PREFILL", "1") == "1"
        self.events_path = os.environ.get("HERMES_AGENTDIET_EVENTS") or ""
        self.events_full = _env("HERMES_AGENTDIET_EVENTS_FULL", "1") == "1"
        self.tokenizer = _env("HERMES_AGENTDIET_TOKENIZER", "server").strip().lower()
        self.cap = _env("HERMES_AGENTDIET_CAP", "1") == "1"
        self.cap_margin = int(_env("HERMES_AGENTDIET_CAP_MARGIN", "64"))
        raw = _env("HERMES_AGENTDIET_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')
        self.extra_body: Dict[str, Any] = json.loads(raw) if raw.strip() else {}
        if self.mode not in ("ours", "delete", "skip"):
            raise ValueError(f"HERMES_AGENTDIET_MODE={self.mode!r} not in ours|delete|skip")
        if self.tokenizer not in ("server", "tiktoken", "rough"):
            raise ValueError(f"HERMES_AGENTDIET_TOKENIZER={self.tokenizer!r} not in server|tiktoken|rough")
        if self.writeback not in ("structured", "faithful"):
            raise ValueError(f"HERMES_AGENTDIET_WRITEBACK={self.writeback!r} not in structured|faithful")
        if self.mode == "ours" and (not self.model or not self.base_url):
            raise ValueError("HERMES_AGENTDIET mode=ours needs HERMES_AGENTDIET_MODEL and HERMES_AGENTDIET_BASE_URL "
                             "(or HERMES_INFERENCE_MODEL / OPENAI_BASE_URL)")


_CFG: Optional[Config] = None


def get_config() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = Config()
        logger.info("AGENTDIET config: mode=%s theta=%d ctx_before=%d ctx_after=%d min_save=%d ratio=%.2f "
                    "writeback=%s model=%s base_url=%s prefill=%s tokenizer=%s cap=%s events=%s",
                    _CFG.mode, _CFG.theta, _CFG.ctx_before, _CFG.ctx_after, _CFG.min_save, _CFG.ratio,
                    _CFG.writeback, _CFG.model, _CFG.base_url, _CFG.prefill, _CFG.tokenizer, _CFG.cap,
                    _CFG.events_path or "-")
    return _CFG


# --------------------------------------------------------------------------- tokens
_ENC = None
_ENC_KIND = ""


_SERVER_TOK_OK: Optional[bool] = None


def _count_tokens_server(cfg: Config, text: str) -> Optional[int]:
    """vLLM /tokenize on the reflector endpoint: the served model's own tokenizer, no extra deps.
    Returns None (and disables itself for the process) if the endpoint does not answer."""
    global _SERVER_TOK_OK
    if _SERVER_TOK_OK is False:
        return None
    import urllib.request
    root = cfg.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    body = json.dumps({"model": cfg.model, "prompt": text, "add_special_tokens": False}).encode()
    req = urllib.request.Request(root + "/tokenize", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {cfg.api_key or 'EMPTY'}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            n = json.loads(r.read().decode()).get("count")
        if not isinstance(n, int):
            raise ValueError(f"no count in /tokenize reply")
        if _SERVER_TOK_OK is None:
            _SERVER_TOK_OK = True
            logger.info("AGENTDIET token counter: server /tokenize at %s (model %s)", root, cfg.model)
        return n
    except Exception as exc:  # noqa: BLE001
        _SERVER_TOK_OK = False
        logger.warning("AGENTDIET /tokenize unavailable at %s (%s); falling back to %s",
                       root, str(exc)[:200], "tiktoken/rough")
        return None


def count_tokens(text: str, cfg: Optional[Config] = None) -> int:
    """Served-model tokenizer via /tokenize when configured; else tiktoken gpt-4o like the
    artifact; else rough 4-chars/token."""
    global _ENC, _ENC_KIND
    if not text:
        return 0
    if cfg is not None and cfg.tokenizer == "server":
        n = _count_tokens_server(cfg, text)
        if n is not None:
            _ENC_KIND = "server"
            return n
    if cfg is not None and cfg.tokenizer == "rough":
        _ENC_KIND = "rough"
        return max(1, len(text) // 4)
    if _ENC is None:
        try:
            import tiktoken  # type: ignore
            _ENC = tiktoken.encoding_for_model("gpt-4o")
            _ENC_KIND = "tiktoken:gpt-4o"
        except Exception as exc:  # noqa: BLE001
            _ENC = False
            _ENC_KIND = "rough"
            logger.warning("AGENTDIET tiktoken unavailable (%s); using rough estimate", exc)
    if _ENC:
        return len(_ENC.encode(text, disallowed_special=()))
    try:
        from agent.model_metadata import estimate_tokens_rough
        return int(estimate_tokens_rough(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


# --------------------------------------------------------------------------- steps
def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def _tc_name_args(tc: Any) -> Tuple[str, str]:
    fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
    if isinstance(fn, dict):
        return str(fn.get("name") or ""), str(fn.get("arguments") or "")
    return str(getattr(fn, "name", "") or ""), str(getattr(fn, "arguments", "") or "")


def group_steps(messages: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """[(start, end)) spans: an assistant message plus the tool messages that follow it."""
    steps: List[Tuple[int, int]] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "assistant":
            j = i + 1
            while j < n and isinstance(messages[j], dict) and messages[j].get("role") == "tool":
                j += 1
            steps.append((i, j))
            i = j
        else:
            i += 1
    return steps


def first_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return _text(m.get("content"))
    return ""


def serialize_step(messages: List[Dict[str, Any]], span: Tuple[int, int], sid: int) -> str:
    a = messages[span[0]]
    out = [f'<step id="{sid}">']
    think = _text(a.get("content")).strip()
    if think:
        out.append(f"<think>\n{think}\n</think>")
    for tc in a.get("tool_calls") or []:
        name, args = _tc_name_args(tc)
        out.append(f'<call tool="{name}">{args.strip()}</call>')
    for k, m in enumerate(messages[span[0] + 1:span[1]]):
        out.append(f'<result idx="{k}">\n{_text(m.get("content"))}\n</result>')
    out.append("</step>")
    return "\n".join(out)


def is_reduced(messages: List[Dict[str, Any]], span: Tuple[int, int]) -> bool:
    a = messages[span[0]]
    if _text(a.get("content")).startswith(("(System reminder: compressed", "(System reminder: long content")):
        return True
    for m in messages[span[0] + 1:span[1]]:
        c = _text(m.get("content"))
        if c.startswith(REMINDER) or c.startswith(DELETED):
            return True
    return False


# --------------------------------------------------------------------------- reflector
_RE_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.S | re.I)
_RE_CALL = re.compile(r'<call\s+tool="([^"]*)">\s*(.*?)\s*</call>', re.S | re.I)
_RE_RESULT = re.compile(r'<result(?:\s+idx="(\d+)")?\s*>\s*(.*?)\s*</result>', re.S | re.I)


def parse_reduced(text: str) -> Dict[str, Any]:
    think = _RE_THINK.search(text)
    calls = _RE_CALL.findall(text)
    results = _RE_RESULT.findall(text)
    return {
        "think": think.group(1) if think else None,
        "calls": [(n, a) for n, a in calls],
        "results": {(int(i) if i else k): body for k, (i, body) in enumerate(results)},
    }


def reflector_cap(cfg: Config, old_tokens: int) -> Optional[int]:
    """Largest output that could still be accepted, plus slack; None when capping is off."""
    if not cfg.cap:
        return None
    max_ok = max(old_tokens - cfg.min_save, int(cfg.ratio * old_tokens))
    return max(1, min(cfg.max_tokens, max_ok + cfg.cap_margin))


def call_reflector(cfg: Config, window_text: str, sid: int, max_tokens: Optional[int] = None
                   ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Return (content_between_step_tags, usage, meta).  Raises on transport failure.
    meta["finish_reason"] == "length" means the reflector hit the cap: treat as a reject."""
    from openai import OpenAI  # present in the Hermes image (main client)

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY", timeout=cfg.timeout, max_retries=1)
    user_prompt = f"{window_text}\n\nNow, compress the step {sid}."
    prefill = f'Sure. Here is the compressed content of step {sid}: <step id="{sid}">'
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user_prompt}]
    extra = dict(cfg.extra_body)
    used_prefill = False
    if cfg.prefill:
        msgs.append({"role": "assistant", "content": prefill})
        extra.update({"continue_final_message": True, "add_generation_prompt": False})
        used_prefill = True

    def _create(m, eb):
        return client.chat.completions.create(
            model=cfg.model, messages=m, temperature=0.0, max_tokens=max_tokens or cfg.max_tokens,
            stop=["</step>"], extra_body=eb or None)

    try:
        resp = _create(msgs, extra)
    except Exception as exc:  # noqa: BLE001 - retry once without the prefill on a 4xx
        if used_prefill and ("400" in str(exc) or "continue_final_message" in str(exc) or "add_generation_prompt" in str(exc)):
            logger.warning("AGENTDIET prefill rejected by endpoint (%s); retrying without prefill", str(exc)[:200])
            msgs = msgs[:2]
            msgs[1]["content"] += f'\nOutput only <step id="{sid}"> ... </step>.'
            extra = dict(cfg.extra_body)
            used_prefill = False
            resp = _create(msgs, extra)
        else:
            raise
    choice = resp.choices[0]
    content = choice.message.content or ""
    finish = getattr(choice, "finish_reason", None)
    usage = {}
    if getattr(resp, "usage", None) is not None:
        usage = {"prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                 "completion_tokens": getattr(resp.usage, "completion_tokens", None)}
    # Same post-processing as the artifact: cut at </step>, strip a leading <step ...>.
    content, splitter, _ = content.partition("</step>")
    if not splitter and finish not in ("stop", "length", None):
        raise RuntimeError(f"invalid reflector output (finish_reason={finish})")
    if "<step" in content[:200]:
        content = content.partition("<step")[2]
        if ">" in content[:40]:
            content = content.partition(">")[2]
    return content.strip(), usage, {"finish_reason": finish, "prefill": used_prefill}


# --------------------------------------------------------------------------- write-back
def _writeback_structured(messages: List[Dict[str, Any]], span: Tuple[int, int], reduced: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite tool results (and JSON-valid tool-call args) in place; keep ids and roles."""
    a = messages[span[0]]
    changed = {"think": False, "calls": 0, "results": 0}
    if reduced["think"] is not None and _text(a.get("content")).strip() and isinstance(a.get("content"), str):
        if len(reduced["think"]) < len(a["content"]):
            a["content"] = reduced["think"]
            changed["think"] = True
    tcs = a.get("tool_calls") or []
    for k, tc in enumerate(tcs):
        if k >= len(reduced["calls"]):
            break
        name, new_args = reduced["calls"][k]
        old_name, old_args = _tc_name_args(tc)
        if name != old_name or not new_args or len(new_args) >= len(old_args):
            continue
        try:
            json.loads(new_args)
        except Exception:  # noqa: BLE001 - never hand the model invalid JSON args
            continue
        fn = tc["function"] if isinstance(tc, dict) else tc.function
        if isinstance(fn, dict):
            fn["arguments"] = new_args
        else:
            fn.arguments = new_args
        changed["calls"] += 1
    tool_msgs = messages[span[0] + 1:span[1]]
    for k, m in enumerate(tool_msgs):
        body = reduced["results"].get(k)
        if body is None:
            continue
        old = _text(m.get("content"))
        if len(body) < len(old):
            m["content"] = (REMINDER if k == 0 else "") + body
            changed["results"] += 1
    if changed["results"] == 0 and tool_msgs:
        # Mark the step as processed even when only args/think shrank, so it is not re-analysed.
        tool_msgs[0]["content"] = REMINDER + _text(tool_msgs[0].get("content"))
    return changed


def _writeback_faithful(messages: List[Dict[str, Any]], span: Tuple[int, int], reduced_text: Optional[str]) -> None:
    """Artifact behaviour: collapse the whole step into one marker-prefixed assistant message."""
    new_msg = {"role": "assistant",
               "content": (REMINDER + reduced_text.strip()) if reduced_text else DELETED}
    messages[span[0]:span[1]] = [new_msg]


# --------------------------------------------------------------------------- snapshots
def _snapshot(messages: List[Dict[str, Any]], span: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Per-message copy of the fields the write-back can touch, keyed by absolute message index,
    so a replay can rebuild the prompt as it was BEFORE (orig) and AFTER (new) this edit."""
    out = []
    for i in range(span[0], min(span[1], len(messages))):
        m = messages[i]
        rec: Dict[str, Any] = {"i": i, "role": m.get("role"), "content": _text(m.get("content"))}
        if m.get("tool_call_id"):
            rec["tool_call_id"] = m.get("tool_call_id")
        if m.get("tool_calls"):
            rec["tool_calls"] = [{"id": (tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)),
                                  "name": _tc_name_args(tc)[0], "arguments": _tc_name_args(tc)[1]}
                                 for tc in m["tool_calls"]]
        out.append(rec)
    return out


# --------------------------------------------------------------------------- events
def _emit(cfg: Config, agent: Any, rec: Dict[str, Any]) -> None:
    rec.setdefault("ts", time.time())
    rec.setdefault("session_id", getattr(agent, "session_id", None))
    st = getattr(agent, "_agentdiet", None)
    if st is None:
        st = agent._agentdiet = {"analysis_count": 0, "erase_count": 0, "erase_in_tokens": 0, "erase_out_tokens": 0,
                                 "reflect_prompt_tokens": 0, "reflect_completion_tokens": 0, "errors": 0}
    kind = rec.get("kind")
    if kind in ("accepted", "rejected", "error"):
        st["analysis_count"] += 1
    if kind == "accepted":
        st["erase_count"] += 1
        st["erase_in_tokens"] += rec.get("old_tokens", 0)
        st["erase_out_tokens"] += rec.get("new_tokens", 0)
    if kind == "error":
        st["errors"] += 1
    u = rec.get("reflect_usage") or {}
    st["reflect_prompt_tokens"] += u.get("prompt_tokens") or 0
    st["reflect_completion_tokens"] += u.get("completion_tokens") or 0
    logger.info("AGENTDIET %s", json.dumps(rec, default=str)[:600])
    if cfg.events_path:
        os.makedirs(os.path.dirname(cfg.events_path) or ".", exist_ok=True)
        with open(cfg.events_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


# --------------------------------------------------------------------------- entry point
def maybe_reduce_step(agent: Any, messages: List[Dict[str, Any]]) -> None:
    """Called once per tool round, after the round's tool results are appended."""
    cfg = get_config()
    if cfg.mode == "skip":
        return
    steps = group_steps(messages)
    t = len(steps) - 1 - cfg.ctx_after
    # artifact: `if turn < N_CTX_BEFORE + N_CTX_AFTER: return`; the window may
    # reach back to the user prompt (its step -1) when t - ctx_before < 0.
    if t < 0 or len(steps) < cfg.ctx_before + cfg.ctx_after:
        return
    span = steps[t]
    if span[1] - span[0] < 2:
        return  # assistant-only step (no tool results): nothing to reduce safely
    if is_reduced(messages, span):
        return
    target_text = serialize_step(messages, span, t)
    old_tokens = count_tokens(target_text, cfg)
    api_call = getattr(agent, "_api_call_count", None)
    if old_tokens < cfg.theta:
        _emit(cfg, agent, {"kind": "skip_short", "step": t, "old_tokens": old_tokens, "msg_span": list(span),
                           "api_call": api_call})
        return

    if cfg.mode == "delete":
        orig = _snapshot(messages, span) if cfg.events_full else None
        if cfg.writeback == "faithful":
            _writeback_faithful(messages, span, None)
        else:
            for m in messages[span[0] + 1:span[1]]:
                m["content"] = DELETED
        _emit(cfg, agent, {"kind": "accepted", "mode": "delete", "step": t, "msg_span": list(span),
                           "old_tokens": old_tokens, "new_tokens": 0, "writeback": cfg.writeback,
                           "api_call": api_call, "orig": orig})
        return

    lo = t - cfg.ctx_before
    window: List[str] = []
    if lo < 0:
        window.append(first_user_text(messages))
    for i in range(max(0, lo), min(len(steps), t + cfg.ctx_after + 1)):
        window.append(serialize_step(messages, steps[i], i))
    cap = reflector_cap(cfg, old_tokens)
    t0 = time.time()
    try:
        reduced_text, usage, meta = call_reflector(cfg, "\n".join(window), t, max_tokens=cap)
    except Exception as exc:  # noqa: BLE001 - reflector trouble must not kill the run
        _emit(cfg, agent, {"kind": "error", "step": t, "msg_span": list(span), "old_tokens": old_tokens,
                           "api_call": api_call, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                           "latency_ms": int((time.time() - t0) * 1000)})
        return
    latency_ms = int((time.time() - t0) * 1000)
    new_tokens = count_tokens(reduced_text, cfg)
    accept = (old_tokens - new_tokens >= cfg.min_save) or (new_tokens < cfg.ratio * old_tokens)
    rec = {"step": t, "msg_span": list(span), "old_tokens": old_tokens, "new_tokens": new_tokens,
           "reflect_usage": usage, "latency_ms": latency_ms, "token_counter": _ENC_KIND,
           "api_call": api_call, "max_tokens_cap": cap, **meta}
    if meta.get("finish_reason") == "length":
        _emit(cfg, agent, {"kind": "rejected", "reason": "over_cap", **rec})
        return
    if not accept:
        _emit(cfg, agent, {"kind": "rejected", "reason": "saving_below_rule", **rec})
        return
    orig = _snapshot(messages, span) if cfg.events_full else None
    if cfg.writeback == "faithful":
        _writeback_faithful(messages, span, reduced_text)
        rec["writeback"] = "faithful"
    else:
        parsed = parse_reduced(reduced_text)
        if not parsed["results"] and parsed["think"] is None and not parsed["calls"]:
            _emit(cfg, agent, {"kind": "rejected", "reason": "unparseable", **rec})
            return
        rec["writeback"] = "structured"
        rec["changed"] = _writeback_structured(messages, span, parsed)
    if cfg.events_full:
        rec["orig"] = orig
        rec["new"] = _snapshot(messages, (span[0], span[0] + (1 if cfg.writeback == "faithful" else span[1] - span[0])))
    # The edited prefix is a new prompt from here on: every later turn re-prefills from span[0].
    _emit(cfg, agent, {"kind": "accepted", "mode": "ours", **rec})
