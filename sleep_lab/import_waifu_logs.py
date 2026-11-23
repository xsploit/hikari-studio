#!/usr/bin/env python3
"""Import Hikari's real Web Waifu memory logs into SLEEP-lab datasets.

Reads (read-only) the `ladybug-memory.db.json` snapshot stores written by
webwaifu-rebuild and web-waifu-4, plus Hikari's verbatim persona from
`modules/ai_companion.py`, and writes four JSONL files to `sleep_lab/datasets/`:

  wake_cycle_1.jsonl      first chronological half of episodes/facts (wake buffer #1)
  wake_cycle_2.jsonl      second chronological half (wake buffer #2)
  eval_recall_quiz.jsonl  held-out zero-context probe prompts, tagged cycle 1 / 2
  persona_anchor.jsonl    verbatim persona system prompt + real in-voice reply anchors

plus `datasets/manifest.json` (counts and source hashes only, no content).

Snapshot schema handled (`{"reason", "snapshots": {key: {kind, scopeKey, updatedAt, value}}, ...}`):
  kind=grillo-records   key grillo-records:<table>  value=list of rows
      turn_events / evidence_records  -> conversation turns (deduped by turn id)
      memory_blocks / memory_slots / memory_slot_patches / memory_candidates / diary_entries -> facts
      grillo_activity_log / worker_context_traces -> skipped (worker internals, prompts)
  kind=semantic         value=list of {userText, assistantText, createdAt, embedding...} -> exchanges
  kind=grillo           value={blocks, candidates, diaryEntries, ...} -> facts
  kind=relationships    value={participantKey: {facts, summary, diaryEntry, diaryHistory, stage...}}
  kind=grillo-singleton -> skipped (worker status)

Privacy: nothing private is printed. Only counts are logged. API keys / tokens /
emails / IPs are redacted and long numeric platform IDs are replaced with short
stable hashes before anything is written.

Usage:
  python sleep_lab/import_waifu_logs.py            # default paths
  python sleep_lab/import_waifu_logs.py --include-test --out-dir other/dir
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
STUDIO = HERE.parent
APPDATA = Path.home() / "AppData" / "Roaming"

DEFAULT_SOURCES = [
    APPDATA / "webwaifu-rebuild" / "ladybug-memory.db.json",
    APPDATA / "web-waifu-4" / "ladybug-memory.db.json",
]
DEFAULT_PERSONA_FILE = STUDIO / "modules" / "ai_companion.py"
DEFAULT_OUT = HERE / "datasets"

PAIR_WINDOW_MS = 10 * 60 * 1000      # user turn -> assistant reply must be within 10 min
SESSION_GAP_MS = 30 * 60 * 1000      # split snaps to a >30 min gap if one is near the middle

# --------------------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------------------
_SECRET_PATTERNS = [
    (re.compile(r"\bsk-(?:or-v1-|proj-|ant-)?[A-Za-z0-9_\-]{16,}"), "[REDACTED_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), "[REDACTED_KEY]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}"), "[REDACTED_KEY]"),
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"), "[REDACTED_KEY]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "[REDACTED_KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)\b(bearer|token|api[_-]?key|secret|password|oauth)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"\b[MN][A-Za-z\d]{23,25}\.[\w\-]{6}\.[\w\-]{27,38}\b"), "[REDACTED_DISCORD_TOKEN]"),
    (re.compile(r"\b[a-f0-9]{32,}\b", re.I), "[REDACTED_HEX]"),
    (re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
]
_SNOWFLAKE = re.compile(r"\b\d{15,21}\b")


def short_hash(value: str, n: int = 8) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def redact(text: str) -> tuple[str, int]:
    """Return (clean_text, number_of_redactions)."""
    hits = 0
    for pat, repl in _SECRET_PATTERNS:
        text, n = pat.subn(repl, text)
        hits += n
    text, n = _SNOWFLAKE.subn(lambda m: f"id{short_hash(m.group(0), 6)}", text)
    return text, hits + n


# --------------------------------------------------------------------------------------
# Content cleaning (strip the bot's routing prefix / metadata suffix)
# --------------------------------------------------------------------------------------
_PREFIX = re.compile(
    r"^\s*(?:Discord|Twitch|Local|YouTube|Kick)\s+(?:trusted\s+)?(?:controller|viewer|user|chatter|mod|broadcaster)?\s*"
    r"(?P<name>[^:\n]{1,40}):\s*",
    re.I,
)
_META = re.compile(r"\s+metadata:\s+source=.*$", re.S)
_USER_LINE = re.compile(r"^User:\s*", re.I)


def clean_utterance(raw: str) -> tuple[str, str | None]:
    """Strip "Discord trusted controller NAME:" prefix and " metadata: ..." suffix."""
    if not isinstance(raw, str):
        return "", None
    text = raw.replace(" ", " ").strip()
    text = _META.sub("", text)
    speaker = None
    m = _PREFIX.match(text)
    if m:
        speaker = m.group("name").strip()
        text = text[m.end():]
    return re.sub(r"[ \t]+", " ", text).strip(), speaker


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def to_ms(value: Any) -> int | None:
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v if v > 10**11 else v * 1000  # seconds -> ms


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def channel_label(scope: str, source: str | None = None) -> str:
    s = f"{scope} {source or ''}".lower()
    if "discord" in s and "voice" in s:
        return "Discord voice"
    if "discord" in s:
        return "Discord"
    if "twitch" in s:
        return "Twitch"
    return "local studio"


def is_test_scope(*keys: str) -> bool:
    joined = " ".join(k for k in keys if k).lower()
    return any(t in joined for t in ("test", "restart", "neuro-sama", "recovery"))


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------
@dataclass
class Turn:
    turn_id: str
    role: str
    text: str
    speaker: str | None
    ts: int
    scope: str
    source: str
    origin: str


@dataclass
class Episode:
    kind: str                 # exchange | fact
    ts: int
    origin: str
    channel: str
    text: str                 # rendered memory line for the wake context buffer
    user: str | None = None
    assistant: str | None = None
    speaker: str | None = None
    fact_type: str | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        if self.kind == "exchange":
            return "x:" + norm_key(self.user or "")[:200] + "|" + norm_key(self.assistant or "")[:120]
        return "f:" + norm_key(self.text_core)

    @property
    def text_core(self) -> str:
        return self.assistant if self.kind == "fact" and self.assistant else (self.user or self.text)


@dataclass
class Stats:
    redactions: int = 0
    skipped_test: int = 0
    turns_raw: int = 0
    turns_dedup: int = 0
    semantic_raw: int = 0
    facts_raw: int = 0
    by_origin: Counter = field(default_factory=Counter)


class Importer:
    def __init__(self, include_test: bool = False):
        self.include_test = include_test
        self.stats = Stats()
        self.turns: dict[str, Turn] = {}
        self.exchanges: list[Episode] = []
        self.facts: list[Episode] = []

    # ---- helpers -------------------------------------------------------------------
    def _clean(self, text: Any) -> str:
        if not isinstance(text, str):
            return ""
        out, n = redact(text.strip())
        self.stats.redactions += n
        return out

    def _skip(self, *keys: str) -> bool:
        if not self.include_test and is_test_scope(*keys):
            self.stats.skipped_test += 1
            return True
        return False

    def _fact(self, content: Any, ts: Any, origin: str, scope: str, fact_type: str, srcs: Iterable[str] = ()):
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        text = self._clean(content)
        text = clean_utterance(text)[0] if text else ""
        if len(text) < 6:
            return
        ms = to_ms(ts) or 0
        self.stats.facts_raw += 1
        label = fact_type.replace("_", " ")
        self.facts.append(Episode(
            kind="fact", ts=ms, origin=origin, channel=channel_label(scope),
            text=f"Memory note ({label}, {channel_label(scope)}): {text}",
            assistant=text, fact_type=fact_type, sources=list(srcs)[:8],
        ))

    # ---- snapshot parsers ----------------------------------------------------------
    def load(self, path: Path) -> None:
        origin = path.parent.name
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        snaps = data.get("snapshots") if isinstance(data, dict) else None
        if not isinstance(snaps, dict):
            raise ValueError(f"{path}: not a ladybug snapshot store (missing 'snapshots')")
        for key, snap in snaps.items():
            if not isinstance(snap, dict):
                continue
            kind = snap.get("kind") or key.split(":", 1)[0]
            value = snap.get("value")
            scope = str(snap.get("scopeKey") or "")
            updated = snap.get("updatedAt")
            if kind == "grillo-records":
                self._records(key.split(":", 1)[-1], value or [], origin, updated)
            elif kind == "semantic":
                if self._skip(key, scope):
                    continue
                self._semantic(value or [], origin, scope)
            elif kind == "grillo":
                if self._skip(key, scope):
                    continue
                self._grillo(value or {}, origin, scope)
            elif kind == "relationships":
                self._relationships(value or {}, origin, updated)
            self.stats.by_origin[origin] += 1

    def _records(self, table: str, rows: list, origin: str, updated: Any) -> None:
        for r in rows:
            if not isinstance(r, dict):
                continue
            scope = str(r.get("scope_key") or r.get("scopeKey") or "")
            if self._skip(scope):
                continue
            ts = r.get("created_at") or r.get("createdAt") or r.get("updated_at") or updated
            if table in ("turn_events", "evidence_records"):
                tid = str(r.get("turn_id") or r.get("id") or "")
                role = str(r.get("role") or "")
                if not tid or role not in ("user", "assistant"):
                    continue
                self.stats.turns_raw += 1
                if tid in self.turns:
                    continue
                text, speaker = clean_utterance(self._clean(r.get("content")))
                meta = r.get("metadata") or {}
                speaker = speaker or r.get("author_name") or meta.get("authorName")
                self.turns[tid] = Turn(tid, role, text, speaker, to_ms(ts) or 0, scope,
                                       str(r.get("source") or ""), origin)
            elif table == "memory_blocks":
                for item in r.get("items") or _json_list(r.get("items_json")):
                    self._fact(item, ts, origin, scope, str(r.get("block_name") or "memory_block"))
            elif table == "memory_slots":
                for item in _json_list(r.get("content_json")):
                    self._fact(item, ts, origin, scope, str(r.get("slot_name") or "memory_slot"))
            elif table == "memory_slot_patches":
                patch = _json_obj(r.get("patch_json"))
                for item in patch.get("items", []) if isinstance(patch, dict) else []:
                    self._fact(item, ts, origin, scope, str(r.get("slot_name") or "slot_patch"))
            elif table == "memory_candidates":
                self._fact(r.get("content"), ts, origin, scope, str(r.get("type") or "candidate"),
                           r.get("evidence_turn_ids") or [])
                if r.get("summary"):
                    self._fact(r.get("summary"), ts, origin, scope, "candidate_summary")
            elif table == "diary_entries":
                for fld in ("summary", "personal_thought", "interaction_summary"):
                    if r.get(fld):
                        self._fact(r[fld], ts, origin, scope, f"diary_{fld}", r.get("source_turn_ids") or [])
            # grillo_activity_log / worker_context_traces / others: worker internals -> skipped

    def _semantic(self, rows: list, origin: str, scope: str) -> None:
        for r in rows:
            if not isinstance(r, dict):
                continue
            self.stats.semantic_raw += 1
            user_raw = r.get("userText")
            if not user_raw and isinstance(r.get("text"), str):
                user_raw = _USER_LINE.sub("", r["text"])
            user, speaker = clean_utterance(self._clean(user_raw))
            asst = self._clean(r.get("assistantText"))
            if not user and not asst:
                continue
            ms = to_ms(r.get("createdAt")) or 0
            sc = str(r.get("scopeKey") or scope)
            self.exchanges.append(_exchange(user, asst, speaker, ms, origin, sc, list(r.get("sourceTurnIds") or [])))

    def _grillo(self, value: dict, origin: str, scope: str) -> None:
        for b in value.get("blocks") or []:
            for item in b.get("items") or []:
                self._fact(item, b.get("updatedAt") or b.get("createdAt"), origin, scope,
                           str(b.get("blockName") or "memory_block"), b.get("sourceCandidateIds") or [])
        for c in value.get("candidates") or []:
            self._fact(c.get("content"), c.get("createdAt"), origin, scope, str(c.get("type") or "candidate"),
                       c.get("sourceTurnIds") or [])
            if c.get("summary"):
                self._fact(c["summary"], c.get("createdAt"), origin, scope, "candidate_summary")
        for d in value.get("diaryEntries") or []:
            for fld in ("summary", "personalThought", "interactionSummary"):
                if d.get(fld):
                    self._fact(d[fld], d.get("createdAt"), origin, scope, f"diary_{_snake(fld)}",
                               d.get("sourceTurnIds") or [])

    def _relationships(self, value: dict, origin: str, updated: Any) -> None:
        for pkey, rel in value.items():
            if not isinstance(rel, dict) or self._skip(pkey):
                continue
            ts = rel.get("lastSeenAt") or updated
            for f in rel.get("facts") or []:
                self._fact(f.get("content") if isinstance(f, dict) else f, ts, origin, pkey, "user_said")
            if rel.get("summary"):
                self._fact(rel["summary"], ts, origin, pkey, "relationship_summary")
            if rel.get("diaryEntry"):
                self._fact(rel["diaryEntry"], ts, origin, pkey, "relationship_diary")
            for d in rel.get("diaryHistory") or []:
                self._fact(d.get("entry") or d.get("content") if isinstance(d, dict) else d, ts, origin, pkey,
                           "relationship_diary")
            stage, mood = rel.get("relationshipStage"), rel.get("mood")
            if stage or mood:
                dims = {k: rel.get(k) for k in ("trust", "attraction", "respect", "irritation", "jealousy", "guard")
                        if isinstance(rel.get(k), (int, float))}
                dim_txt = ", ".join(f"{k} {v}" for k, v in dims.items())
                self._fact(
                    f"Hikari-chan's relationship with SUBSECT on {channel_label(pkey)} is at the '{stage}' stage, "
                    f"mood '{mood}' ({dim_txt}).",
                    ts, origin, pkey, "relationship_state",
                )

    # ---- assemble ------------------------------------------------------------------
    def pair_turns(self) -> None:
        turns = sorted(self.turns.values(), key=lambda t: (t.scope, t.ts, 0 if t.role == "user" else 1))
        self.stats.turns_dedup = len(turns)
        i = 0
        while i < len(turns):
            t = turns[i]
            if t.role == "user":
                nxt = turns[i + 1] if i + 1 < len(turns) else None
                if nxt and nxt.role == "assistant" and nxt.scope == t.scope and 0 <= nxt.ts - t.ts <= PAIR_WINDOW_MS:
                    self.exchanges.append(_exchange(t.text, nxt.text, t.speaker, t.ts, t.origin, t.scope,
                                                    [t.turn_id, nxt.turn_id]))
                    i += 2
                    continue
                self.exchanges.append(_exchange(t.text, "", t.speaker, t.ts, t.origin, t.scope, [t.turn_id]))
            else:  # orphan assistant line
                self.exchanges.append(_exchange("", t.text, None, t.ts, t.origin, t.scope, [t.turn_id]))
            i += 1

    def episodes(self) -> list[Episode]:
        merged: dict[str, Episode] = {}
        for ep in self.exchanges + self.facts:
            if not (ep.user or ep.assistant):
                continue
            k = ep.key
            if k in merged:  # keep earliest timestamp, union sources
                old = merged[k]
                if ep.ts and (not old.ts or ep.ts < old.ts):
                    old.ts = ep.ts
                old.sources = sorted(set(old.sources) | set(ep.sources))[:8]
                continue
            merged[k] = ep
        eps = [e for e in merged.values()]
        # facts with no timestamp sort just after the newest event of the same origin file
        latest = Counter()
        for e in eps:
            latest[e.origin] = max(latest[e.origin], e.ts)
        for e in eps:
            if not e.ts:
                e.ts = latest[e.origin] or 0
        eps.sort(key=lambda e: (e.ts, 0 if e.kind == "exchange" else 1, e.key))
        return eps


def _exchange(user: str, asst: str, speaker: str | None, ms: int, origin: str, scope: str, srcs: list) -> Episode:
    who = speaker or "SUBSECT"
    chan = channel_label(scope)
    when = iso(ms) if ms else "unknown time"
    parts = [f"[{when}] ({chan})"]
    if user:
        parts.append(f"{who}: {user}")
    if asst:
        parts.append(f"Hikari-chan: {asst}")
    return Episode(kind="exchange", ts=ms, origin=origin, channel=chan, text="\n".join(parts),
                   user=user or None, assistant=asst or None, speaker=who, sources=[str(s) for s in srcs][:8])


def _json_list(raw: Any) -> list:
    v = _json_obj(raw)
    return v if isinstance(v, list) else []


def _json_obj(raw: Any) -> Any:
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw) if isinstance(raw, str) and raw.strip() else None
    except json.JSONDecodeError:
        return None


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# --------------------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------------------
def split_index(eps: list[Episode]) -> int:
    """Middle index, snapped to a session gap (>30 min) if one lies within +-10% of the middle."""
    n = len(eps)
    mid = n // 2
    window = max(1, n // 10)
    best = None
    for i in range(max(1, mid - window), min(n - 1, mid + window) + 1):
        gap = eps[i].ts - eps[i - 1].ts
        if gap >= SESSION_GAP_MS and (best is None or abs(i - mid) < abs(best - mid)):
            best = i
    return best if best is not None else mid


# --------------------------------------------------------------------------------------
# Held-out recall probes (cloze over salient spans + relational state probes)
# --------------------------------------------------------------------------------------
_STOP = set("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just let me more most my myself no nor not now of off on
once only or other our ours out over own really right same she should so some such than that the their theirs them then
there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yeah yes okay ok like gonna wanna got get gets thing things stuff something
anything kind sort bro dog dude man homie lol im i'm it's that's don't can't you're hikari hikari-chan subsect subby
said say says going go know think mean want well also still even much many way time memory note""".split())

_SPAN = re.compile(
    r"(?:\b[A-Z][\w.\-]*(?:\s+(?:[A-Z][\w.\-]*|\d[\w.\-]*)){0,3})"  # Proper Noun Phrase (DeepSeek V4 Pro)
    r"|(?:\b\d[\w.\-]*\b)"                                           # numbers / 720p / 3.1
)


_SELF_NAMES = {"i", "subsect", "subby", "hikari", "hikari-chan", "discord", "twitch", "memory", "note", "user",
               "okay", "ok", "oh", "hey", "yo", "bro", "dude", "dog", "homie", "man", "lol", "yeah", "nah", "uh", "um"}


class SpanPicker:
    """Choose the blank for a cloze probe, using corpus statistics.

    Tier A (kept first): numbers / 720p / 3.1, CamelCase or mixed-case tokens (DeepSeek, WebWifu4),
    multi-word capitalised names, and capitalised words that are never written lower-case in the corpus.
    Tier B: rare long content words (>= 7 letters, seen in <= 2 episodes).
    Anything weaker (sentence-initial filler, common verbs) is skipped, so a memory with no specific
    detail simply gets no probe.
    """

    def __init__(self, corpus: Iterable[str]):
        self.df: Counter = Counter()
        self.lower_seen: set[str] = set()
        for text in corpus:
            words = re.findall(r"[A-Za-z][\w\-']*", text or "")
            self.df.update({w.lower() for w in words})
            self.lower_seen.update(w for w in words if w[0].islower())

    def _proper(self, word: str) -> bool:
        w = word.strip(".,'\"")
        if not w or w.lower() in _SELF_NAMES or w.lower() in _STOP:
            return False
        mixed = any(c.isupper() for c in w[1:]) or any(c.isdigit() for c in w)
        return mixed or (w[0].isupper() and w.lower() not in self.lower_seen)

    def pick(self, sentence: str) -> tuple[str, str] | None:
        best: tuple[int, str] | None = None
        for m in _SPAN.finditer(sentence):
            span = m.group(0).strip(" .,-'\"")
            words = span.split()
            if not span or len(span) > 40:
                continue
            has_digit = any(c.isdigit() for c in span)
            proper = [w for w in words if self._proper(w)]
            if not has_digit and not proper:
                continue
            if not has_digit and len(words) == 1 and m.start() == 0 and not any(c.isupper() for c in span[1:]):
                continue  # sentence-initial capital on an ordinary word
            while len(words) > 1 and not self._proper(words[0]) and not any(c.isdigit() for c in words[0]):
                words = words[1:]  # "Implement Grillo" -> "Grillo"
            score = 10 + 3 * len(proper) + (4 if has_digit else 0)
            if best is None or score > best[0]:
                best = (score, " ".join(words))
        if best:
            return best[1], "A"
        toks = [w.strip(".,!?\"'()[]") for w in sentence.split()][1:]  # never the sentence-initial word
        rare = [w for w in toks if len(w) >= 7 and w.isalpha() and w.lower() not in _STOP
                and w.lower() not in _GENERIC and not w.lower().endswith("ly") and self.df[w.lower()] <= 2]
        if rare:
            return min(rare, key=lambda w: (self.df[w.lower()], -len(w))), "B"
        return None


_GENERIC = set("""example examples exchange exchanges appreciation appreciates appreciate interaction interactions
reflection reflects reflecting moment moments something someone everyone anything nothing another however because
continued continuing continue consolidating consolidated important importance feeling feelings thought thoughts
greeting greetings message messages conversation conversations playful lighthearted friendly engaging atmosphere
relationship relationships between without through actually probably definitely literally""".split())

_CHANNEL_PHRASE = {"Discord voice": "in Discord voice chat", "Discord": "on Discord", "Twitch": "on Twitch stream",
                   "local studio": "in the local studio"}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if 25 <= len(p.strip()) <= 260]


def keywords(text: str, k: int = 6) -> list[str]:
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z0-9][\w.\-']+", text):
        lw = w.lower().strip(".'")
        if lw in _STOP or len(lw) < 3 or lw in seen:
            continue
        seen.add(lw)
        out.append(lw)
    return out[:k]


def build_probes(eps: list[Episode], cycle: int, max_probes: int, picker: SpanPicker) -> list[dict]:
    """Held-out probes: the probe *wording* never appears in training data; the answer is a detail that
    only exists in this cycle's wake buffer, so it is answerable with zero context only if it was
    consolidated into the weights."""
    probes: list[dict] = []
    used_answers: set[str] = set()
    for ep in eps:
        where = _CHANNEL_PHRASE.get(ep.channel, f"on {ep.channel}")
        if ep.kind == "fact" and ep.fact_type == "relationship_state":
            m = re.search(r"'(?P<stage>[^']+)' stage, mood '(?P<mood>[^']+)'", ep.text)
            if m:
                probes.append(_probe(cycle, ep, "relational", "A",
                                     f"Without looking anything up: where does your relationship with SUBSECT "
                                     f"stand {where} - what stage is it at, and what's your mood toward them?",
                                     f"{m.group('stage')} stage, {m.group('mood')} mood",
                                     [m.group("stage").lower(), m.group("mood").lower()]))
            continue
        source_text = (ep.user if ep.kind == "exchange" else ep.assistant) or ""
        best = None
        for sent in sentences(source_text):
            got = picker.pick(sent)
            if not got or norm_key(got[0]) in used_answers:
                continue
            if best is None or (got[1] == "A" and best[2] == "B"):
                best = (sent, got[0], got[1])
            if got[1] == "A":
                break
        if not best:
            continue
        sent, span, tier = best
        idx = sent.find(span)
        masked = sent[:idx] + "____" + sent[idx + len(span):]
        if ep.kind == "exchange":
            q = (f"Earlier {where}, SUBSECT told you: \"{masked}\" "
                 f"What was the missing part (____)? Answer from memory.")
            ptype = "episode_cloze"
        else:
            q = (f"From your long-term memory about SUBSECT and your streams, fill in the blank: "
                 f"\"{masked}\" Answer with just the missing words.")
            ptype = "fact_cloze"
        probes.append(_probe(cycle, ep, ptype, tier, q, span, keywords(span, 4) or [span.lower()]))
        used_answers.add(norm_key(span))
    # Tier A first, then tier B; keep chronological order in the file
    ranked = sorted(range(len(probes)), key=lambda i: (probes[i]["tier"], i))[:max_probes]
    return [probes[i] for i in sorted(ranked)]


def _probe(cycle: int, ep: Episode, ptype: str, tier: str, question: str, answer: str, kw: list[str]) -> dict:
    pid = "q_" + short_hash(f"{cycle}|{ptype}|{question}", 10)
    return {
        "id": pid, "cycle": cycle, "probe_type": ptype, "tier": tier, "question": question, "answer": answer,
        "answer_keywords": kw, "source_episode": ep_id(ep), "ts": ep.ts, "zero_context": True,
    }


def ep_id(ep: Episode) -> str:
    return "ep_" + short_hash(ep.key, 12)


# --------------------------------------------------------------------------------------
# Persona anchors
# --------------------------------------------------------------------------------------
def load_persona(path: Path) -> dict:
    """Parse HIKARI_PERSONALITY from ai_companion.py with ast (no import / no side effects)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target.id], node.value
        if "HIKARI_PERSONALITY" in targets and value is not None:
            persona = ast.literal_eval(value)
            if not isinstance(persona, dict) or "system_prompt" not in persona:
                break
            return persona
    raise ValueError(f"HIKARI_PERSONALITY with a system_prompt not found in {path}")


def build_persona_anchors(persona: dict, eps: list[Episode], n: int) -> list[dict]:
    rows = [{
        "id": "persona_system", "kind": "persona_system", "persona_id": persona.get("id"),
        "name": persona.get("name"), "system_prompt": persona["system_prompt"], "source": "modules/ai_companion.py",
    }]
    good = [e for e in eps if e.kind == "exchange" and e.user and e.assistant
            and 40 <= len(e.assistant) <= 600 and len(e.user) >= 8]
    # spread anchors across the whole timeline
    step = max(1, len(good) // max(1, n))
    for e in good[::step][:n]:
        rows.append({"id": "anchor_" + short_hash(e.key, 10), "kind": "style_anchor",
                     "system": persona["system_prompt"], "user": e.user, "assistant": e.assistant,
                     "channel": e.channel, "ts": e.ts})
    return rows


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def episode_row(ep: Episode, cycle: int) -> dict:
    return {
        "id": ep_id(ep), "cycle": cycle, "kind": ep.kind, "ts": ep.ts, "time": iso(ep.ts) if ep.ts else None,
        "origin_file": ep.origin, "channel": ep.channel, "speaker": ep.speaker, "fact_type": ep.fact_type,
        "user": ep.user, "assistant": ep.assistant, "text": ep.text,
        "source_ids": [("src_" + short_hash(s, 10)) for s in ep.sources],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", action="append", type=Path, help="ladybug-memory.db.json (repeatable)")
    ap.add_argument("--persona-file", type=Path, default=DEFAULT_PERSONA_FILE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-test", action="store_true", help="keep test/restart/neuro-sama scopes")
    ap.add_argument("--max-probes-per-cycle", type=int, default=40)
    ap.add_argument("--persona-anchors", type=int, default=16)
    args = ap.parse_args(argv)

    sources = args.source or DEFAULT_SOURCES
    imp = Importer(include_test=args.include_test)
    loaded = []
    for src in sources:
        if not src.exists():
            print(f"[import] WARNING missing source: {src.parent.name}/{src.name}", file=sys.stderr)
            continue
        imp.load(src)
        loaded.append(src)
    if not loaded:
        print("[import] no sources found", file=sys.stderr)
        return 2
    imp.pair_turns()
    eps = imp.episodes()
    if len(eps) < 4:
        print(f"[import] only {len(eps)} episodes - nothing to split", file=sys.stderr)
        return 3

    cut = split_index(eps)
    cycles = {1: eps[:cut], 2: eps[cut:]}
    out = args.out_dir
    for c, rows in cycles.items():
        write_jsonl(out / f"wake_cycle_{c}.jsonl", [episode_row(e, c) for e in rows])

    picker = SpanPicker([e.user or "" for e in eps] + [e.assistant or "" for e in eps])
    probes = []
    for c, rows in cycles.items():
        probes += build_probes(rows, c, args.max_probes_per_cycle, picker)
    write_jsonl(out / "eval_recall_quiz.jsonl", probes)

    persona = load_persona(args.persona_file)
    anchors = build_persona_anchors(persona, eps, args.persona_anchors)
    write_jsonl(out / "persona_anchor.jsonl", anchors)

    def summary(rows):
        k = Counter(e.kind for e in rows)
        chars = sum(len(e.text) for e in rows)
        return {"episodes": len(rows), "exchanges": k["exchange"], "facts": k["fact"],
                "context_chars": chars, "approx_tokens": chars // 4,
                "first": iso(rows[0].ts) if rows and rows[0].ts else None,
                "last": iso(rows[-1].ts) if rows and rows[-1].ts else None,
                "by_channel": dict(Counter(e.channel for e in rows))}

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper": "arXiv:2606.03979 - Language Models Need Sleep (Behrouz, Hashemi, Javanmard, Mirrokni)",
        "sources": [{"file": f"{p.parent.name}/{p.name}", "bytes": p.stat().st_size, "sha256": file_sha(p)} for p in loaded],
        "persona_file_sha256": file_sha(args.persona_file),
        "include_test": args.include_test,
        "raw": {"turn_rows": imp.stats.turns_raw, "unique_turns": imp.stats.turns_dedup,
                "semantic_rows": imp.stats.semantic_raw, "fact_rows": imp.stats.facts_raw,
                "skipped_test_rows": imp.stats.skipped_test, "redactions": imp.stats.redactions},
        "deduped_episodes": len(eps),
        "cycles": {str(c): summary(r) for c, r in cycles.items()},
        "eval_probes": dict(Counter(f"cycle_{p['cycle']}:{p['probe_type']}" for p in probes)),
        "persona_anchor_rows": len(anchors),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # counts only - never content
    print(f"[import] sources loaded: {len(loaded)} ({', '.join(p.parent.name for p in loaded)})")
    r = manifest["raw"]
    print(f"[import] turn rows {r['turn_rows']} -> unique turns {r['unique_turns']}; semantic rows "
          f"{r['semantic_rows']}; fact rows {r['fact_rows']}; skipped test rows {r['skipped_test_rows']}; "
          f"redactions {r['redactions']}")
    print(f"[import] deduped episodes: {len(eps)} (split at index {cut})")
    for c, s in manifest["cycles"].items():
        print(f"[import] wake_cycle_{c}: {s['episodes']} episodes ({s['exchanges']} exchanges, {s['facts']} facts), "
              f"~{s['approx_tokens']} tokens, {s['first']} -> {s['last']}")
    print(f"[import] eval probes: {len(probes)} {manifest['eval_probes']}")
    print(f"[import] persona anchors: {len(anchors)} rows (1 verbatim system prompt, {len(anchors) - 1} style anchors)")
    print(f"[import] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
