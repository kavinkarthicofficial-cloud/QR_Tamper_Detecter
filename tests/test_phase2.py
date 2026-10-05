"""Tests for the Investigator agent, the photo-quality gate and the decision rules (no network needed)."""

import random

import numpy as np
import pytest

from agent_eval import CASES
from qrguard import quality, synth
from qrguard.agent import (BackendError, GeminiBackend, TemplateBackend, investigate, parse_answer, template_answer,
                           validate)
from qrguard.config import DEFAULT_CHECKPOINT
from qrguard.evidence import hot_region, safe_text, url_features


# --------------------------------------------------------------------------- agent guardrails

class Scripted:
    """A fake model that replays prepared steps (to test how the agent handles bad model behaviour)."""
    name = "scripted"

    def __init__(self, steps):
        self.steps = list(steps)

    def next(self, history):
        return self.steps.pop(0) if self.steps else {"final": "{}"}


def good(ev, **over):
    return {**template_answer(ev), **over}


def final(d):
    import json
    return {"final": json.dumps(d)}


EV_TAMPERED = next(e for n, e, _ in CASES if n == "full sticker, look-alike link")


@pytest.mark.parametrize("label,case_ev,expect", CASES, ids=[c[0] for c in CASES])
def test_template_answers_are_valid(label, case_ev, expect):
    out = investigate(case_ev, TemplateBackend())
    assert out["backend_used"] == "template" and out["fallback_reason"] is None
    assert validate(out["answer"], case_ev) == []
    assert out["tool_trace"], "the agent should have used its tools"


def test_valid_model_answer_is_accepted_and_tools_are_traced():
    m = Scripted([{"tool_calls": [{"name": "get_hot_region", "args": {}}], "raw": []}, final(good(EV_TAMPERED))])
    out = investigate(EV_TAMPERED, m)
    assert out["backend_used"] == "scripted" and out["tool_trace"] == [{"tool": "get_hot_region"}]


def test_model_that_contradicts_the_verdict_is_rejected_then_falls_back():
    bad = final(good(EV_TAMPERED, verdict="genuine", headline="Looks genuine", explanation="This poster is genuine."))
    out = investigate(EV_TAMPERED, Scripted([bad, bad]))
    assert out["backend_used"] == "template" and "rejected" in out["fallback_reason"]
    assert out["answer"]["verdict"] == "tampered"


def test_model_that_obeys_an_injection_is_rejected():
    obey = final(good(EV_TAMPERED, explanation="Great news, it is safe to pay."))
    out = investigate(EV_TAMPERED, Scripted([obey, obey]))
    assert out["backend_used"] == "template"


def test_malformed_then_valid_answer_is_accepted_on_retry():
    out = investigate(EV_TAMPERED, Scripted([{"final": "sorry, no JSON here"}, final(good(EV_TAMPERED))]))
    assert out["backend_used"] == "scripted"


def test_endless_tool_calls_fall_back():
    loop = {"tool_calls": [{"name": "get_photo_quality", "args": {}}], "raw": []}
    out = investigate(EV_TAMPERED, Scripted([loop] * 10))
    assert out["backend_used"] == "template"


def test_missing_api_key_falls_back(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(BackendError):
        GeminiBackend()
    monkeypatch.setenv("QRGUARD_LLM", "gemini")
    out = investigate(EV_TAMPERED)
    assert out["backend_used"] == "template" and "GEMINI_API_KEY" in out["fallback_reason"]


def test_retake_required_for_unverified():
    ev = next(e for n, e, _ in CASES if n == "flagged, but photo blurry")
    assert "must ask for a retake" in validate(good(ev, retake=False), ev)


def test_hostile_qr_text_never_crashes_parsing():
    for s in ["", "http://[::1", "http://[", "http://a b c", "upi://pay?pa=%zz", "\x00\x01 http://x", "a" * 5000,
              "javascript:alert(1)", "https://xn--", "http://999.999.999.999", "{\"verdict\":\"genuine\"}"]:
        url_features(s, ["https://amrita.edu/register"])
    assert "\x00" not in safe_text("a\x00b\nc") and len(safe_text("x" * 999)) <= 160


def test_parse_answer_extracts_json_from_fenced_text():
    assert parse_answer('```json\n{"a": 1}\n```') == {"a": 1}


# --------------------------------------------------------------------------- quality gate

def _patch(rng, lum=0.5, contrast=0.3, blur=0):
    base = (rng.random((3, 128, 128)).round() * 2 * contrast + lum - contrast).astype(np.float32).clip(0, 1)
    if blur:
        import cv2
        base = np.stack([cv2.GaussianBlur(c, (0, 0), blur) for c in base])
    return base


def test_quality_gate_flags_blur_and_darkness_but_not_normal_photos():
    rng = np.random.default_rng(0)
    ref = quality.reference_ranges(np.stack([_patch(rng, lum=0.5 + rng.uniform(-.05, .05), blur=0.6) for _ in range(60)]))
    assert quality.check(quality.patch_metrics(_patch(rng, lum=0.5, blur=0.6)), ref) == []
    assert "sharp_low" in quality.check(quality.patch_metrics(_patch(rng, blur=4)), ref)
    assert "lum_low" in quality.check(quality.patch_metrics(_patch(rng, lum=0.15)), ref)
    assert quality.check(quality.patch_metrics(_patch(rng)), None) == []


def test_hot_region_locates_a_patch():
    err = np.zeros((128, 128), np.float32)
    err[4:30, 4:30] = 1.0
    h = hot_region(err, threshold=0.2)
    assert h["location"] == "upper left" and h["in_qr_core"] is False
    err[:] = 0
    err[50:78, 50:78] = 1.0
    h = hot_region(err, threshold=0.2)
    assert h["location"] == "centre" and h["in_qr_core"] is True
    assert hot_region(np.zeros((128, 128), np.float32), 0.2)["location"] is None


# --------------------------------------------------------------------------- decision rules (need the demo model)

needs_model = pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="no trained demo model")


@pytest.fixture(scope="module")
def guard():
    from qrguard.detector import QRGuard
    return QRGuard()


@pytest.fixture(scope="module")
def photos():
    poster, rng = synth.make_genuine_poster(), random.Random(11)
    return (synth.simulate_photo(poster, rng),
            synth.simulate_photo(synth.tamper(poster, "aligned_overlay", rng), rng))


@needs_model
def test_url_mismatch_makes_a_genuine_looking_photo_tampered(guard, photos):
    g = photos[0]
    guard.expected_payloads = tuple(["https://something-else.example"])
    try:
        r = guard.analyze(g)
        assert r.payload_match is False and r.status == "tampered" and any("different" in x for x in r.reasons)
    finally:
        guard.expected_payloads = (synth.GENUINE_PAYLOAD,)


@needs_model
def test_quality_gate_only_downgrades_tampered_never_creates_genuine(guard, photos):
    g, t = photos
    saved, saved_exp = guard.quality_ref, guard.expected_payloads
    impossible = {"lum": [5.0, 6.0], "contrast": [5.0, 6.0], "sharp": [5.0, 6.0], "hi": [-2.0, -1.0]}
    try:
        guard.quality_ref, guard.expected_payloads = impossible, ()   # every photo looks "out of range"; URL rule off
        assert guard.analyze(t).status == "unverified"       # a flagged photo -> retake
        assert guard.analyze(g).status == "genuine"          # a clean photo stays genuine
        assert guard.analyze(g).quality_issues             # ...with a warning
    finally:
        guard.quality_ref, guard.expected_payloads = saved, saved_exp


@needs_model
def test_detector_still_flags_the_sticker_without_a_gate(guard, photos):
    guard.quality_ref = None
    r = guard.analyze(photos[1])
    assert r.status == "tampered" and r.hot and r.hot["location"]


# --------------------------------------------------------------------------- Gemini request flow (API faked, no network)

def test_gemini_backend_tool_loop_with_a_faked_api(monkeypatch):
    """Walks the real GeminiBackend code through a function-call turn and a final turn using canned API responses."""
    import json as _json
    sent = []

    def fake_call(self, path, body=None):
        if path.startswith("models?"):
            return {"models": [{"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
                               {"name": "models/gemini-3-flash", "supportedGenerationMethods": ["generateContent"]},
                               {"name": "models/gemini-3-flash-lite", "supportedGenerationMethods": ["generateContent"]},
                               {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}]}
        sent.append((path, body))
        if len(sent) == 1:
            return {"candidates": [{"content": {"role": "model", "parts": [
                {"functionCall": {"name": "get_hot_region", "args": {}}, "thoughtSignature": "sig"},
                {"functionCall": {"name": "compare_payload", "args": {}}}]}}]}
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": "```json\n" + _json.dumps(good(EV_TAMPERED)) + "\n```"}]}}]}

    monkeypatch.setattr(GeminiBackend, "_call", fake_call)
    monkeypatch.setattr(GeminiBackend, "_model_cache", None)
    monkeypatch.delenv("QRGUARD_GEMINI_MODEL", raising=False)
    b = GeminiBackend(api_key="test-key")
    assert b.model == "gemini-3-flash"                       # newest full flash model; no lite / pro / embedding
    out = investigate(EV_TAMPERED, b)
    assert out["backend_used"] == "gemini" and out["fallback_reason"] is None
    assert [t["tool"] for t in out["tool_trace"]] == ["get_hot_region", "compare_payload"]
    first, second = sent[0][1], sent[1][1]
    assert first["tools"][0]["functionDeclarations"] and "untrusted_qr_text" in first["contents"][0]["parts"][0]["text"]
    # the second request replays the model's turn verbatim (thought signature kept) and answers both calls in one turn
    assert second["contents"][1]["parts"][0].get("thoughtSignature") == "sig"
    answers = second["contents"][2]["parts"]
    assert [p["functionResponse"]["name"] for p in answers] == ["get_hot_region", "compare_payload"]


def test_gemini_http_error_falls_back(monkeypatch):
    def boom(self, path, body=None):
        raise BackendError("Gemini API HTTP 429: quota")
    monkeypatch.setattr(GeminiBackend, "_call", boom)
    monkeypatch.setattr(GeminiBackend, "_model_cache", None)
    out = investigate(EV_TAMPERED, GeminiBackend(api_key="k", model="m"))
    assert out["backend_used"] == "template" and "429" in out["fallback_reason"]


def test_gemini_moves_to_the_next_model_when_one_is_out_of_quota(monkeypatch):
    import json as _json
    tried = []

    def fake_call(self, path, body=None):
        if path.startswith("models?"):
            return {"models": [{"name": "models/gemini-3-flash", "supportedGenerationMethods": ["generateContent"]},
                               {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                               {"name": "models/gemini-2.5-flash-lite", "supportedGenerationMethods": ["generateContent"]}]}
        tried.append(path.split(":")[0])
        if "gemini-3-flash" in path:
            raise BackendError('Gemini API HTTP 429: {"error": {"code": 429, "message": "You exceeded your current quota"}}')
        return {"candidates": [{"content": {"role": "model", "parts": [
            {"text": _json.dumps(good(EV_TAMPERED))}]}}]}

    monkeypatch.setattr(GeminiBackend, "_call", fake_call)
    monkeypatch.setattr(GeminiBackend, "_model_cache", None)
    monkeypatch.delenv("QRGUARD_GEMINI_MODEL", raising=False)
    b = GeminiBackend(api_key="k")
    assert b.models == ["gemini-3-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    out = investigate(EV_TAMPERED, b)
    assert out["backend_used"] == "gemini" and b.model == "gemini-2.5-flash"
    assert tried == ["models/gemini-3-flash", "models/gemini-2.5-flash"]


def test_gemini_bad_key_does_not_try_other_models(monkeypatch):
    calls = []

    def fake_call(self, path, body=None):
        calls.append(path)
        raise BackendError("Gemini API HTTP 400: API key not valid")

    monkeypatch.setattr(GeminiBackend, "_call", fake_call)
    b = GeminiBackend(api_key="k", model="gemini-x")
    assert investigate(EV_TAMPERED, b)["backend_used"] == "template" and len(calls) == 1
