#!/usr/bin/env python3
"""
PRISM v34 — Veto IA (Claude)
=============================
Dernier filtre avant l'ouverture d'une position : Claude lit l'actualité
récente (recherche web) et le contexte de marché, et peut BLOQUER une
entrée. Il ne peut jamais en déclencher une : le moteur technique décide,
l'IA ne fait que retirer des trades.

Activation (désactivé par défaut) :
  export PRISM_AI_VETO=1
  export ANTHROPIC_API_KEY=...        # ou `ant auth login`
  export PRISM_AI_VETO_MIN_CONF=0.7   # optionnel : confiance mini pour bloquer

En cas d'erreur (réseau, clé absente, réponse illisible), le trade est
AUTORISÉ : le bot se comporte alors exactement comme v33.

Test manuel :
  python3 ai_veto.py SOL-USDT long
"""

import csv
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

try:
    import anthropic
except ImportError:
    anthropic = None

log = logging.getLogger("prism_live")

BASE_DIR   = Path(os.environ.get("PRISM_DATA_DIR", str(Path(__file__).parent)))
VETO_LOG   = BASE_DIR / "live_logs" / "ai_veto_decisions.csv"
MODEL      = "claude-opus-5-5"
MIN_CONF   = float(os.environ.get("PRISM_AI_VETO_MIN_CONF", "0.7"))
FNG_URL    = "https://api.alternative.me/fng/?limit=2"

SYSTEM_PROMPT = """Tu es le contrôle de risque d'un bot de trading crypto sur contrats perpétuels \
(levier ×8 à ×15, horizon de quelques heures à 3 jours). Le moteur technique a déjà validé un \
signal ; ton seul rôle est de dire s'il faut le BLOQUER.

Bloque uniquement pour un risque concret et récent que les indicateurs techniques ne voient pas :
- piratage, exploit, faille ou incident de sécurité sur le projet ou une plateforme majeure ;
- delisting, suspension des dépôts/retraits, problème réglementaire ou judiciaire visant l'actif ;
- annonce macro ou événement programmé dans les prochaines heures (CPI, FOMC, unlock massif de \
tokens…) qui rend le mouvement imprévisible ;
- actualité qui va clairement à l'encontre du sens du trade (par ex. nouvelle très négative \
pour un long).

N'utilise pas ton intuition sur la direction du prix : sans risque identifié, réponds "allow". \
Fais au plus quelques recherches web ciblées (actualité de l'actif et du marché crypto sur les \
dernières 24-48 h).

Termine ta réponse par un unique objet JSON sur une ligne, sans texte après :
{"decision": "allow" | "veto", "confidence": <0.0 à 1.0>, "reason": "<une phrase en français>"}"""

def _fear_greed() -> str:
    try:
        data = requests.get(FNG_URL, timeout=10).json()["data"]
        return ", ".join(f"{d['value']} ({d['value_classification']})" for d in data)
    except Exception:
        return "indisponible"

def _extract_json(text: str) -> dict | None:
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if obj.get("decision") in ("allow", "veto"):
            return obj
    return None

def _record(sym: str, side: str, pattern: str, decision: dict, entry_px: float | None):
    VETO_LOG.parent.mkdir(exist_ok=True)
    exists = VETO_LOG.exists()
    with open(VETO_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts", "sym", "side", "pattern", "decision", "confidence",
                        "blocked", "entry_px", "reason"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), sym, side, pattern,
                    decision.get("decision"), decision.get("confidence"),
                    decision.get("blocked"), entry_px, decision.get("reason", "")])

def enabled() -> bool:
    return os.environ.get("PRISM_AI_VETO") == "1" and anthropic is not None

def ask_claude(sym: str, side: str, pattern: str, score: int,
               leverage: int, market: dict | None = None) -> dict:
    """Retourne {"decision", "confidence", "reason"} ; ne lève jamais."""
    asset = sym.split("-")[0]
    context = {
        "actif": asset,
        "paire": sym,
        "sens": side,
        "pattern": pattern,
        "score_technique": score,
        "levier": leverage,
        "fear_and_greed (aujourd'hui, hier)": _fear_greed(),
        "date_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        **(market or {}),
    }
    try:
        client = anthropic.Anthropic(timeout=180, max_retries=2)
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "medium"},
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 4}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content":
                       "Signal à contrôler :\n" + json.dumps(context, ensure_ascii=False, indent=1)}],
        )
    except anthropic.AuthenticationError:
        return {"decision": "allow", "confidence": 0, "reason": "erreur : clé API invalide"}
    except anthropic.RateLimitError:
        return {"decision": "allow", "confidence": 0, "reason": "erreur : limite de débit API"}
    except anthropic.APIStatusError as e:
        return {"decision": "allow", "confidence": 0, "reason": f"erreur API {e.status_code}"}
    except anthropic.APIConnectionError:
        return {"decision": "allow", "confidence": 0, "reason": "erreur : réseau"}
    except Exception as e:
        return {"decision": "allow", "confidence": 0, "reason": f"erreur : {e}"}

    if response.stop_reason == "refusal":
        return {"decision": "allow", "confidence": 0, "reason": "refus du modèle"}
    text = "".join(b.text for b in response.content if b.type == "text")
    parsed = _extract_json(text)
    if parsed is None:
        return {"decision": "allow", "confidence": 0,
                "reason": f"réponse illisible (stop={response.stop_reason})"}
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0
    return parsed

def should_block(sym: str, side: str, pattern: str, score: int, leverage: int,
                 market: dict | None = None, entry_px: float | None = None) -> tuple[bool, str]:
    """
    Appelé juste avant l'ouverture d'une position.
    Retourne (bloquer, raison). Sans activation → (False, "").
    """
    if not enabled():
        return False, ""
    decision = ask_claude(sym, side, pattern, score, leverage, market)
    blocked  = decision["decision"] == "veto" and decision["confidence"] >= MIN_CONF
    decision["blocked"] = blocked
    _record(sym, side, pattern, decision, entry_px)
    log.info(f"VETO-IA {sym:12s} {side:5s} | {decision['decision']} "
             f"conf={decision['confidence']:.2f} bloqué={blocked} | {decision.get('reason', '')}")
    return blocked, decision.get("reason", "")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 3:
        print("Usage : python3 ai_veto.py SOL-USDT long|short")
        sys.exit(1)
    os.environ.setdefault("PRISM_AI_VETO", "1")
    if anthropic is None:
        print("Le paquet `anthropic` n'est pas installé : pip install anthropic")
        sys.exit(1)
    print(json.dumps(ask_claude(sys.argv[1], sys.argv[2], "C", 70, 10),
                     ensure_ascii=False, indent=1))
