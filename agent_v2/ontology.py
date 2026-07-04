"""Parser for ``hackathon_release/ontology.md``.

Extracts the phase list and, per phase, the candidate actions plus the
apparatus/objects each action may involve.  The ontology is an open set: it is
used as guidance for hypothesis generation, not as a closed label space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from . import config


@dataclass
class Action:
    action_id: str
    zh: str
    en: str
    definition: str
    objects: List[str] = field(default_factory=list)


@dataclass
class Phase:
    phase_id: str
    zh: str
    en: str
    description: str
    actions: List[Action] = field(default_factory=list)


def _clean_id(cell: str) -> str:
    # Action/phase ids are wrapped in backticks and sometimes broken by stray
    # backticks/backslashes/<br> — strip all of that to recover the raw id.
    cell = cell.replace("<br>", "").replace("`", "").replace("\\", "")
    return cell.strip()


def _clean_text(cell: str) -> str:
    return cell.replace("<br>", " ").replace("\\", "").strip()


def _split_objects(cell: str) -> List[str]:
    cell = _clean_text(cell)
    parts = re.split(r"[、,，/]", cell)
    return [p.strip() for p in parts if p.strip()]


def _table_rows(lines: List[str], start: int) -> tuple[List[List[str]], int]:
    """Collect contiguous markdown table rows starting at/after ``start``."""
    rows: List[List[str]] = []
    i = start
    while i < len(lines) and not lines[i].lstrip().startswith("|"):
        if lines[i].strip().startswith("###") or lines[i].strip().startswith("##"):
            return rows, i
        i += 1
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows, i


def _is_separator(row: List[str]) -> bool:
    return all(set(c) <= set("-: ") and c for c in row)


def load_ontology(path: Path | None = None) -> List[Phase]:
    path = path or (config.DATA_ROOT / "ontology.md")
    lines = path.read_text(encoding="utf-8").splitlines()

    phases: Dict[str, Phase] = {}

    # 1) phase table under "## Phase及action列表"
    for idx, line in enumerate(lines):
        if line.strip().startswith("## Phase"):
            rows, _ = _table_rows(lines, idx + 1)
            for row in rows:
                if len(row) < 3 or _is_separator(row):
                    continue
                pid = _clean_id(row[0])
                if not pid or pid.lower() in ("phase id", "phaseid"):
                    continue
                phases[pid] = Phase(phase_id=pid, zh=_clean_text(row[1]),
                                    en=_clean_text(row[2]),
                                    description=_clean_text(row[3]) if len(row) > 3 else "")
            break

    # 2) per-phase action sections: "### ... `phase_id`"
    header_re = re.compile(r"^###\s+.*`([A-Za-z_]+)`")
    for idx, line in enumerate(lines):
        m = header_re.match(line.strip())
        if not m:
            continue
        pid = m.group(1)
        phase = phases.get(pid)
        if phase is None:  # action-only phase (e.g. rotary_evaporation)
            phase = Phase(phase_id=pid, zh="", en=pid.replace("_", " "), description="")
            phases[pid] = phase
        rows, _ = _table_rows(lines, idx + 1)
        for row in rows:
            if len(row) < 3 or _is_separator(row):
                continue
            aid = _clean_id(row[0])
            if not aid or aid.lower() in ("action id", "actionid"):
                continue
            phase.actions.append(Action(
                action_id=aid,
                zh=_clean_text(row[1]),
                en=_clean_text(row[2]),
                definition=_clean_text(row[3]) if len(row) > 3 else "",
                objects=_split_objects(row[4]) if len(row) > 4 else [],
            ))

    return list(phases.values())


def phase_catalog_text(phases: List[Phase]) -> str:
    """Compact phase menu for the phase-hypothesis prompt."""
    out = []
    for p in phases:
        out.append(f"- {p.phase_id} ({p.zh}): {p.description}")
    return "\n".join(out)


def action_catalog_text(phase: Phase) -> str:
    """Actions + candidate objects for one phase, for the segmentation prompt."""
    out = []
    for a in phase.actions:
        objs = "，".join(a.objects[:8])
        out.append(f"- {a.action_id} ({a.zh}): {a.definition} [常见器具: {objs}]")
    return "\n".join(out)


def find_phase(phases: List[Phase], phase_id: str) -> Phase | None:
    for p in phases:
        if p.phase_id == phase_id:
            return p
    return None
