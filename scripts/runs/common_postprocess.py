from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_selection_audit(output_path: str | Path, selection_audit: list[dict[str, Any]]) -> None:
    """Save nested validation/test and smart-signal diagnostics without breaking on nested dicts."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    if not selection_audit:
        return
    audit_rows = []
    smart_rows = []
    for row in selection_audit:
        clean = dict(row)
        smart = {k: clean.pop(k) for k in list(clean.keys()) if k.startswith("smart_") or k in {
            "overlay_fraction", "portfolio_kelly", "portfolio_risk_reward", "portfolio_garch_vol",
            "portfolio_vol_regime", "composite_quality", "selected_count", "comments",
            "smart_signal_error", "smart_position_sizing_applied"
        }}
        if smart:
            smart_rows.append({"date": clean.get("date"), **smart})
        for key, val in list(clean.items()):
            if isinstance(val, (dict, list, tuple)):
                clean[key] = json.dumps(val, default=str)
        audit_rows.append(clean)
    pd.DataFrame(audit_rows).to_csv(out / "selection_audit.csv", index=False)
    if smart_rows:
        # Keep the full per-ticker signal table in selection_audit JSON. The flat audit is dashboard-friendly.
        flat = []
        for row in smart_rows:
            row = dict(row)
            if "smart_signal_table" in row:
                row["smart_signal_table"] = json.dumps(row["smart_signal_table"], default=str)
            flat.append(row)
        pd.DataFrame(flat).to_csv(out / "smart_rebalance_audit.csv", index=False)


def generate_interactive_report_safely(output_path: str | Path) -> None:
    try:
        from scripts.results.interactive_report import build_interactive_report
        path = build_interactive_report(output_path)
        print(f"Interactive Plotly report saved to {path}")
    except Exception as exc:
        print(f"Interactive Plotly report was not generated: {exc}")
