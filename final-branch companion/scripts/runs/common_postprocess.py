from __future__ import annotations

import csv
import gc
import hashlib
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd


def _json_default(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


_RUNTIME_ONLY_PARAM_KEYS = {"scenario_paths"}


def _sanitize_runtime_params(value: Any) -> Any:
    """Remove runtime-only tensors from auditable/search-parameter payloads."""
    if isinstance(value, dict):
        return {
            str(k): _sanitize_runtime_params(v)
            for k, v in value.items()
            if str(k) not in _RUNTIME_ONLY_PARAM_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_runtime_params(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_runtime_params(v) for v in value)
    return value


def _strip_json_object_key(raw: str, key: str) -> tuple[str, bool]:
    """Remove one top-level JSON key without materializing its (possibly huge) value.

    This is used to repair legacy ``trial_audit.csv`` files where ``scenario_paths``
    was serialized as a giant nested array.  The scanner keeps memory bounded to one
    CSV row instead of ``json.loads``-ing the full tensor.
    """
    if not isinstance(raw, str) or not raw or f'"{key}"' not in raw:
        return raw, False
    text = raw
    changed = False
    token = f'"{key}"'
    while True:
        key_start = text.find(token)
        if key_start < 0:
            break
        colon = text.find(":", key_start + len(token))
        if colon < 0:
            break
        i = colon + 1
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break

        # Scan the JSON value to its exclusive end, respecting nested arrays/objects
        # and quoted strings.
        ch = text[i]
        if ch in "[{":
            stack = [ch]
            i += 1
            in_string = False
            escaped = False
            while i < len(text) and stack:
                c = text[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif c == "\\":
                        escaped = True
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c in "[{":
                        stack.append(c)
                    elif c == "]" and stack and stack[-1] == "[":
                        stack.pop()
                    elif c == "}" and stack and stack[-1] == "{":
                        stack.pop()
                i += 1
            value_end = i
        elif ch == '"':
            i += 1
            escaped = False
            while i < len(text):
                c = text[i]
                i += 1
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    break
            value_end = i
        else:
            while i < len(text) and text[i] not in ",}":
                i += 1
            value_end = i

        left = key_start
        while left > 0 and text[left - 1].isspace():
            left -= 1
        if left > 0 and text[left - 1] == ",":
            left -= 1
            right = value_end
        else:
            right = value_end
            while right < len(text) and text[right].isspace():
                right += 1
            if right < len(text) and text[right] == ",":
                right += 1
        text = text[:left] + text[right:]
        changed = True
    return text, changed


def repair_trial_audit_csv(output_path: str | Path) -> dict[str, int]:
    """Stream-repair legacy trial ledgers that accidentally embedded scenario paths.

    Returns counts so the caller can report whether a rewrite occurred.
    """
    folder = Path(output_path)
    path = folder / "trial_audit.csv"
    if not path.exists() or path.stat().st_size == 0:
        return {"rows": 0, "repaired_rows": 0}

    # Fast path: avoid rewriting normal/small ledgers.
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        head = fh.read(min(path.stat().st_size, 1_000_000))
    if "scenario_paths" not in head and path.stat().st_size < 8 * 1024 * 1024:
        return {"rows": 0, "repaired_rows": 0}

    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    tmp = path.with_suffix(path.suffix + ".repair.tmp")
    rows = 0
    repaired = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as src, tmp.open("w", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src)
            if not reader.fieldnames:
                return {"rows": 0, "repaired_rows": 0}
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                rows += 1
                raw_params = row.get("params", "")
                clean_params, changed = _strip_json_object_key(raw_params, "scenario_paths")
                if changed:
                    repaired += 1
                    row["params"] = clean_params
                    try:
                        params_obj = json.loads(clean_params) if clean_params else {}
                    except Exception:
                        params_obj = clean_params
                    alpha_raw = row.get("alpha")
                    try:
                        alpha_obj: Any = float(alpha_raw) if alpha_raw not in (None, "", "nan", "NaN") else None
                    except Exception:
                        alpha_obj = alpha_raw
                    payload = {
                        "source": row.get("source"),
                        "method": row.get("method"),
                        "params": params_obj,
                        "alpha": alpha_obj,
                    }
                    recipe_raw = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))
                    if "recipe_id" in row:
                        row["recipe_id"] = hashlib.sha256(recipe_raw.encode("utf-8")).hexdigest()[:16]
                writer.writerow(row)
        if repaired:
            # Atomic replacement keeps the original intact until the repaired file
            # is fully written, then immediately releases the giant legacy file.
            tmp.replace(path)
        else:
            tmp.unlink(missing_ok=True)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return {"rows": rows, "repaired_rows": repaired}


def save_selection_audit(output_path: str | Path, selection_audit: list[dict[str, Any]]) -> None:
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    if not selection_audit:
        return
    audit_rows: list[dict[str, Any]] = []
    smart_rows: list[dict[str, Any]] = []
    smart_keys = {
        "overlay_fraction", "backward_vol_timer", "forward_cvar_timer", "forward_cdar_timer", "forward_risk_timer", "conviction_timer",
        "portfolio_kelly", "portfolio_risk_reward", "portfolio_garch_vol",
        "portfolio_realized_vol", "portfolio_vol_regime", "forward_model_cvar",
        "forward_model_cvar_raw", "forward_model_cdar", "forward_model_cdar_raw", "portfolio_scenario_mean",
        "historical_cvar_target", "historical_cvar_target_raw", "historical_cdar_target", "historical_cdar_target_raw",
        "signal_centering_applied", "composite_quality", "selected_count",
        "overlay_combination", "forward_risk_measure", "comments", "smart_signal_error",
        "smart_position_sizing_applied", "smart_signal_table",
    }
    for row in selection_audit:
        clean = dict(row)
        smart = {k: clean.pop(k) for k in list(clean) if k.startswith("smart_") or k in smart_keys}
        if smart:
            smart_rows.append({"date": clean.get("date"), **smart})
        for key, val in list(clean.items()):
            if isinstance(val, (dict, list, tuple)):
                clean[key] = json.dumps(val, default=_json_default)
        audit_rows.append(clean)
    pd.DataFrame(audit_rows).to_csv(out / "selection_audit.csv", index=False)
    if smart_rows:
        for row in smart_rows:
            if "smart_signal_table" in row:
                row["smart_signal_table"] = json.dumps(row["smart_signal_table"], default=_json_default)
        pd.DataFrame(smart_rows).to_csv(out / "smart_rebalance_audit.csv", index=False)



def candidate_trial_records(
    rebalance_date: str,
    validation_table: pd.DataFrame,
    test_table: pd.DataFrame,
    selected_candidate_id: str | None = None,
    attempt_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten candidate tables into a publication-grade trial audit."""
    rows: list[dict[str, Any]] = []
    for stage, table in (("validation", validation_table), ("test", test_table)):
        if table is None or table.empty:
            continue
        for record in table.to_dict(orient="records"):
            params = record.get("params", {})
            if isinstance(params, str):
                try:
                    params_obj = json.loads(params)
                except Exception:
                    params_obj = params
            else:
                params_obj = params
            params_obj = _sanitize_runtime_params(params_obj)
            payload = {
                "source": record.get("source"),
                "method": record.get("method"),
                "params": params_obj,
                "alpha": record.get("alpha"),
            }
            recipe_raw = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))
            clean: dict[str, Any] = {
                "rebalance_date": rebalance_date,
                "stage": stage,
                "recipe_id": hashlib.sha256(recipe_raw.encode("utf-8")).hexdigest()[:16],
                "selected_candidate": str(record.get("candidate_id")) == str(selected_candidate_id),
            }
            for key, value in record.items():
                if key == "params":
                    value = params_obj
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(value, default=_json_default, sort_keys=True)
                else:
                    clean[key] = value
            rows.append(clean)

    # Failed/pruned candidate attempts have no Sharpe statistic but still count as
    # researcher degrees of freedom. Structural incompatibilities are retained for
    # transparency with counts_as_trial=False.
    for attempt in (attempt_records or []):
        params_obj = _sanitize_runtime_params(attempt.get("params", {}))
        payload = {
            "source": attempt.get("source"), "method": attempt.get("method"),
            "params": params_obj, "alpha": attempt.get("alpha"),
        }
        recipe_raw = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))
        rows.append({
            "rebalance_date": rebalance_date,
            "stage": "attempt",
            "recipe_id": hashlib.sha256(recipe_raw.encode("utf-8")).hexdigest()[:16],
            "selected_candidate": False,
            "candidate_id": None,
            "source": attempt.get("source"),
            "method": attempt.get("method"),
            "params": json.dumps(params_obj, default=_json_default, sort_keys=True),
            "alpha": attempt.get("alpha"),
            "attempt_status": attempt.get("status", "failed"),
            "attempt_error": attempt.get("error"),
            "counts_as_trial": bool(attempt.get("counts_as_trial", True)),
        })
    return rows


def save_candidate_trials(output_path: str | Path, trial_rows: list[dict[str, Any]]) -> None:
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    if trial_rows:
        pd.DataFrame(trial_rows).to_csv(out / "trial_audit.csv", index=False)

def save_risk_matrices(output_path: str | Path, snapshots: list[dict[str, Any]]) -> None:
    """Persist risk-matrix snapshots without duplicating the full run in memory.

    A long ETF run can contain tens of millions of matrix elements.  Building one
    giant indented JSON string with ``json.dumps(snapshots)`` temporarily duplicates
    that whole object and can make macOS/Linux kill the process after the backtest
    has already finished.  Stream one snapshot at a time instead.  The resulting
    file is still a normal JSON array, so existing readers remain compatible.
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    if not snapshots:
        return

    target = out / "risk_matrices.json"
    tmp = out / "risk_matrices.json.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write("[")
            for i, snapshot in enumerate(snapshots):
                if i:
                    fh.write(",\n")
                # Only this one month's NumPy arrays are converted to Python lists
                # by _json_default at a time.  Compact separators also reduce the
                # output size substantially versus indent=2.
                json.dump(
                    snapshot,
                    fh,
                    default=_json_default,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                fh.flush()
            fh.write("]")
        tmp.replace(target)

        # Keep a publication/dashboard-sized companion so Plotly never needs to
        # deserialize every full covariance/correlation matrix just to display them.
        try:
            import numpy as np
            n = len(snapshots)
            keep = set(range(n)) if n <= 12 else set(np.linspace(0, n - 1, 12).round().astype(int).tolist())
            compact: list[dict[str, Any]] = []
            for i, snapshot in enumerate(snapshots):
                if i not in keep:
                    continue
                labels = list(snapshot.get("tickers", []))[:80]
                item: dict[str, Any] = {
                    "date": snapshot.get("date"),
                    "tickers": labels,
                }
                dim = len(labels)
                for kind in ("correlation", "covariance_horizon"):
                    arr = np.asarray(snapshot.get(kind, []), dtype=float)
                    if arr.ndim == 2 and arr.size:
                        item[kind] = arr[:dim, :dim].tolist()
                compact.append(item)
            (out / "risk_matrices_report.json").write_text(
                json.dumps(compact, separators=(",", ":"), default=_json_default),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"Compact risk-matrix report cache was not generated: {exc}")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass




def _iter_json_array(path: Path, chunk_size: int = 1024 * 1024):
    """Yield objects from a JSON array while keeping only one object in memory."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
        buffer = ""
        pos = 0
        started = False
        eof = False
        while True:
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if not started and pos < len(buffer):
                    if buffer[pos] != "[":
                        raise ValueError(f"Expected JSON array in {path}")
                    started = True
                    pos += 1
                    continue
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
                    pos += 1
                if pos < len(buffer):
                    break
                if eof:
                    return
                buffer = buffer[pos:]
                pos = 0
                chunk = fh.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            if buffer[pos] == "]":
                return
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                    pos = end
                    yield item
                    # Periodically discard consumed text so giant files do not stay
                    # resident as one ever-growing Python string.
                    if pos > 4 * chunk_size:
                        buffer = buffer[pos:]
                        pos = 0
                    break
                except json.JSONDecodeError:
                    chunk = fh.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk


def ensure_risk_matrix_report_cache(output_path: str | Path, max_snapshots: int = 12, max_dim: int = 80) -> Path | None:
    """Create a compact risk-matrix cache from an existing full JSON artifact.

    Recovery runs use this streaming path because the original optimization has
    already ended and ``save_risk_matrices`` cannot be called again.  Keeping only
    the most recent snapshots is sufficient for the dashboard while the complete
    research artifact remains untouched on disk.
    """
    out = Path(output_path)
    compact_path = out / "risk_matrices_report.json"
    if compact_path.exists():
        return compact_path
    full_path = out / "risk_matrices.json"
    if not full_path.exists() or full_path.stat().st_size == 0:
        return None
    kept: deque[dict[str, Any]] = deque(maxlen=max(int(max_snapshots), 1))
    for snapshot in _iter_json_array(full_path):
        if not isinstance(snapshot, dict):
            continue
        labels = list(snapshot.get("tickers", []))[:max(int(max_dim), 1)]
        dim = len(labels)
        item: dict[str, Any] = {"date": snapshot.get("date"), "tickers": labels}
        for kind in ("correlation", "covariance_horizon"):
            matrix = snapshot.get(kind, [])
            if isinstance(matrix, list) and matrix:
                item[kind] = [list(row)[:dim] for row in matrix[:dim] if isinstance(row, (list, tuple))]
        kept.append(item)
    if not kept:
        return None
    compact_path.write_text(
        json.dumps(list(kept), separators=(",", ":"), default=_json_default),
        encoding="utf-8",
    )
    return compact_path


def save_run_metadata(output_path: str | Path, metadata: dict[str, Any]) -> None:
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )


def generate_metric_artifacts_safely(output_path: str | Path, split_date: str | None = None) -> None:
    """Create overall/train/test and rolling metric CSVs from a finished run."""
    try:
        from scripts.results.stats import export_metric_artifacts
        export_metric_artifacts(output_path, split_date=split_date)
    except Exception as exc:
        print(f"Metric artifacts were not generated: {exc}")



def generate_research_validation_safely(output_path: str | Path, config: dict[str, Any] | None = None) -> None:
    """Run PSR/DSR, false-strategy, FDR, PBO-proxy and causality diagnostics."""
    try:
        from scripts.validation.research_validation import run_research_validation
        run_research_validation(output_path, config=config)
    except Exception as exc:
        print(f"Research validation artifacts were not generated: {exc}")



def generate_signal_research_safely(output_path: str | Path, config: dict[str, Any] | None = None) -> None:
    """Run the multi-signal dependence/cointegration/subset research when enabled."""
    cfg = dict((config or {}).get("signal_research", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return
    try:
        from scripts.validation.signal_research import run_signal_research
        run_signal_research(
            output_path,
            rolling_window=int(cfg.get("rolling_window", 24)),
            max_lag=int(cfg.get("max_lag", 6)),
            defensive_quantile=float(cfg.get("defensive_quantile", 0.20)),
            min_train=int(cfg.get("min_train", 24)),
            distribution_bootstrap_samples=int(cfg.get("distribution_bootstrap_samples", 199)),
            conditional_bins=int(cfg.get("conditional_bins", 5)),
            max_surface_signals=int(cfg.get("max_surface_signals", 6)),
        )
    except Exception as exc:
        print(f"Signal research artifacts were not generated: {exc}")

def generate_interactive_report_safely(output_path: str | Path) -> None:
    try:
        from scripts.results.interactive_report import build_interactive_report
        path = build_interactive_report(output_path)
        print(f"Interactive Plotly report saved to {path}")
    except Exception as exc:
        print(f"Interactive Plotly report was not generated: {exc}")


def generate_theorem_empirical_audit_safely(output_path: str | Path) -> None:
    """Measure real-data counterparts of manuscript structural results from run artifacts."""
    try:
        from scripts.validation.theorem_empirical_bridge import run_historical_theorem_audit
        run_historical_theorem_audit(output_path)
    except Exception as exc:
        print(f"Theorem empirical audit artifacts were not generated: {exc}")


def run_postprocessing(
    output_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    split_date: str | None = None,
) -> None:
    """Run end-of-backtest artifact generation after large optimization objects are freed.

    The function is intentionally restartable: it consumes only files already saved in
    ``results/<run>``.  This allows ``--postprocess-only`` to finish a run that macOS
    killed after optimization, without repeating the multi-hour backtest.
    """
    out = Path(output_path)
    stats = repair_trial_audit_csv(out)
    if stats.get("repaired_rows", 0):
        print(
            f"Repaired trial_audit.csv: removed runtime scenario_paths from "
            f"{stats['repaired_rows']} rows before validation."
        )
    gc.collect()
    generate_research_validation_safely(out, config=config)
    gc.collect()
    generate_metric_artifacts_safely(out, split_date=split_date)
    gc.collect()
    generate_signal_research_safely(out, config=config)
    gc.collect()
    try:
        ensure_risk_matrix_report_cache(out)
    except Exception as exc:
        print(f"Compact risk-matrix report cache was not generated from existing JSON: {exc}")
    gc.collect()
    generate_interactive_report_safely(out)
    gc.collect()
