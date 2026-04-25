"""Sodir CSV pre-processing — derived FK columns the blueprint needs.

The blueprint expects integer foreign keys for cross-CSV connections,
but the raw factmaps tables don't always include them — some tables
identify referenced rows by name instead of NPDID. These helpers add
the missing FK columns by joining on the natural key, idempotent so
repeated runs don't re-derive.

Ported from `build_sodir_graph.py:56-124` with all the user-facing
"this is a graph enhancement" steps stripped out — only the mechanical
joins live here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def apply(csv_dir: Path, *, verbose: bool = True) -> dict[str, int]:
    """Run every applicable pre-processing step on the CSVs under
    `csv_dir`. Each step is a no-op when its target CSV is missing
    (e.g. user-supplied subset). Returns a dict of step → unmapped
    row count for visibility into FK gaps."""
    report: dict[str, int] = {}

    if (csv_dir / "petreg_licence.csv").exists():
        report["petreg_licence_pk"] = _add_petreg_licence_pk(csv_dir, verbose)

    if (csv_dir / "seismic_acquisition.csv").exists() and (csv_dir / "seismic_acquisition_progress.csv").exists():
        report["seismic_progress_fk"] = _add_seismic_progress_fk(csv_dir, verbose)

    if (csv_dir / "strat_chrono.csv").exists():
        report["chrono_parent_fk"] = _add_chrono_parent_fk(csv_dir, verbose)

    if (csv_dir / "block.csv").exists() and (csv_dir / "announced_history.csv").exists():
        report["announced_block_fk"] = _add_announced_block_fk(csv_dir, verbose)

    return report


def _add_petreg_licence_pk(csv_dir: Path, verbose: bool) -> int:
    """`petreg_licence.csv` ships with a GUID PK that nothing joins on
    cleanly. Add a sequential integer `ptl_id` and propagate it to the
    related child CSVs (message, licensee, operator) so FK joins work.
    Idempotent: re-running on a CSV that already has `ptl_id` is a
    no-op."""
    parent_path = csv_dir / "petreg_licence.csv"
    df = pd.read_csv(parent_path, low_memory=False)
    if "ptl_id" not in df.columns:
        df["ptl_id"] = range(1, len(df) + 1)
        df.to_csv(parent_path, index=False)
        if verbose:
            print(f"  pre-process: petreg_licence.csv ← ptl_id (1..{len(df)})")

    guid_to_id = dict(zip(df["ptlPetregLicenceID"], df["ptl_id"], strict=False))

    unmapped_total = 0
    for child in ("petreg_licence_message.csv", "petreg_licence_licensee.csv", "petreg_licence_operator.csv"):
        path = csv_dir / child
        if not path.exists():
            continue
        cdf = pd.read_csv(path, low_memory=False)
        if "ptl_id" in cdf.columns:
            continue  # already pre-processed
        cdf["ptl_id"] = cdf["ptlPetregLicenceID"].map(guid_to_id).astype("Int64")
        unmapped = int(cdf["ptl_id"].isna().sum())
        unmapped_total += unmapped
        cdf.to_csv(path, index=False)
        if verbose:
            print(f"  pre-process: {child} ← ptl_id ({unmapped} unmapped)")
    return unmapped_total


def _add_seismic_progress_fk(csv_dir: Path, verbose: bool) -> int:
    """`seismic_acquisition_progress.csv` joins to `seismic_acquisition`
    by name, not NPDID. Resolve the NPDID and write it as
    `seaNpdidSurvey`."""
    survey = pd.read_csv(csv_dir / "seismic_acquisition.csv", low_memory=False)
    progress_path = csv_dir / "seismic_acquisition_progress.csv"
    df = pd.read_csv(progress_path, low_memory=False)
    if "seaNpdidSurvey" in df.columns and df["seaNpdidSurvey"].notna().any():
        return 0
    name_to_npdid = dict(zip(survey["seaName"], survey["seaNpdidSurvey"], strict=False))
    df["seaNpdidSurvey"] = df["seaSurveyName"].map(name_to_npdid).astype("Int64")
    unmapped = int(df["seaNpdidSurvey"].isna().sum())
    df.to_csv(progress_path, index=False)
    if verbose:
        print(f"  pre-process: seismic_acquisition_progress.csv ← seaNpdidSurvey ({unmapped} unmapped)")
    return unmapped


def _add_chrono_parent_fk(csv_dir: Path, verbose: bool) -> int:
    """`strat_chrono.csv` is self-referencing — each row links to a
    parent stratigraphic unit by name. Resolve to NPDID for a clean
    in-graph self-edge."""
    path = csv_dir / "strat_chrono.csv"
    df = pd.read_csv(path, low_memory=False)
    if "strat_chrono_parent_npdid" in df.columns and df["strat_chrono_parent_npdid"].notna().any():
        return 0
    if "strat_chrono_name" not in df.columns or "strat_chrono_parent_name" not in df.columns:
        return 0
    name_to_npdid = dict(zip(df["strat_chrono_name"], df["NPDID_strat_chrono"], strict=False))
    df["strat_chrono_parent_npdid"] = df["strat_chrono_parent_name"].map(name_to_npdid).astype("Int64")
    unmapped = int(df["strat_chrono_parent_npdid"].isna().sum())
    df.to_csv(path, index=False)
    if verbose:
        print(f"  pre-process: strat_chrono.csv ← strat_chrono_parent_npdid ({unmapped} unmapped)")
    return unmapped


def _add_announced_block_fk(csv_dir: Path, verbose: bool) -> int:
    """`announced_history.csv` references blocks by name. Resolve to
    NPDID for FK joins."""
    block = pd.read_csv(csv_dir / "block.csv", low_memory=False)
    path = csv_dir / "announced_history.csv"
    df = pd.read_csv(path, low_memory=False)
    if "blcNpdidBlock" in df.columns and df["blcNpdidBlock"].notna().any():
        return 0
    name_to_npdid = dict(zip(block["blcName"], block["blcNpdidBlock"], strict=False))
    df["blcNpdidBlock"] = df["block"].map(name_to_npdid).astype("Int64")
    unmapped = int(df["blcNpdidBlock"].isna().sum())
    df.to_csv(path, index=False)
    if verbose:
        print(f"  pre-process: announced_history.csv ← blcNpdidBlock ({unmapped} unmapped)")
    return unmapped
