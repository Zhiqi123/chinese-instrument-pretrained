"""
Full dataset cleaning pipeline:
  Step 0: Preflight — validate env, config, dependencies
  Step 1: Scan + record_id — traverse audio dirs, output scan_manifest.csv
  Step 2: Metadata — extract audio properties + QC metrics
  Step 3: Auto QC — D03/D04/D05a/D06a auto-drop + pre-screen
  Step 4: Apply manual review — manual_review.csv keep/drop
  Step 5: Dedup + Group — auto-group + D09 exact dup + fingerprint merge + D10
  Step 6: Quota — B01 phrase downsample + R01/R02/R03 gates + B02/B03 monitor
  Step 7: Freeze — assign sample_id + freeze gates + frozen_manifest.csv
  Step 8: Split — 3-seed group-aware split + hard gates
  Step 9: Segment — 24kHz resample + non-overlap segment + energy filter + export
  Step 10: Final checks — global gates + frozen_summary.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn, Optional

import numpy as np

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Manifest column definitions

# scan_manifest columns (Step 1)
SCAN_MANIFEST_COLUMNS: list[str] = [
    "record_id",
    "source_dataset",
    "source_path",
    "file_name",
    "file_ext",
    "raw_dir_label",
    "scan_status",
    "mapped_family_label",
    "drop_reason",
]

# audit_manifest columns (Step 2)
AUDIT_MANIFEST_COLUMNS: list[str] = [
    "record_id",
    "sample_id",
    "source_dataset",
    "source_path",
    "file_name",
    "file_ext",
    "source_subtype_dir",
    "family_label",
    "subtype_label",
    "chmusic_class_id",
    "content_type",
    "normalized_base_name",
    "audio_sha256",
    "audio_fingerprint_id",
    "performer_or_recording_id",
    "recording_group_id",
    "grouping_method",
    "take_id",
    "is_near_duplicate",
    "duplicate_of",
    "duration_sec",
    "sample_rate_hz",
    "channels",
    "bit_depth",
    "integrated_loudness_lufs",
    "rms_db",
    "silence_ratio",
    "clipping_ratio",
    "has_vocal",
    "is_decodable",
    "manual_review_required",
    "manual_review_status",
    "sample_status",
    "drop_reason",
    "applied_rule_ids",
    "notes",
]

# frozen_manifest columns (Step 7)
FROZEN_MANIFEST_COLUMNS: list[str] = [
    "record_id",
    "sample_id",
    "source_dataset",
    "source_path",
    "file_name",
    "file_ext",
    "source_subtype_dir",
    "family_label",
    "subtype_label",
    "chmusic_class_id",
    "content_type",
    "normalized_base_name",
    "audio_sha256",
    "audio_fingerprint_id",
    "performer_or_recording_id",
    "recording_group_id",
    "grouping_method",
    "take_id",
    "is_near_duplicate",
    "duration_sec",
    "sample_rate_hz",
    "channels",
    "bit_depth",
    "integrated_loudness_lufs",
    "rms_db",
    "silence_ratio",
    "clipping_ratio",
    "has_vocal",
    "is_decodable",
    "manual_review_required",
    "manual_review_status",
    "sample_status",
    "applied_rule_ids",
    "notes",
]

# split_manifest columns (Step 8)
SPLIT_MANIFEST_COLUMNS: list[str] = [
    "record_id",
    "sample_id",
    "source_dataset",
    "source_path",
    "file_name",
    "file_ext",
    "source_subtype_dir",
    "family_label",
    "subtype_label",
    "chmusic_class_id",
    "content_type",
    "normalized_base_name",
    "audio_sha256",
    "audio_fingerprint_id",
    "performer_or_recording_id",
    "recording_group_id",
    "grouping_method",
    "take_id",
    "is_near_duplicate",
    "split_seed",
    "split",
    "duration_sec",
    "sample_rate_hz",
    "channels",
    "bit_depth",
    "integrated_loudness_lufs",
    "rms_db",
    "silence_ratio",
    "clipping_ratio",
    "has_vocal",
    "is_decodable",
    "manual_review_required",
    "manual_review_status",
    "sample_status",
    "applied_rule_ids",
    "notes",
]

# segment_manifest columns (Step 9)
SEGMENT_MANIFEST_COLUMNS: list[str] = [
    "record_id",
    "sample_id",
    "source_dataset",
    "family_label",
    "split_seed",
    "split",
    "segment_id",
    "segment_index",
    "start_time_sec",
    "end_time_sec",
    "segment_path",
    "is_padded",
    "overlap_ratio",
    "selection_method",
    "active_frame_ratio",
]

# content_type keyword priority (phrase > technique > controlled)
_CONTENT_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("phrase", ["乐曲片段", "选段", "节选"]),
    ("technique", ["演奏技法", "演奏技巧", "技法", "技巧"]),
    ("controlled", ["音阶", "音程", "筒音", "空弦音"]),
]

# bit_depth mapping
_SUBTYPE_TO_BIT_DEPTH: dict[str, int] = {
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
}

# Primary datasets (train/val/test split)
_PRIMARY_DATASETS: set[str] = {"CCMusic", "External"}

# freeze_config.yaml required fields
_FREEZE_REQUIRED_KEYS: list[str] = [
    "active_classes",
    "family_abbr",
    "ccmusic_whitelist",
    "chmusic_enabled_ids",
    "chmusic_excluded_ids",
    "thresholds",
    "split_seeds",
    "seed_namespaces",
]

# manual_review.csv columns
_MANUAL_REVIEW_COLUMNS: list[str] = [
    "record_id",
    "review_type",
    "decision",
    "drop_reason",
    "content_type_override",
    "peer_record_id",
    "canonical_record_id",
    "bit_depth_override",
    "justification",
    "reviewer",
]

# Valid review_type enum
_VALID_REVIEW_TYPES: set[str] = {
    "content_type",
    "i10_encoding",
    "d09_cross",
    "g02_fingerprint",
    "manual_merge",
}

# Valid decision enum
_VALID_DECISIONS: set[str] = {"approved", "rejected"}

# Pair review types
_PAIR_REVIEW_TYPES: set[str] = {"d09_cross", "g02_fingerprint", "manual_merge"}

# Review types where rejected causes drop
_DROP_ON_REJECT_REVIEW_TYPES: set[str] = {
    "content_type", "i10_encoding",
    "d09_cross",
}

# Valid drop_reason values
_VALID_DROP_REASONS: set[str] = {
    # From rule_table.csv
    "decode_failure",
    "too_short",
    "excessive_silence",
    "severe_distortion",
    "subtype_out_of_scope",
    "label_requires_guess",
    "exact_duplicate",
    "redundant_near_duplicate",
    "phrase_quota_downsampled",
    "class_replaced",
    # Extended values
    "mapping_failed",
    "unrecognized_encoding",
    "class_out_of_scope",
    "class_excluded",
}

# Allowed drop_reason per review_type when rejected
_ALLOWED_DROP_REASONS_BY_REVIEW_TYPE: dict[str, set[str]] = {
    "content_type":    {"label_requires_guess"},
    "i10_encoding":    {"unrecognized_encoding"},
    "d09_cross":       {"exact_duplicate"},
}

# Applicable optional fields per review_type
_APPLICABLE_OPTIONAL_FIELDS: dict[str, set[str]] = {
    "content_type":    {"content_type_override"},
    "i10_encoding":    {"bit_depth_override"},
    "d09_cross":       {"peer_record_id", "canonical_record_id"},
    "g02_fingerprint": {"peer_record_id"},
    "manual_merge":    {"peer_record_id"},
}

# pending_review_queue.csv columns
PENDING_REVIEW_QUEUE_COLUMNS: list[str] = [
    "record_id", "review_type", "source_dataset", "family_label",
    "file_name", "source_path", "trigger_rule_id", "trigger_detail", "peer_record_id",
]

# take_id mapping from filenames
_TAKE_LABEL_MAP: dict[str, int] = {
    "第一遍": 1,
    "第二遍": 2,
    "第三遍": 3,
}


# Helpers

def _resolve_path(config_dir: Path, raw_path: str) -> Path:
    """Resolve config path to absolute. Relative paths use config dir as base."""
    p = Path(raw_path)
    if p.is_absolute():
        return p.resolve()
    return (config_dir / p).resolve()


def _format_value(value: Any) -> str:
    """Convert Python value to CSV string. Bool->lowercase, None->empty, float->decimal."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == -1.0:
            return "-1.0"
        # f-string extra
        formatted = f"{value:.6f}".rstrip("0")
        if formatted.endswith("."):
            formatted += "0"
        return formatted
    return str(value)


def _write_manifest(
    row: list[dict[str, Any]],
    columns: list[str],
    output_path: Path,
) -> None:
    """Write manifest row to CSV. UTF-8, LF, QUOTE_MINIMAL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(columns)
        for row in row:
            writer.writerow([_format_value(row.get(col, "")) for col in columns])
    logger.info(" Write %s %d row ", output_path, len(rows))


# ChMusic filename parsing and sorting

def parse_chmusic_class_id(file_name: str) -> int:
    """Parse class_id from ChMusic filename. Format: <class_id>.<seq>.wav"""
    stem = file_name.rsplit(".wav", 1)[0]
    parts = stem.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"mapping_failed: {file_name}")
    return int(parts[0])


def chmusic_sort_key(file_name: str) -> tuple[int, ...]:
    """Sort key for ChMusic filenames. E.g. 1.3.wav -> (1, 3)."""
    stem = file_name.rsplit(".wav", 1)[0]
    return tuple(int(t) for t in stem.split("."))


# CCMusic filename normalization

def normalize_filename(filename: str) -> str:
    """Normalize CCMusic filename: strip ext, remove take suffix, collapse whitespace."""
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"\s*第[一二三]遍\s*$", "", name)
    name = re.sub(r"[\s\u3000]+", " ", name)
    return name.strip()


# content_type classification

def classify_content_type(file_name: str) -> Optional[str]:
    """Classify content_type by CCMusic filename keywords. Returns None if no match."""
    for content_type, keywords in _CONTENT_TYPE_KEYWORDS:
        for kw in keywords:
            if kw in file_name:
                return content_type
    return None


# Whitelist lookup construction

def build_ccmusic_whitelist_lookup(
    whitelist: dict[str, list[str]],
) -> dict[str, str]:
    """Flatten ccmusic_whitelist to {subtype_dir: family_label} dict."""
    lookup: dict[str, str] = {}
    for family_label, subtypes in whitelist.items():
        for subtype_dir in subtypes:
            if subtype_dir in lookup:
                raise ValueError(
                    f"Whitelist conflict: subtype '{subtype_dir}' maps to both "
                    f"'{lookup[subtype_dir]}' and '{family_label}'"
                )
            lookup[subtype_dir] = family_label
    return lookup


def build_chmusic_id_lookup(
    enabled_ids: list[int],
    excluded_ids: list[int],
    id_map: dict[int, str],
) -> dict[int, tuple[str, str, str]]:
    """Build ChMusic class_id lookup table."""
    lookup: dict[int, tuple[str, str, str]] = {}
    # Mark excluded first
    for cid in excluded_ids:
        lookup[cid] = ("out_of_scope", "", "class_excluded")
    # Mark enabled
    for cid in enabled_ids:
        if cid in id_map:
            lookup[cid] = ("in_scope", id_map[cid], "")
        else:
            # enabled but no mapping — config error
            raise ValueError(
                f"chmusic_enabled_ids contains {cid}，but no mapping in chmusic_id_map"
            )
    return lookup


# Step 0: Preflight

def step0_preflight(config_path: Path) -> dict[str, Any]:
    """Preflight checks: env, deps, config, smoke test. Returns config dict."""
    logger.info("=" * 60)
    logger.info("Step 0: Preflight")
    logger.info("=" * 60)

    errors: list[str] = []
    config_dir = config_path.parent.resolve()

    # 0.1 Python version check
    # Check current process version
    py_ver = sys.version_info
    logger.info("Python version: %d.%d.%d", py_ver.major, py_ver.minor, py_ver.micro)
    if not (3, 9) <= (py_ver.major, py_ver.minor) < (3, 13):
        errors.append(
            f"Python version OK >= 3.9 < 3.13 "
            f"got {py_ver.major}.{py_ver.minor}.{py_ver.micro}"
        )

    # 0.2 Dependency version check
    _check_dependency_versions(errors)

    # 0.3 fpcalc version check
    _check_fpcalc_version(errors)

    # 0.4 Load config.yaml
    import yaml

    if not config_path.is_file():
        errors.append(f"config.yaml not found: {config_path}")
        _abort_if_errors(errors)

    with config_path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.info("Loaded config.yaml: %s", config_path)

    # 0.4a Validate python_interpreter
    configured_interp = config.get("python_interpreter", "")
    if configured_interp:
        logger.info("config.yaml python_interpreter: %s", configured_interp)
        try:
            result = subprocess.run(
                [configured_interp, "-c",
                 "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                errors.append(
                    f"python_interpreter '{configured_interp}' execution failed: "
                    f"{result.stderr.strip()}"
                )
            else:
                interp_ver_str = result.stdout.strip()
                parts = [int(x) for x in interp_ver_str.split(".")]
                if not (3, 9) <= (parts[0], parts[1]) < (3, 13):
                    errors.append(
                        f"python_interpreter '{configured_interp}' version {interp_ver_str} "
                        f"unsupported (need >= 3.9 and < 3.13)"
                    )
                else:
                    logger.info(
                        "python_interpreter version: %s（OK）", interp_ver_str,
                    )
        except FileNotFoundError:
            errors.append(
                f"python_interpreter '{configured_interp}' not found PATH "
            )
        except subprocess.TimeoutExpired:
            errors.append(
                f"python_interpreter '{configured_interp}' version "
            )
    else:
        logger.warning("config.yaml python_interpreter version ")

    # 0.5 Validate data root paths
    data_root_paths: dict[str, str] = config.get("data_root_paths", {})
    if not data_root_paths:
        errors.append("config.yaml missing data_root_paths")
    else:
        # CCMusic and ChMusic External
        for required_ds in ("CCMusic", "ChMusic"):
            if required_ds not in data_root_paths:
                errors.append(
                    f"data_root_paths missing required key '{required_ds}'"
                )
        # Valid keys
        _ALLOWED_DS_KEYS = {"CCMusic", "ChMusic", "External"}
        for ds_name in data_root_paths:
            if ds_name not in _ALLOWED_DS_KEYS:
                errors.append(
                    f"data_root_paths contains unknown key '{ds_name}'（valid keys: {_ALLOWED_DS_KEYS}）"
                )
    for ds_name, ds_path_str in data_root_paths.items():
        ds_path = Path(ds_path_str)
        if not ds_path.is_absolute():
            errors.append(f"data_root_paths.{ds_name} must be absolute path: {ds_path_str}")
        elif not ds_path.is_dir():
            errors.append(f"data_root_paths.{ds_name} not found readable: {ds_path_str}")
        else:
            logger.info("Data root %s: %s（readable）", ds_name, ds_path_str)

    # 0.6 Resolve output dir
    output_dir = _resolve_path(config_dir, config.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # 0.7 Load and validate freeze_config.yaml
    freeze_config_path = config_dir / "freeze_config.yaml"
    if not freeze_config_path.is_file():
        errors.append(f"freeze_config.yaml not found: {freeze_config_path}")
        _abort_if_errors(errors)

    with freeze_config_path.open("r", encoding="utf-8") as f:
        freeze_config: dict[str, Any] = yaml.safe_load(f)

    logger.info("Loaded freeze_config.yaml: %s", freeze_config_path)
    _validate_freeze_config(freeze_config, errors)

    # 0.8 Load manual_review.csv (if exists)
    manual_review_path = config_dir / "manual_review.csv"
    manual_review: list[dict[str, str]] = []
    if manual_review_path.is_file():
        manual_review = _load_and_validate_manual_review(manual_review_path, errors)
        logger.info("Loaded manual_review.csv: %d row", len(manual_review))
    else:
        logger.info("manual_review.csv not found survey rows ")

    # 0.9 Collect errors before abort
    _abort_if_errors(errors)

    # 0.10 Smoke test: write -> read -> resample
    _smoke_test()

    logger.info("Step 0 preflight passed")

    return {
        "config": config,
        "freeze_config": freeze_config,
        "manual_review": manual_review,
        "output_dir": output_dir,
        "config_dir": config_dir,
    }


def _check_dependency_versions(errors: list[str]) -> None:
    """ Python version """
    deps: list[tuple[str, str, str]] = [
        ("librosa", "0.10.0", "librosa"),
        ("soundfile", "0.12.0", "soundfile"),
        ("soxr", "0.3.0", "soxr"),
    ]
    from importlib.metadata import version as pkg_version, PackageNotFoundError

    for display_name, min_ver, pkg_name in deps:
        try:
            installed = pkg_version(pkg_name)
            if _compare_versions(installed, min_ver) < 0:
                errors.append(
                    f"{display_name} version >= {min_ver} got {installed}"
                )
            else:
                logger.info("%s version: %s >= %s ", display_name, installed, min_ver)
        except PackageNotFoundError:
            errors.append(f"{display_name} not installed")


def _compare_versions(a: str, b: str) -> int:
    """ version -1/0/1

    0.10.1 0.10.0 pre-release
    """
    def _parse(v: str) -> list[int]:
        # Numeric part only, ignore rc/dev/post
        clean = re.match(r"[\d.]+", v)
        raw = clean.group().rstrip(".") if clean else v
        return [int(x) for x in raw.split(".") if x]

    pa, pb = _parse(a), _parse(b)
    # Pad to equal length
    max_len = max(len(pa), len(pb))
    pa.extend([0] * (max_len - len(pa)))
    pb.extend([0] * (max_len - len(pb)))
    for x, y in zip(pa, pb):
        if x < y:
            return -1
        if x > y:
            return 1
    return 0


def _check_fpcalc_version(errors: list[str]) -> None:
    """ fpcalc version >= 1.5 """
    try:
        result = subprocess.run(
            ["fpcalc", "-v"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # fpcalc -v output: fpcalc version X.Y.Z (...)
        output = result.stdout.strip() or result.stderr.strip()
        match = re.search(r"(\d+\.\d+(?:\.\d+)?)", output)
        if match:
            ver = match.group(1)
            if _compare_versions(ver, "1.5") < 0:
                errors.append(f"fpcalc version >= 1.5 got {ver}")
            else:
                logger.info("fpcalc version: %s >= 1.5 ", ver)
        else:
            errors.append(f" fpcalc version : {output}")
    except FileNotFoundError:
        errors.append("fpcalc not installed PATH ")
    except subprocess.TimeoutExpired:
        errors.append("fpcalc -v rows ")


def _validate_freeze_config(fc: dict[str, Any], errors: list[str]) -> None:
    """ freeze_config.yaml

    - Required fields
    - active_classes and
    - thresholds and value
    - split_seeds
    - chmusic_enabled_ids / excluded_ids
    """
    # Required fields
    for key in _FREEZE_REQUIRED_KEYS:
        if key not in fc:
            errors.append(f"freeze_config.yaml missing required field: {key}")

    # Cannot continue if key fields missing
    if any(key not in fc for key in _FREEZE_REQUIRED_KEYS):
        return

    # active_classes validation
    ac = fc["active_classes"]
    if not isinstance(ac, list) or len(ac) != 6:
        errors.append(f"active_classes 6 got {len(ac) if isinstance(ac, list) else type(ac)}")
    elif len(set(ac)) != 6:
        errors.append("active_classes contains duplicates")

    # family_abbr: must cover active_classes
    fa = fc["family_abbr"]
    if not isinstance(fa, dict):
        errors.append("family_abbr must be dict")
    elif isinstance(ac, list):
        missing = set(ac) - set(fa.keys())
        if missing:
            errors.append(f"family_abbr missing classes from active_classes: {missing}")

    # thresholds validation
    th = fc["thresholds"]
    if not isinstance(th, dict):
        errors.append("thresholds must be dict")
    else:
        _validate_threshold_value(th, "d05a_default", 0.0, 1.0, errors)
        _validate_threshold_value(th, "d06a_default", 0.0, 1.0, errors)
        _validate_threshold_value(th, "active_frame_ratio", 0.0, 1.0, errors)
        _validate_threshold_value(th, "phrase_ratio_cap", 0.0, 1.0, errors)
        _validate_threshold_value(th, "concentration_cap_floor", 1, 50, errors)

    # split_seeds validation
    seeds = fc["split_seeds"]
    if not isinstance(seeds, list) or len(seeds) != 3:
        errors.append(f"split_seeds 3 value got {seeds}")

    # seed_namespaces validation
    sn = fc["seed_namespaces"]
    if not isinstance(sn, dict):
        errors.append("seed_namespaces must be dict")
    else:
        for key in ("split", "phrase_downsampling"):
            if key not in sn:
                errors.append(f"seed_namespaces missing field: {key}")

    # chmusic enabled/excluded mutual exclusion
    enabled = set(fc.get("chmusic_enabled_ids", []))
    excluded = set(fc.get("chmusic_excluded_ids", []))
    overlap = enabled & excluded
    if overlap:
        errors.append(f"chmusic_enabled_ids chmusic_excluded_ids overlap: {overlap}")

    # chmusic_id_map: every enabled ID must have mapping
    id_map = fc.get("chmusic_id_map", {})
    if id_map:
        for cid in enabled:
            if cid not in id_map:
                errors.append(
                    f"chmusic_id_map missing enabled ID {cid}  mapping"
                )
            else:
                mapped_fl = id_map[cid]
                if isinstance(ac, list) and mapped_fl not in set(ac) and mapped_fl not in set(fc.get("family_abbr", {}).keys()):
                    errors.append(
                        f"chmusic_id_map[{cid}] maps to '{mapped_fl}'，not in known family_label set"
                    )


def _validate_threshold_value(
    th: dict[str, Any],
    key: str,
    low: float,
    high: float,
    errors: list[str],
) -> None:
    """ value and """
    if key not in th:
        errors.append(f"thresholds missing field: {key}")
        return
    val = th[key]
    if not isinstance(val, (int, float)):
        errors.append(f"thresholds.{key} value got {type(val).__name__}")
        return
    if not (low <= val <= high):
        errors.append(f"thresholds.{key} value {val} out of range [{low}, {high}]")


def _normalize_pair_key(rid: str, rt: str, peer: str) -> tuple[str, str, str]:
    """ key (min_id, review_type, max_id)

    maps to key
    (A, rt, B) and (B, rt, A)
    """
    return (min(rid, peer), rt, max(rid, peer))


def _load_and_validate_manual_review(
    path: Path,
    errors: list[str],
) -> list[dict[str, str]]:
    """Load and validate manual_review.csv（）。

    14
    Basic validation 4
    -
    - review_type
    - decision
    - Required fields

    10 [SS]5.4
    1. (record_id, review_type)
    2. (min(rid, peer), review_type, max(rid, peer))
    3. peer_record_id != record_id
    4. peer_record_id
    5. d09_cross + approved: canonical_record_id rid peer
    6. i10_encoding + approved: bit_depth_override 16/24/32
    7. content_type + approved: content_type_override controlled/technique/phrase
    8. rejected drop review_type: drop_reason
       g02_fingerprint/manual_merge rejected: drop_reason should be empty
    9. drop_reason value §16.3
    10. §5.4: peer_record_id/canonical_record_id/bit_depth_override/content_type_override
    """
    row: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            errors.append("manual_review.csv is empty")
            return row
        # Validate header
        expected = set(_MANUAL_REVIEW_COLUMNS)
        actual = set(reader.fieldnames)
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            msg = "manual_review.csv header mismatch"
            if missing:
                msg += f"；missing: {missing}"
            if extra:
                msg += f"；extra: {extra}"
            errors.append(msg)
            return row

        # 1 and 2
        seen_single: set[tuple[str, str]] = set()
        seen_pair: set[tuple[str, str, str]] = set()

        for i, row in enumerate(reader, start=2): # 2 row 1 row
            line_prefix = f"manual_review.csv row {i} row"

            # --- Basic validation ---
            # Required fields
            for col in ("record_id", "review_type", "decision", "justification", "reviewer"):
                if not row.get(col, "").strip():
                    errors.append(f"{line_prefix}: {col} must not be empty")
            # review_type enum
            rt = row.get("review_type", "").strip()
            if rt and rt not in _VALID_REVIEW_TYPES:
                errors.append(f"{line_prefix}: review_type '{rt}' not in valid enum")
            # decision enum
            dec = row.get("decision", "").strip()
            if dec and dec not in _VALID_DECISIONS:
                errors.append(f"{line_prefix}: decision '{dec}' not in valid enum")

            rid = row.get("record_id", "").strip()
            peer = row.get("peer_record_id", "").strip()

            # --- Hard check 1 & 2: unique keys ---
            if rt and rid:
                if rt in _PAIR_REVIEW_TYPES:
                    # Check 4: pair review requires peer_record_id
                    if not peer:
                        errors.append(
                            f"{line_prefix}: review_type '{rt}' peer_record_id must not be empty"
                        )
                    else:
                        # Check 3: no self-reference
                        if peer == rid:
                            errors.append(
                                f"{line_prefix}: pair review: peer must differ from record_id"
                            )
                        # Check 2: pair unique key
                        pair_key = _normalize_pair_key(rid, rt, peer)
                        if pair_key in seen_pair:
                            errors.append(
                                f"{line_prefix}: duplicate pair key "
                                f"({pair_key[0]}, {pair_key[1]}, {pair_key[2]})"
                            )
                        seen_pair.add(pair_key)
                else:
                    # Check 1: single-sample unique key
                    single_key = (rid, rt)
                    if single_key in seen_single:
                        errors.append(
                            f"{line_prefix}: duplicate single-sample key ({rid}, {rt})"
                        )
                    seen_single.add(single_key)

            # --- Hard check 5: d09_cross + approved -> canonical_record_id ---
            if rt == "d09_cross" and dec == "approved":
                canonical = row.get("canonical_record_id", "").strip()
                if not canonical:
                    errors.append(
                        f"{line_prefix}: d09_cross + approved: canonical_record_id must not be empty"
                    )
                elif canonical not in (rid, peer):
                    errors.append(
                        f"{line_prefix}: d09_cross + approved: canonical_record_id "
                        f"'{canonical}' must equal record_id or peer_record_id"
                    )

            # --- Hard check 6: i10_encoding + approved -> bit_depth_override ---
            if rt == "i10_encoding" and dec == "approved":
                bd = row.get("bit_depth_override", "").strip()
                if not bd:
                    errors.append(
                        f"{line_prefix}: i10_encoding + approved: bit_depth_override must not be empty"
                    )
                elif bd not in ("16", "24", "32"):
                    errors.append(
                        f"{line_prefix}: i10_encoding + approved: bit_depth_override "
                        f"'{bd}' must be 16/24/32"
                    )

            # --- Hard check 7: content_type + approved -> content_type_override ---
            if rt == "content_type" and dec == "approved":
                ct = row.get("content_type_override", "").strip()
                if not ct:
                    errors.append(
                        f"{line_prefix}: content_type + approved: content_type_override must not be empty"
                    )
                elif ct not in ("controlled", "technique", "phrase"):
                    errors.append(
                        f"{line_prefix}: content_type + approved: content_type_override "
                        f"'{ct}' must be controlled/technique/phrase"
                    )

            # --- Hard check 8: rejected + drop -> drop_reason required ---
            if dec == "rejected":
                drop_reason = row.get("drop_reason", "").strip()
                if rt in _DROP_ON_REJECT_REVIEW_TYPES:
                    # These rejected cause drop, drop_reason required
                    if not drop_reason:
                        errors.append(
                            f"{line_prefix}: review_type '{rt}' rejected: drop_reason must not be empty"
                        )
                    elif drop_reason in _VALID_DROP_REASONS:
                        # Valid globally, check review_type-specific subset
                        allowed_subset = _ALLOWED_DROP_REASONS_BY_REVIEW_TYPE.get(rt)
                        if allowed_subset is not None and drop_reason not in allowed_subset:
                            errors.append(
                                f"{line_prefix}: review_type '{rt}' rejected: "
                                f"drop_reason must be {sorted(allowed_subset)}，"
                                f" value '{drop_reason}'"
                            )
                else:
                    # g02/manual_merge rejected: only affects grouping, drop_reason should be empty
                    if drop_reason:
                        errors.append(
                            f"{line_prefix}: review_type '{rt}' rejected: drop_reason should be empty"
                            f"（only affects grouping, not drop）"
                        )

            # --- 9: drop_reason value §16.3 ---
            drop_reason_val = row.get("drop_reason", "").strip()
            if drop_reason_val and drop_reason_val not in _VALID_DROP_REASONS:
                errors.append(
                    f"{line_prefix}: drop_reason '{drop_reason_val}' not in valid enum"
                )

            # --- Hard check 10: inapplicable fields must be empty ---
            if rt in _APPLICABLE_OPTIONAL_FIELDS:
                # Optional fields not in this review_type must be empty
                optional_fields = {
                    "peer_record_id", "canonical_record_id",
                    "bit_depth_override", "content_type_override",
                }
                allowed = _APPLICABLE_OPTIONAL_FIELDS[rt]
                for field in optional_fields - allowed:
                    val = row.get(field, "").strip()
                    if val:
                        errors.append(
                            f"{line_prefix}: review_type '{rt}' {field} should be empty, "
                            f" value '{val}'(inapplicable field)"
                        )

            row.append(row)

    return row


def _smoke_test() -> None:
    """ write → read → resample

    sine Write WAV file librosa Read back
    rows soxr_hq Resample and
    """
    import librosa
    import soundfile as sf

    logger.info(" rows : write → read → resample ...")

    orig_sr = 44100
    target_sr = 24000
    duration = 0.5  # s
    t = np.linspace(0, duration, int(orig_sr * duration), endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "smoke_test.wav"

        # Write
        sf.write(str(wav_path), tone, orig_sr, subtype="PCM_16")

        # Read back
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        assert sr == orig_sr, f"Read backSample rate {orig_sr} got {sr}"
        assert len(y) > 0, "Read back "

        # Resample
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
        expected_len = int(len(y) * target_sr / orig_sr)
        # ±2 Resample
        assert abs(len(y_resampled) - expected_len) <= 2, (
            f"Resample {expected_len} got {len(y_resampled)}"
        )

        # Write-back verify
        out_path = Path(tmpdir) / "smoke_resampled.wav"
        sf.write(str(out_path), y_resampled, target_sr, subtype="PCM_16")
        info = sf.info(str(out_path))
        assert info.samplerate == target_sr, f"Export SR mismatch: expected {target_sr}，got {info.samplerate}"

    logger.info("Smoke test passed")


def _abort_if_errors(errors: list[str]) -> None:
    """Print error summary and abort if errors."""
    if not errors:
        return
    logger.error("Preflight failed, %d  errors:", len(errors))
    for i, err in enumerate(errors, 1):
        logger.error("  [%d] %s", i, err)
    sys.exit(1)


# Step 1: Scan + record_id

def step1_scan(
    config: dict[str, Any],
    freeze_config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Scan audio directories, assign record_id, output scan_manifest.csv。

    - CCMusic: 1 subtype Unicode code point
      .wav file Unicode code point
    - ChMusic: 0 .wav fileSort by dot-numeric key B

    record_id {source}_{seq:06d} seq 1 source_dataset
    rows (source_dataset, raw_dir_label, file_name) CCMusic

    Args:
        config: config.yaml
        freeze_config: Parsed freeze_config.yaml
        output_dir: Output dir

    Returns:
        scan_manifest rows
    """
    logger.info("=" * 60)
    logger.info("Step 1: Scan + record_id")
    logger.info("=" * 60)

    data_roots: dict[str, str] = config["data_root_paths"]

    # Build whitelist lookup
    ccm_whitelist_lookup = build_ccmusic_whitelist_lookup(
        freeze_config["ccmusic_whitelist"]
    )
    chm_id_lookup = build_chmusic_id_lookup(
        freeze_config["chmusic_enabled_ids"],
        freeze_config["chmusic_excluded_ids"],
        freeze_config.get("chmusic_id_map", {}),
    )

    # Reverse family_abbr for External scanning
    family_abbr: dict[str, str] = freeze_config["family_abbr"]
    abbr_to_family: dict[str, str] = {v: k for k, v in family_abbr.items()}
    active_classes_set: set[str] = set(freeze_config["active_classes"])

    # 1.1 Scan CCMusic (depth 1)
    ccm_root = Path(data_roots["CCMusic"])
    ccm_rows = _scan_ccmusic(ccm_root, ccm_whitelist_lookup)
    logger.info("CCMusic scan done: %d  files", len(ccm_rows))

    # 1.2 Scan ChMusic (depth 0)
    chm_root = Path(data_roots["ChMusic"])
    chm_rows = _scan_chmusic(chm_root, chm_id_lookup)
    logger.info("ChMusic scan done: %d  files", len(chm_rows))

    # 1.2b Scan External (optional)
    ext_rows: list[dict[str, Any]] = []
    if "External" in data_roots:
        ext_root = Path(data_roots["External"])
        ext_rows = _scan_external(ext_root, abbr_to_family, active_classes_set)
        logger.info("External scan done: %d  files", len(ext_rows))

    # 1.3 Assign record_id
    # CCMusic: assign in (raw_dir_label, file_name) order
    for seq, row in enumerate(ccm_rows, start=1):
        row["record_id"] = f"ccm_{seq:06d}"

    # ChMusic: assign in dot-numeric order
    for seq, row in enumerate(chm_rows, start=1):
        row["record_id"] = f"chm_{seq:06d}"

    # External: assign in (raw_dir_label, file_name) order
    for seq, row in enumerate(ext_rows, start=1):
        row["record_id"] = f"ext_{seq:06d}"

    # 1.4 Merge and sort
    # row (source_dataset ASC, raw_dir_label ASC, file_name ASC)
    all_rows = ccm_rows + chm_rows + ext_rows
    # Both datasets already sorted internally,
    # final sort ensures correctness
    # —— 
    # 
    all_rows.sort(key=lambda r: (r["source_dataset"], r["raw_dir_label"], r["file_name"]))

    # 1.5 Output scan_manifest.csv
    scan_path = output_dir / "scan_manifest.csv"
    _write_manifest(all_rows, SCAN_MANIFEST_COLUMNS, scan_path)

    # Stats
    in_scope = sum(1 for r in all_rows if r["scan_status"] == "in_scope")
    logger.info("Scan summary: total %d files in_scope %d ", len(all_rows), in_scope)

    return all_rows


def _scan_ccmusic(
    root: Path,
    whitelist_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    """ CCMusic 1

    Unicode code point Python sorted row
    file Unicode code point

    Args:
        root: CCMusic Data root
        whitelist_lookup: {subtype_dir: family_label}

    Returns:
        (raw_dir_label, file_name) rows record_id assigned later
    """
    row: list[dict[str, Any]] = []

    # Get subdirs sorted by name
    subdirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    for subdir in subdirs:
        subtype_dir_name = subdir.name
        # Get .wav files sorted by name
        wav_files = sorted(
            [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"],
            key=lambda f: f.name,
        )

        for wav_file in wav_files:
            # Determine scan_status
            if subtype_dir_name in whitelist_lookup:
                scan_status = "in_scope"
                mapped_family = whitelist_lookup[subtype_dir_name]
                drop_reason = ""
            else:
                scan_status = "out_of_scope"
                mapped_family = ""
                drop_reason = "subtype_out_of_scope"

            row.append({
                "record_id": "",  # assigned later
                "source_dataset": "CCMusic",
                "source_path": str(wav_file.resolve()),
                "file_name": wav_file.name,
                "file_ext": ".wav",
                "raw_dir_label": subtype_dir_name,
                "scan_status": scan_status,
                "mapped_family_label": mapped_family,
                "drop_reason": drop_reason,
            })

    return row


def _scan_chmusic(
    root: Path,
    id_lookup: dict[int, tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """ ChMusic 0

    file Sort by dot-numeric key B chmusic_sort_key

    Args:
        root: ChMusic Data root Musics
        id_lookup: {class_id: (scan_status, family_label, drop_reason)}

    Returns:
        chmusic_sort_key rows record_id assigned later
    """
    row: list[dict[str, Any]] = []

    # Get all .wav files
    wav_files = [f for f in root.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]

    # Sort by dot-numeric key
    wav_files.sort(key=lambda f: chmusic_sort_key(f.name))

    # raw_dir_label is root dir name
    raw_dir_label = root.name

    for wav_file in wav_files:
        file_name = wav_file.name
        try:
            class_id = parse_chmusic_class_id(file_name)
        except ValueError:
            # Filename format error -> mapping_failed
            row.append({
                "record_id": "",
                "source_dataset": "ChMusic",
                "source_path": str(wav_file.resolve()),
                "file_name": file_name,
                "file_ext": ".wav",
                "raw_dir_label": raw_dir_label,
                "scan_status": "mapping_failed",
                "mapped_family_label": "",
                "drop_reason": "mapping_failed",
            })
            continue

        if class_id in id_lookup:
            scan_status, family_label, drop_reason = id_lookup[class_id]
            mapped_family = family_label # in_scope value out_of_scope
        else:
            # class_id not in enabled/excluded -> out_of_scope
            scan_status = "out_of_scope"
            mapped_family = ""
            drop_reason = "class_out_of_scope"

        row.append({
            "record_id": "",
            "source_dataset": "ChMusic",
            "source_path": str(wav_file.resolve()),
            "file_name": file_name,
            "file_ext": ".wav",
            "raw_dir_label": raw_dir_label,
            "scan_status": scan_status,
            "mapped_family_label": mapped_family,
            "drop_reason": drop_reason,
        })

    return row


def _scan_external(
    root: Path,
    abbr_to_family: dict[str, str],
    active_classes: set[str],
) -> list[dict[str, Any]]:
    """Scan external data directory（<root>/<family_abbr>/tracks/*.wav）。

        <root>/
        ├── guzheng/
        │   └── tracks/
        │       ├── 专辑名_track_01.wav
        │       └── ...
        └── pipa/
            └── tracks/
                └── ...

    freeze_config.yaml family_abbr value
    

    Args:
        root: Data root
        abbr_to_family: {family_abbr: family_label} reverse map
        active_classes: Active class set

    Returns:
        (raw_dir_label, file_name) rows record_id assigned later
    """
    row: list[dict[str, Any]] = []

    subdirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    for subdir in subdirs:
        dir_name = subdir.name
        tracks_dir = subdir / "tracks"
        if not tracks_dir.is_dir():
            logger.debug("  Skip %s（no tracks/ subdir）", subdir)
            continue

        wav_files = sorted(
            [f for f in tracks_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"],
            key=lambda f: f.name,
        )

        if not wav_files:
            continue

        # Map dir_name -> family_label
        if dir_name in abbr_to_family:
            family_label = abbr_to_family[dir_name]
            if family_label in active_classes:
                scan_status = "in_scope"
                mapped_family = family_label
                drop_reason = ""
            else:
                scan_status = "out_of_scope"
                mapped_family = ""
                drop_reason = "class_not_active"
        else:
            scan_status = "out_of_scope"
            mapped_family = ""
            drop_reason = "unknown_family_abbr"

        for wav_file in wav_files:
            row.append({
                "record_id": "",  # assigned later
                "source_dataset": "External",
                "source_path": str(wav_file.resolve()),
                "file_name": wav_file.name,
                "file_ext": ".wav",
                "raw_dir_label": dir_name,
                "scan_status": scan_status,
                "mapped_family_label": mapped_family,
                "drop_reason": drop_reason,
            })

    return row


# Step 2: Metadata extraction

def step2_metadata(
    scan_rows: list[dict[str, Any]],
    freeze_config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """ scan_manifest in_scope row audit_manifest

    
    - Field mapping（raw_dir_label -> source_subtype_dir、mapped_family_label -> family_label）
    - Audio property extraction（duration, sample_rate, channels, bit_depth）
    - QC metric computation（silence_ratio, clipping_ratio）
    - File hash computation（audio_sha256）
    - content_type classification
    - normalized_base_name generation
    - Default field init

    Args:
        scan_rows: step1_scan scan_manifest row
        freeze_config: Parsed freeze_config.yaml
        output_dir: Output dir

    Returns:
        audit_manifest rows
    """
    import soundfile as sf
    from tqdm import tqdm

    logger.info("=" * 60)
    logger.info("Step 2: Metadata extraction")
    logger.info("=" * 60)

    # in_scope row
    in_scope_rows = [r for r in scan_rows if r["scan_status"] == "in_scope"]
    logger.info("in_scope samples: %d", len(in_scope_rows))

    # value silence/clipping got value Step 3
    thresholds = freeze_config["thresholds"]
    clipping_abs_threshold = thresholds.get("clipping_abs_threshold", 0.998)

    # ChMusic id_map for chmusic_class_id
    chm_id_map: dict[int, str] = {
        int(k): v for k, v in freeze_config.get("chmusic_id_map", {}).items()
    }

    audit_rows: list[dict[str, Any]] = []

    for scan_row in tqdm(in_scope_rows, desc="Extract metadata", unit="file"):
        audit_row = _init_audit_row(scan_row, freeze_config)

        source_path = Path(scan_row["source_path"])

        # 2.1 audio_sha256 file SHA-256
        try:
            file_bytes = source_path.read_bytes()
            audit_row["audio_sha256"] = hashlib.sha256(file_bytes).hexdigest()
        except OSError as e:
            logger.warning(" file SHA-256 : %s — %s", source_path, e)
            audit_row["is_decodable"] = False
            _fill_undecodable_defaults(audit_row)
            audit_rows.append(audit_row)
            continue

        # 2.2 Extract audio properties
        try:
            info = sf.info(str(source_path))
            audit_row["duration_sec"] = info.duration
            audit_row["sample_rate_hz"] = info.samplerate
            audit_row["channels"] = info.channels
            audit_row["bit_depth"] = _SUBTYPE_TO_BIT_DEPTH.get(info.subtype, -1)
            audit_row["is_decodable"] = True
        except Exception as e:
            logger.warning("soundfile.info failed: %s — %s", source_path, e)
            audit_row["is_decodable"] = False
            _fill_undecodable_defaults(audit_row)
            audit_rows.append(audit_row)
            continue

        # 2.3 Load waveform and compute QC metrics
        try:
            y, sr = _load_audio(source_path)
        except Exception as e:
            logger.warning("Audio decode failed: %s — %s", source_path, e)
            audit_row["is_decodable"] = False
            _fill_undecodable_defaults(audit_row)
            audit_rows.append(audit_row)
            continue

        # Silence ratio
        audit_row["silence_ratio"] = _compute_silence_ratio(y, sr)

        # Clipping ratio
        audit_row["clipping_ratio"] = _compute_clipping_ratio(y, clipping_abs_threshold)

        audit_rows.append(audit_row)

    # 2.4 Sort（sample_id not yet assigned,
    #     use record_id order for now）
    # audit_manifest rows sample_id ASC §14.2
    # 
    audit_rows.sort(key=lambda r: r["record_id"])

    # 2.5 Output audit_manifest.csv
    audit_path = output_dir / "audit_manifest.csv"
    _write_manifest(audit_rows, AUDIT_MANIFEST_COLUMNS, audit_path)

    # Stats
    decodable = sum(1 for r in audit_rows if r["is_decodable"] is True)
    logger.info(
        "Metadata extraction done: %d %d %d ",
        len(audit_rows), decodable, len(audit_rows) - decodable,
    )

    return audit_rows


def _init_audit_row(
    scan_row: dict[str, Any],
    freeze_config: dict[str, Any],
) -> dict[str, Any]:
    """ scan_manifest row audit_manifest row

    rowsField mapping value

    Args:
        scan_row: scan_manifest rows
        freeze_config: Freeze config

    Returns:
        audit_manifest row
    """
    source_dataset = scan_row["source_dataset"]
    file_name = scan_row["file_name"]
    mapped_family = scan_row["mapped_family_label"]
    raw_dir_label = scan_row["raw_dir_label"]

    # Field mapping
    # raw_dir_label -> source_subtype_dir
    # CCMusic: value External: value family_abbr ChMusic:
    if source_dataset in _PRIMARY_DATASETS:
        source_subtype_dir = raw_dir_label
    else:
        source_subtype_dir = ""

    # mapped_family_label -> family_label
    family_label = mapped_family

    # subtype_label: CCMusic=subtype_dir, others=family_label
    if source_dataset == "CCMusic":
        subtype_label = source_subtype_dir
    else:
        subtype_label = family_label

    # chmusic_class_id: ChMusic=parsed, others=-1
    if source_dataset == "ChMusic":
        try:
            chmusic_class_id = parse_chmusic_class_id(file_name)
        except ValueError:
            chmusic_class_id = -1  # should not reach here
    else:
        chmusic_class_id = -1

    # content_type classification
    if source_dataset == "CCMusic":
        content_type = classify_content_type(file_name)
        if content_type is None:
            # No match -> empty (flagged later)
            content_type = ""
    else:
        # External/ChMusic: default phrase
        content_type = "phrase"

    # normalized_base_name
    if source_dataset in _PRIMARY_DATASETS:
        normalized_base_name = normalize_filename(file_name)
    else:
        # ChMusic: file
        normalized_base_name = file_name.rsplit(".wav", 1)[0] if file_name.endswith(".wav") else file_name.rsplit(".", 1)[0]

    return {
        # Inherited fields
        "record_id": scan_row["record_id"],
        "source_dataset": source_dataset,
        "source_path": scan_row["source_path"],
        "file_name": file_name,
        "file_ext": scan_row["file_ext"],
        # Mapped fields
        "source_subtype_dir": source_subtype_dir,
        "family_label": family_label,
        "subtype_label": subtype_label,
        "chmusic_class_id": chmusic_class_id,
        "content_type": content_type,
        "normalized_base_name": normalized_base_name,
        # Computed fields (Step 2)
        "audio_sha256": "",
        "duration_sec": -1.0,
        "sample_rate_hz": -1,
        "channels": -1,
        "bit_depth": -1,
        "silence_ratio": -1.0,
        "clipping_ratio": -1.0,
        "is_decodable": False,
        # Placeholder fields (later steps)
        "sample_id": "",                    # Step 7
        "audio_fingerprint_id": "",         # Step 5
        "performer_or_recording_id": "",    # Step 5
        "recording_group_id": "",           # Step 5
        "grouping_method": "",              # Step 5
        "take_id": 0,                       # Step 5
        "is_near_duplicate": False,         # Step 5
        "duplicate_of": "",                 # Step 5
        # Optional (empty)
        "integrated_loudness_lufs": "",
        "rms_db": "",
        "has_vocal": False,
        # Initial status
        "manual_review_required": False,
        "manual_review_status": "",
        "sample_status": "keep",            # in_scope default keep
        "drop_reason": "",
        "applied_rule_ids": "",
        "notes": "",
    }


def _fill_undecodable_defaults(row: dict[str, Any]) -> None:
    """ file value §13 Step 2

    is_decodable=false duration_sec=-1.0, sample_rate_hz=-1,
    channels=-1, bit_depth=-1, silence_ratio=-1.0, clipping_ratio=-1.0
    """
    row["duration_sec"] = -1.0
    row["sample_rate_hz"] = -1
    row["channels"] = -1
    row["bit_depth"] = -1
    row["silence_ratio"] = -1.0
    row["clipping_ratio"] = -1.0


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """ librosa file

    §17.2 Sample rate

    Args:
        path: file

    Returns:
        ((waveform, sample_rate))
    """
    import librosa
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y, sr


def _compute_silence_ratio(y: np.ndarray, sr: int) -> float:
    """Compute silence ratio. 25ms frames, 10ms hop, -40 dBFS threshold.

    Args:
        y: Mono float waveform
        sr: Sample rate

    Returns:
        silence_ratio (0.0 ~ 1.0)
    """
    import librosa

    # Frame params
    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms

    # Guard: waveform too short
    if len(y) < frame_length:
        return 1.0  # Entire segment < one frame = all silent

    # Frame
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

    # Per-frame RMS
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=0))

    # dB value log(0)
    rms_db = 20.0 * np.log10(rms_per_frame + 1e-10)

    # Silence frame detection
    silent_frames = rms_db < -40.0
    silence_ratio = float(silent_frames.sum()) / len(silent_frames)

    return silence_ratio


def _compute_clipping_ratio(y: np.ndarray, threshold: float = 0.998) -> float:
    """Compute clipping ratio: fraction of samples with abs >= threshold.

    Args:
        y: Normalized float waveform
        threshold: value 0.998

    Returns:
        clipping_ratio (0.0 ~ 1.0)
    """
    if len(y) == 0:
        return 0.0
    clipped_count = int(np.sum(np.abs(y) >= threshold))
    return float(clipped_count) / len(y)


# Step 3-5 Helpers

def _append_rule_id(row: dict[str, Any], rule_id: str) -> None:
    """ applied_rule_ids ID Skip

    rows rule_id §16.4
    """
    existing = row.get("applied_rule_ids", "") or ""
    ids = [rid for rid in existing.split(",") if rid] if existing else []
    if rule_id not in ids:
        ids.append(rule_id)
    row["applied_rule_ids"] = ",".join(ids)


def _resolve_threshold(
    family_label: str,
    source_dataset: str,
    threshold_key: str,
    default_value: float,
    freeze_config: dict[str, Any],
) -> float:
    """ value §7.4

    1. freeze_config.threshold_overrides per-family scope
    2. value

    Args:
        family_label: family_label
        source_dataset: CCMusic ChMusic
        threshold_key: value "d05a", "d06a"
        default_value: value
        freeze_config:

    Returns:
        value
    """
    overrides = freeze_config.get("threshold_overrides") or []
    if not overrides:
        return default_value

    for entry in overrides:
        if entry.get("family_label") != family_label:
            continue
        if threshold_key not in entry:
            continue
        scope = entry.get("scope", "all")
        # scope=chmusic_only ChMusic
        if scope == "chmusic_only" and source_dataset != "ChMusic":
            continue
        # scope=all scope=chmusic_only source_dataset==ChMusic →
        return float(entry[threshold_key])

    return default_value


def _add_pending_review(
    row: dict[str, Any],
    review_type: str,
    trigger_rule_id: str,
    trigger_detail: str,
    peer_record_id: str = "",
) -> None:
    """ rows

    _pending_reviews Step 4
    """
    if "_pending_reviews" not in row:
        row["_pending_reviews"] = []
    row["_pending_reviews"].append({
        "review_type": review_type,
        "trigger_rule_id": trigger_rule_id,
        "trigger_detail": trigger_detail,
        "peer_record_id": peer_record_id,
    })


def _write_pending_review_queue(
    items: list[dict[str, Any]],
    output_dir: Path,
) -> NoReturn:
    """ pending_review_queue.csv §5.5

    Args:
        items: PENDING_REVIEW_QUEUE_COLUMNS
        output_dir: Output dir
    """
    out_path = output_dir / "pending_review_queue.csv"
    _write_manifest(items, PENDING_REVIEW_QUEUE_COLUMNS, out_path)
    logger.error(
        " %d %s",
        len(items), out_path,
    )
    sys.exit(1)


def _parse_take_id(filename: str) -> int:
    """ file take_id §9.1 Stage A Step 3

    在原始file名（含扩展名前）搜索"第一遍/第二遍/第三遍"标记。
    0

    Args:
        filename: file

    Returns:
        take_id: 0 | 1 | 2 | 3
    """
    # " X "
    stem = filename.rsplit(".", 1)[0]
    for label, tid in _TAKE_LABEL_MAP.items():
        if stem.endswith(label):
            return tid
    return 0


def _run_fpcalc_raw(source_path: str, timeout_sec: int = 300) -> Optional[list[int]]:
    """ fpcalc -raw -length 120

    Args:
        source_path: file
        timeout_sec: s

    Returns:
        uint32 None
    """
    try:
        result = subprocess.run(
            ["fpcalc", "-raw", "-length", "120", source_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            logger.warning("fpcalc %d: %s", result.returncode, source_path)
            return None
        # FINGERPRINT=<comma-separated uint32>
        for line in result.stdout.strip().splitlines():
            if line.startswith("FINGERPRINT="):
                raw_str = line[len("FINGERPRINT="):]
                if not raw_str.strip():
                    return None
                return [int(x) for x in raw_str.split(",")]
        logger.warning("fpcalc FINGERPRINT row: %s", source_path)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("fpcalc : %s", source_path)
        return None
    except Exception as e:
        logger.warning("fpcalc : %s — %s", source_path, e)
        return None


def _fingerprint_sha256(fp_array: list[int]) -> str:
    """ SHA-256 audio_fingerprint_id

    uint32 SHA-256
    """
    raw = ",".join(str(v) for v in fp_array)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _hamming_similarity(fp_a: list[int], fp_b: list[int]) -> float:
    """ C

    0xFFFFFFFF ^ x ~x Python

    Args:
        fp_a: A uint32
        fp_b: B uint32

    Returns:
        [0.0, 1.0]
    """
    length = min(len(fp_a), len(fp_b))
    if length == 0:
        return 0.0
    # XOR popcount
    # Python 3.10+ int.bit_count() 3.9 fallback
    if sys.version_info >= (3, 10):
        diff_bits = sum(
            (fp_a[i] ^ fp_b[i]).bit_count()
            for i in range(length)
        )
    else:
        diff_bits = sum(
            bin(fp_a[i] ^ fp_b[i]).count("1")
            for i in range(length)
        )
    matching_bits = length * 32 - diff_bits
    return matching_bits / (length * 32)


def _sliding_containment(fp_long: list[int], fp_short: list[int]) -> float:
    """ C

    len(fp_long) >= len(fp_short)

    Args:
        fp_long:
        fp_short:

    Returns:
        [0.0, 1.0]
    """
    n_short = len(fp_short)
    n_long = len(fp_long)
    if n_short == 0:
        return 0.0
    if n_long < n_short:
        fp_long, fp_short = fp_short, fp_long
        n_long, n_short = n_short, n_long

    total_bits = n_short * 32
    use_builtin = sys.version_info >= (3, 10)

    max_matching = 0
    # j [0, n_long - n_short]
    for j in range(n_long - n_short + 1):
        if use_builtin:
            diff = sum(
                (fp_long[j + i] ^ fp_short[i]).bit_count()
                for i in range(n_short)
            )
        else:
            diff = sum(
                bin(fp_long[j + i] ^ fp_short[i]).count("1")
                for i in range(n_short)
            )
        matching = total_bits - diff
        if matching > max_matching:
            max_matching = matching
    return max_matching / total_bits


# Step 3: Auto QC

def step3_auto_qc(
    audit_rows: list[dict[str, Any]],
    freeze_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """ rows D03/D04/D05a/D06a + I10

    cleaning_rule_table.csv §7 + §8 + §13 Step 3 :
      1. D03 → D04 → D05a → D06a
      2. I10 → content_type

    applied_rule_ids

    Args:
        audit_rows: step2_metadata rows
        freeze_config:

    Returns:
        rows
    """
    logger.info("=" * 60)
    logger.info("Step 3: Auto QC ")
    logger.info("=" * 60)

    thresholds = freeze_config["thresholds"]
    d05a_default = float(thresholds["d05a_default"])
    d06a_default = float(thresholds["d06a_default"])

    # Stats
    stats: dict[str, int] = {
        "D03": 0, "D04": 0, "D05a": 0, "D06a": 0,
        "I10_flag": 0, "content_type_flag": 0,
    }

    for row in audit_rows:
        family_label = row["family_label"]
        source_dataset = row["source_dataset"]
        is_decodable = row["is_decodable"] is True

        # 3.1 per-family value §7.4
        d05a_th = _resolve_threshold(
            family_label, source_dataset, "d05a", d05a_default, freeze_config,
        )
        d06a_th = _resolve_threshold(
            family_label, source_dataset, "d06a", d06a_default, freeze_config,
        )

        # 3.2 —— drop
        first_drop_reason: Optional[str] = None

        # D03:
        if not is_decodable:
            _append_rule_id(row, "D03")
            stats["D03"] += 1
            if first_drop_reason is None:
                first_drop_reason = "decode_failure"

        # D04:
        if is_decodable and row["duration_sec"] < 3.0:
            _append_rule_id(row, "D04")
            stats["D04"] += 1
            if first_drop_reason is None:
                first_drop_reason = "too_short"

        # D05a:
        if is_decodable and row["silence_ratio"] > d05a_th:
            _append_rule_id(row, "D05a")
            stats["D05a"] += 1
            if first_drop_reason is None:
                first_drop_reason = "excessive_silence"

        # D06a:
        if is_decodable and row["clipping_ratio"] > d06a_th:
            _append_rule_id(row, "D06a")
            stats["D06a"] += 1
            if first_drop_reason is None:
                first_drop_reason = "severe_distortion"

        if first_drop_reason is not None and row["sample_status"] == "keep":
            row["sample_status"] = "drop"
            row["drop_reason"] = first_drop_reason

        # 3.3 —— sample_status=keep
        if row["sample_status"] != "keep":
            continue

        # I10 —— §7.7
        if row["bit_depth"] == -1:
            _append_rule_id(row, "I10")
            stats["I10_flag"] += 1
            row["manual_review_required"] = True
            row["manual_review_status"] = "pending"
            _add_pending_review(
                row, "i10_encoding", "I10",
                "bit_depth=-1,unrecognized_subtype",
            )

        # content_type —— External Step 2 phrase
        if source_dataset in _PRIMARY_DATASETS and row["content_type"] == "":
            stats["content_type_flag"] += 1
            row["manual_review_required"] = True
            row["manual_review_status"] = "pending"
            _add_pending_review(
                row, "content_type", "I01",
                "content_type_missing",
            )

    # 3.5 Stats
    total_drop = sum(1 for r in audit_rows if r["sample_status"] == "drop")
    total_pending = sum(1 for r in audit_rows if r.get("manual_review_required") is True
                        and r.get("manual_review_status") == "pending")
    logger.info("Step 3 :")
    logger.info(" : D03=%d, D04=%d, D05a=%d, D06a=%d",
                stats["D03"], stats["D04"], stats["D05a"], stats["D06a"])
    logger.info(" : I10=%d, content_type=%d",
                stats["I10_flag"], stats["content_type_flag"])
    logger.info(" drop=%d, pending=%d, keep=%d",
                total_drop, total_pending,
                len(audit_rows) - total_drop)

    return audit_rows


# Step 4:

def step4_apply_manual_review(
    audit_rows: list[dict[str, Any]],
    manual_review: list[dict[str, str]],
    freeze_config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """ manual_review.csv keep/drop

    §13 Step 4 + §5
      1. pending
      2. manual_review.csv
      3. → pending_review_queue.csv
      4. approved/rejected → sample_status/drop_reason/

    d09_cross / g02_fingerprint / manual_merge
    Step 5

    Args:
        audit_rows: step3_auto_qc rows
        manual_review: manual_review.csv row
        freeze_config: Freeze config
        output_dir: Output dir

    Returns:
        rows
    """
    logger.info("=" * 60)
    logger.info("Step 4: ")
    logger.info("=" * 60)

    # 4.1 manual_review
    # key = (record_id, review_type)
    mr_single: dict[tuple[str, str], dict[str, str]] = {}
    # key = (min(rid, peer), review_type, max(rid, peer))
    mr_pair: dict[tuple[str, str, str], dict[str, str]] = {}

    for mr_row in manual_review:
        rt = mr_row["review_type"].strip()
        rid = mr_row["record_id"].strip()
        peer = mr_row.get("peer_record_id", "").strip()
        if rt in _PAIR_REVIEW_TYPES:
            mr_pair[_normalize_pair_key(rid, rt, peer)] = mr_row
        else:
            mr_single[(rid, rt)] = mr_row

    # 4.2 pending
    pending_items: list[dict[str, Any]] = []
    # record_id → rows pending review
    pending_by_record: dict[str, list[dict[str, str]]] = {}

    for row in audit_rows:
        reviews = row.pop("_pending_reviews", [])
        if not reviews:
            continue
        record_id = row["record_id"]
        pending_by_record[record_id] = reviews
        for pr in reviews:
            pending_items.append({
                "record_id": record_id,
                "review_type": pr["review_type"],
                "source_dataset": row["source_dataset"],
                "family_label": row["family_label"],
                "file_name": row["file_name"],
                "source_path": row["source_path"],
                "trigger_rule_id": pr["trigger_rule_id"],
                "trigger_detail": pr["trigger_detail"],
                "peer_record_id": pr.get("peer_record_id", ""),
            })

    logger.info(" : %d %d ",
                len(pending_items), len(pending_by_record))

    if not pending_items:
        logger.info(" Skip Step 4")
        return audit_rows

    # 4.3 pending
    # d09_cross / g02_fingerprint / manual_merge
    # Step 5
    unmatched: list[dict[str, Any]] = []
    _STEP5_REVIEW_TYPES = {"d09_cross", "g02_fingerprint", "manual_merge"}

    for item in pending_items:
        rt = item["review_type"]
        rid = item["record_id"]
        peer = item.get("peer_record_id", "")
        # Step 5 Skip
        if rt in _STEP5_REVIEW_TYPES:
            continue
        key = (rid, rt)
        if key not in mr_single:
            unmatched.append(item)

    if unmatched:
        logger.warning(" %d ", len(unmatched))
        _write_pending_review_queue(unmatched, output_dir)

    # 4.4
    # record_id → row
    row_by_id: dict[str, dict[str, Any]] = {r["record_id"]: r for r in audit_rows}

    applied_count = 0
    for record_id, reviews in pending_by_record.items():
        row = row_by_id[record_id]
        all_approved = True

        for pr in reviews:
            rt = pr["review_type"]
            # Step 5 Skip
            if rt in _STEP5_REVIEW_TYPES:
                continue

            key = (record_id, rt)
            mr_entry = mr_single.get(key)
            if mr_entry is None:
                # unmatched
                continue

            decision = mr_entry["decision"].strip()
            applied_count += 1

            if decision == "approved":
                # review_type
                if rt == "content_type":
                    override_val = mr_entry.get("content_type_override", "").strip()
                    if override_val:
                        row["content_type"] = override_val
                elif rt == "i10_encoding":
                    bd_override = mr_entry.get("bit_depth_override", "").strip()
                    if bd_override:
                        row["bit_depth"] = int(bd_override)
            elif decision == "rejected":
                all_approved = False
                row["sample_status"] = "drop"
                # drop_reason
                if rt == "i10_encoding":
                    row["drop_reason"] = "unrecognized_encoding"
                elif rt == "content_type":
                    row["drop_reason"] = "label_requires_guess"

        # manual_review_status Step 5
        non_step5_reviews = [pr for pr in reviews if pr["review_type"] not in _STEP5_REVIEW_TYPES]
        if non_step5_reviews:
            if row["sample_status"] == "drop":
                row["manual_review_status"] = "rejected"
            elif all_approved:
                # Step 5 review
                step5_reviews = [pr for pr in reviews if pr["review_type"] in _STEP5_REVIEW_TYPES]
                if step5_reviews:
                    # pending
                    pass
                else:
                    row["manual_review_status"] = "approved"

    logger.info("Step 4 : %d ", applied_count)

    # Stats
    total_drop = sum(1 for r in audit_rows if r["sample_status"] == "drop")
    total_keep = sum(1 for r in audit_rows if r["sample_status"] == "keep")
    logger.info(" : keep=%d, drop=%d", total_keep, total_drop)

    return audit_rows


# Step 5: + Dedup + Group

def step5_dedup_group(
    audit_rows: list[dict[str, Any]],
    freeze_config: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """ rows Stage A + D09 + Stage B + D10

    §13 Step 5 + §9
      Stage A → → take_id → near-duplicate
      D09 SHA-256 —— /
      Stage B fpcalc → → →
      D10 unique_recordings_count >= 35 canonical
      recording_group_id

    Args:
        audit_rows: step4 rows
        freeze_config:
        config: config.yaml
        output_dir: Output dir

    Returns:
        rows
    """
    from tqdm import tqdm

    logger.info("=" * 60)
    logger.info("Step 5: + ")
    logger.info("=" * 60)

    family_abbr: dict[str, str] = freeze_config["family_abbr"]

    # manual_review d09_cross / g02_fingerprint / manual_merge
    manual_review: list[dict[str, str]] = config.get("_manual_review", [])
    mr_pair: dict[tuple[str, str, str], dict[str, str]] = {}
    mr_single: dict[tuple[str, str], dict[str, str]] = {}
    for mr_row in manual_review:
        rt = mr_row["review_type"].strip()
        rid = mr_row["record_id"].strip()
        peer = mr_row.get("peer_record_id", "").strip()
        if rt in _PAIR_REVIEW_TYPES:
            mr_pair[_normalize_pair_key(rid, rt, peer)] = mr_row
        else:
            mr_single[(rid, rt)] = mr_row

    # record_id → row
    row_by_id: dict[str, dict[str, Any]] = {r["record_id"]: r for r in audit_rows}

    # / ChMusic row keep + decodable
    primary_rows = [r for r in audit_rows if r["source_dataset"] in _PRIMARY_DATASETS]
    chm_rows = [r for r in audit_rows if r["source_dataset"] == "ChMusic"]
    primary_keep = [r for r in primary_rows if r["sample_status"] == "keep" and r["is_decodable"] is True]
    chm_keep = [r for r in chm_rows if r["sample_status"] == "keep" and r["is_decodable"] is True]

    # Stage A —— §9.1
    logger.info("Stage A: ")

    # A2: auto group —— (source_subtype_dir, normalized_base_name)
    # group_key → [row, ...]
    auto_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in primary_rows:
        key = (row["source_subtype_dir"], row["normalized_base_name"])
        auto_groups.setdefault(key, []).append(row)
    # (source_subtype_dir, family_label) normalized_base_name
    # (source_subtype_dir, family_label, de_ws_key) → [group_key, ...]
    ws_merge_map: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
    for gk in auto_groups:
        subtype_dir, norm_name = gk
        # family_label subtype_dir
        members = auto_groups[gk]
        if not members:
            continue
        fl = members[0]["family_label"]
        de_ws = re.sub(r"[\s\u3000]+", "", norm_name)
        merge_key = (subtype_dir, fl, de_ws)
        ws_merge_map.setdefault(merge_key, []).append(gk)

    # rows group_key group_key
    merged_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    # group_key grouping_method
    group_method: dict[tuple[str, str], str] = {}

    for merge_key, gk_list in ws_merge_map.items():
        if len(gk_list) == 1:
            gk = gk_list[0]
            merged_groups[gk] = auto_groups[gk]
            group_method[gk] = "filename_rule"
        else:
            # group normalized_base_name
            gk_list_sorted = sorted(gk_list, key=lambda g: g[1])
            canonical_gk = gk_list_sorted[0]
            all_members: list[dict[str, Any]] = []
            for gk in gk_list_sorted:
                all_members.extend(auto_groups[gk])
            merged_groups[canonical_gk] = all_members
            group_method[canonical_gk] = "whitespace_variant_merge"
            logger.info(" : %s → %d ", gk_list_sorted, len(all_members))

    # A3: take_id file
    for gk, members in merged_groups.items():
        for row in members:
            row["take_id"] = _parse_take_id(row["file_name"])

    # recording_group_id §9.2 CCMusic
    # (source_subtype_dir ASC, normalized_base_name ASC)
    sorted_gk_list = sorted(merged_groups.keys(), key=lambda g: (g[0], g[1]))

    # family_label seq family
    family_gk_groups: dict[str, list[tuple[str, str]]] = {}
    for gk in sorted_gk_list:
        members = merged_groups[gk]
        if not members:
            continue
        fl = members[0]["family_label"]
        family_gk_groups.setdefault(fl, []).append(gk)

    for fl, gk_list in family_gk_groups.items():
        abbr = family_abbr.get(fl, fl)
        for seq, gk in enumerate(gk_list, start=1):
            rgid = f"ccm_{abbr}_G{seq:04d}"
            method = group_method[gk]
            members = merged_groups[gk]
            for row in members:
                row["recording_group_id"] = rgid
                row["grouping_method"] = method
                row["performer_or_recording_id"] = rgid

    # A4: near-duplicate recording_group_id take →
    rgid_members: dict[str, list[dict[str, Any]]] = {}
    for row in primary_rows:
        rgid = row.get("recording_group_id", "")
        if rgid:
            rgid_members.setdefault(rgid, []).append(row)

    for rgid, members in rgid_members.items():
        # → near-duplicate
        if len(members) > 1:
            for m in members:
                m["is_near_duplicate"] = True

    # ChMusic §9.2 unique_singleton
    logger.info("ChMusic : ")

    # family_label file_name
    chm_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in chm_rows:
        fl = row["family_label"]
        chm_by_family.setdefault(fl, []).append(row)

    for fl, row in chm_by_family.items():
        row_sorted = sorted(rows, key=lambda r: chmusic_sort_key(r["file_name"]))
        abbr = family_abbr.get(fl, fl)
        for seq, row in enumerate(rows_sorted, start=1):
            rgid = f"chm_{abbr}_G{seq:04d}"
            row["recording_group_id"] = rgid
            row["grouping_method"] = "unique_singleton"
            row["take_id"] = 0
            row["performer_or_recording_id"] = rgid

    # D09: §9.3
    logger.info("D09: ")

    # audio_sha256 keep + decodable + sha256
    sha_groups: dict[str, list[dict[str, Any]]] = {}
    for row in audit_rows:
        if row["sample_status"] != "keep":
            continue
        if row["is_decodable"] is not True:
            continue
        sha = row.get("audio_sha256", "")
        if not sha:
            continue
        sha_groups.setdefault(sha, []).append(row)

    d09_auto_count = 0
    d09_cross_pending: list[dict[str, Any]] = []

    for sha, members in sha_groups.items():
        if len(members) < 2:
            continue

        # family_label source_dataset
        families = set(m["family_label"] for m in members)
        datasets = set(m["source_dataset"] for m in members)
        is_cross = len(families) > 1 or len(datasets) > 1

        if not is_cross:
            # family_label source_dataset →
            # canonical = (source_dataset ASC, source_subtype_dir ASC, file_name ASC)
            members_sorted = sorted(
                members,
                key=lambda m: (m["source_dataset"], m["source_subtype_dir"], m["file_name"]),
            )
            canonical = members_sorted[0]
            for dup in members_sorted[1:]:
                _append_rule_id(dup, "D09")
                dup["sample_status"] = "drop"
                dup["drop_reason"] = "exact_duplicate"
                dup["duplicate_of"] = canonical["record_id"]
                d09_auto_count += 1
        else:
            # → d09_cross
            # pair record_id (min, max)
            members_sorted = sorted(members, key=lambda m: m["record_id"])
            generated_pairs: set[tuple[str, str]] = set()
            for i in range(len(members_sorted)):
                for j in range(i + 1, len(members_sorted)):
                    a = members_sorted[i]
                    b = members_sorted[j]
                    pair_key = (a["record_id"], b["record_id"])
                    if pair_key in generated_pairs:
                        continue
                    generated_pairs.add(pair_key)
                    _append_rule_id(a, "D09")
                    _append_rule_id(b, "D09")
                    d09_cross_pending.append({
                        "record_id": pair_key[0],
                        "review_type": "d09_cross",
                        "source_dataset": a["source_dataset"],
                        "family_label": a["family_label"],
                        "file_name": a["file_name"],
                        "source_path": a["source_path"],
                        "trigger_rule_id": "D09",
                        "trigger_detail": f"sha256={sha[:16]}...,cross_family={len(families) > 1},cross_dataset={len(datasets) > 1}",
                        "peer_record_id": pair_key[1],
                    })

    logger.info(" D09 : %d ", d09_auto_count)
    logger.info(" D09 : %d ", len(d09_cross_pending))

    # d09_cross
    if d09_cross_pending:
        d09_unmatched: list[dict[str, Any]] = []
        for item in d09_cross_pending:
            a, b = item["record_id"], item["peer_record_id"]
            key = _normalize_pair_key(a, "d09_cross", b)
            if key not in mr_pair:
                d09_unmatched.append(item)
        if d09_unmatched:
            logger.warning("D09 %d ", len(d09_unmatched))
            _write_pending_review_queue(d09_unmatched, output_dir)

        # d09_cross
        _apply_d09_cross_decisions(d09_cross_pending, mr_pair, row_by_id)

    # Stage B: —— §9.5
    logger.info("Stage B: ")

    # keep + decodable rows fpcalc
    primary_for_fp = [
        r for r in audit_rows
        if r["source_dataset"] in _PRIMARY_DATASETS
        and r["sample_status"] == "keep"
        and r["is_decodable"] is True
    ]

    # fpcalc record_id → fp_array
    fp_cache: dict[str, list[int]] = {}
    if primary_for_fp:
        logger.info(" rows fpcalc: %d files", len(primary_for_fp))
        for row in tqdm(primary_for_fp, desc="fpcalc ", unit="file"):
            fp = _run_fpcalc_raw(row["source_path"])
            if fp is not None:
                fp_cache[row["record_id"]] = fp
                row["audio_fingerprint_id"] = _fingerprint_sha256(fp)
            else:
                logger.warning(" fpcalc : %s", row["file_name"])

    # family_label source_subtype_dir
    fp_pending: list[dict[str, Any]] = []
    _find_fingerprint_candidates(primary_for_fp, fp_cache, fp_pending)

    logger.info(" : %d ", len(fp_pending))

    # g02_fingerprint
    if fp_pending:
        fp_unmatched: list[dict[str, Any]] = []
        for item in fp_pending:
            a, b = item["record_id"], item["peer_record_id"]
            key = _normalize_pair_key(a, "g02_fingerprint", b)
            if key not in mr_pair:
                fp_unmatched.append(item)
        if fp_unmatched:
            logger.warning("G02 %d ", len(fp_unmatched))
            _write_pending_review_queue(fp_unmatched, output_dir)

        # g02_fingerprint
        _apply_fingerprint_decisions(fp_pending, mr_pair, row_by_id)

    # manual_merge: §5.3 + §9.5
    logger.info("manual_merge: ")

    # manual_merge
    mm_entries = [
        mr_row for mr_row in manual_review
        if mr_row.get("review_type", "").strip() == "manual_merge"
    ]

    if mm_entries:
        mm_unmatched: list[dict[str, Any]] = []
        mm_applied = 0

        for mr_row in mm_entries:
            rid = mr_row["record_id"].strip()
            peer = mr_row.get("peer_record_id", "").strip()
            decision = mr_row["decision"].strip()

            row_a = row_by_id.get(rid)
            row_b = row_by_id.get(peer)

            if row_a is None:
                logger.error(
                    "manual_merge not found record_id: %s ", rid,
                )
                sys.exit(1)
            if row_b is None:
                logger.error(
                    "manual_merge not found peer_record_id: %s ", peer,
                )
                sys.exit(1)

            # drop
            if row_a["sample_status"] == "drop":
                logger.error(
                    "manual_merge drop record_id: %s ", rid,
                )
                sys.exit(1)
            if row_b["sample_status"] == "drop":
                logger.error(
                    "manual_merge drop peer_record_id: %s ", peer,
                )
                sys.exit(1)

            if row_a["source_dataset"] != row_b["source_dataset"]:
                logger.error(
                    "manual_merge : %s (%s) ↔ %s (%s) ",
                    rid, row_a["source_dataset"], peer, row_b["source_dataset"],
                )
                sys.exit(1)

            # family_label recording_group_id family §E
            if row_a["family_label"] != row_b["family_label"]:
                logger.error(
                    "manual_merge family_label : %s (%s) ↔ %s (%s) ",
                    rid, row_a["family_label"], peer, row_b["family_label"],
                )
                sys.exit(1)

            if decision == "approved":
                # recording_group_id peer → rid
                target_rgid = row_a["recording_group_id"]
                old_rgid = row_b["recording_group_id"]
                if target_rgid and old_rgid and target_rgid != old_rgid:
                    for row in row_by_id.values():
                        if row["recording_group_id"] == old_rgid:
                            row["recording_group_id"] = target_rgid
                            row["performer_or_recording_id"] = target_rgid
                            row["grouping_method"] = "manual_merge"
                    logger.info(" manual_merge : %s → %s", old_rgid, target_rgid)
                mm_applied += 1
            # rejected →

            # manual_review_status
            for involved_row in (row_a, row_b):
                involved_row["manual_review_required"] = True
                if decision == "approved":
                    involved_row["manual_review_status"] = "approved"
                else:
                    # rejected manual_merge " " keep/drop
                    # rejected §16.2
                    involved_row["manual_review_status"] = "rejected"

        logger.info(" manual_merge %d ", mm_applied)
    else:
        logger.info(" manual_merge ")

    # d09_cross + g02_fingerprint
    # manual_review_status pending
    # d09_cross: sample_status=keep → approved, =drop → rejected
    # g02_fingerprint: approved/rejected approved keep/drop -> got
    _writeback_pair_review_status(d09_cross_pending, mr_pair, row_by_id, "d09_cross")
    _writeback_pair_review_status(fp_pending, mr_pair, row_by_id, "g02_fingerprint")

    # D10: CCMusic —— §9.4
    logger.info("D10: ")

    _apply_d10_near_duplicate(audit_rows, family_abbr)

    # recording_group_id
    logger.info("recording_group_id ")

    # audit_manifest
    audit_path = output_dir / "audit_manifest.csv"
    for row in audit_rows:
        row.pop("_pending_reviews", None)
    _write_manifest(audit_rows, AUDIT_MANIFEST_COLUMNS, audit_path)

    # Stats
    total_keep = sum(1 for r in audit_rows if r["sample_status"] == "keep")
    total_drop = sum(1 for r in audit_rows if r["sample_status"] == "drop")
    logger.info("Step 5 : keep=%d, drop=%d", total_keep, total_drop)

    return audit_rows


def _apply_d09_cross_decisions(
    pending_items: list[dict[str, Any]],
    mr_pair: dict[tuple[str, str, str], dict[str, str]],
    row_by_id: dict[str, dict[str, Any]],
) -> None:
    """ d09_cross §9.3 ——

    audio_sha256 pair
    - 1 survivor
    - drop duplicate_of canonical
    - canonical
    """
    # 1. pending_items SHA256 → record_id
    # sha pair trigger_detail sha256
    # row_by_id audio_sha256
    cluster_by_sha: dict[str, set[str]] = {}
    for item in pending_items:
        rid = item["record_id"]
        peer = item["peer_record_id"]
        # row sha record sha
        row_a = row_by_id.get(rid)
        if row_a is None:
            continue
        sha = row_a.get("audio_sha256", "")
        if not sha:
            continue
        cluster_by_sha.setdefault(sha, set()).add(rid)
        cluster_by_sha.setdefault(sha, set()).add(peer)

    # 2. pair canonical
    d09_applied = 0
    for sha, member_ids in cluster_by_sha.items():
        # pair
        canonical_votes: set[str] = set()
        rejected_ids: set[str] = set()

        # pair
        sorted_ids = sorted(member_ids)
        for i in range(len(sorted_ids)):
            for j in range(i + 1, len(sorted_ids)):
                a, b = sorted_ids[i], sorted_ids[j]
                key = _normalize_pair_key(a, "d09_cross", b)
                mr_entry = mr_pair.get(key)
                if mr_entry is None:
                    continue
                decision = mr_entry["decision"].strip()
                if decision == "approved":
                    canonical_id = mr_entry.get("canonical_record_id", "").strip()
                    if canonical_id:
                        canonical_votes.add(canonical_id)
                elif decision == "rejected":
                    # rejected " record_id "
                    # rejected row record_id
                    orig_rid = mr_entry["record_id"].strip()
                    rejected_ids.add(orig_rid)

        # 3.
        if len(canonical_votes) > 1:
            logger.error(
                "D09 sha256=%s... : "
                " pair canonical: %s ",
                sha[:16], canonical_votes,
            )
            sys.exit(1)

        # 3b. canonical rejected
        # record_id approved canonical rejected
        conflict = canonical_votes & rejected_ids
        if conflict:
            logger.error(
                "D09 sha256=%s... : "
                "record_id %s canonical rejected "
                " ",
                sha[:16], conflict,
            )
            sys.exit(1)

        # canonical
        final_canonical: Optional[str] = None
        if canonical_votes:
            final_canonical = canonical_votes.pop()
        elif rejected_ids:
            # rejected survivor reject
            survivors = member_ids - rejected_ids
            if len(survivors) == 1:
                final_canonical = survivors.pop()
            elif len(survivors) == 0:
                logger.error(
                    "D09 sha256=%s... : "
                    " %d rejected survivor ",
                    sha[:16], len(member_ids),
                )
                sys.exit(1)
            else:
                logger.error(
                    "D09 sha256=%s... : "
                    "rejected %d survivor (%s) canonical "
                    " ",
                    sha[:16], len(survivors), survivors,
                )
                sys.exit(1)
        else:
            # ——unmatched
            continue

        # 4. rows canonical drop
        for mid in member_ids:
            row = row_by_id.get(mid)
            if row is None:
                continue
            if mid == final_canonical:
                continue
            if row["sample_status"] == "keep":
                row["sample_status"] = "drop"
                row["drop_reason"] = "exact_duplicate"
                row["duplicate_of"] = final_canonical
                _append_rule_id(row, "D09")
                d09_applied += 1

    logger.info(" D09 : %d ", d09_applied)


def _find_fingerprint_candidates(
    primary_rows: list[dict[str, Any]],
    fp_cache: dict[str, list[int]],
    fp_pending: list[dict[str, Any]],
) -> None:
    """ family_label source_subtype_dir §9.5

    value >= 0.85 >= 0.90

    Args:
        primary_rows: keep+decodable row
        fp_cache: record_id →
        fp_pending:
    """
    # (family_label, source_subtype_dir)
    by_family_subtype: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in primary_rows:
        fl = row["family_label"]
        sd = row["source_subtype_dir"]
        if row["record_id"] not in fp_cache:
            continue
        by_family_subtype.setdefault(fl, {}).setdefault(sd, []).append(row)

    generated_pairs: set[tuple[str, str]] = set()

    for fl, subtype_dict in by_family_subtype.items():
        subtypes = sorted(subtype_dict.keys())
        if len(subtypes) < 2:
            continue
        # source_subtype_dir
        for si in range(len(subtypes)):
            for sj in range(si + 1, len(subtypes)):
                row_i = subtype_dict[subtypes[si]]
                row_j = subtype_dict[subtypes[sj]]
                for ri in row_i:
                    fp_a = fp_cache.get(ri["record_id"])
                    if fp_a is None:
                        continue
                    for rj in row_j:
                        fp_b = fp_cache.get(rj["record_id"])
                        if fp_b is None:
                            continue
                        # pair key
                        pair = tuple(sorted([ri["record_id"], rj["record_id"]]))
                        if pair in generated_pairs:
                            continue
                        # Skip 0.90
                        len_a, len_b = len(fp_a), len(fp_b)
                        if len_a > 0 and len_b > 0:
                            ratio = min(len_a, len_b) / max(len_a, len_b)
                            if ratio < 0.50:
                                continue
                        sim = _hamming_similarity(fp_a, fp_b)
                        hit = sim >= 0.85
                        if not hit:
                            if len(fp_a) >= len(fp_b):
                                cont = _sliding_containment(fp_a, fp_b)
                            else:
                                cont = _sliding_containment(fp_b, fp_a)
                            hit = cont >= 0.90
                        if hit:
                            generated_pairs.add(pair)
                            _append_rule_id(ri, "G02")
                            _append_rule_id(rj, "G02")
                            fp_pending.append({
                                "record_id": pair[0],
                                "review_type": "g02_fingerprint",
                                "source_dataset": ri["source_dataset"] if ri["record_id"] == pair[0] else rj["source_dataset"],
                                "family_label": fl,
                                "file_name": ri["file_name"] if ri["record_id"] == pair[0] else rj["file_name"],
                                "source_path": ri["source_path"] if ri["record_id"] == pair[0] else rj["source_path"],
                                "trigger_rule_id": "G02",
                                "trigger_detail": f"hamming_sim={sim:.4f}",
                                "peer_record_id": pair[1],
                            })


def _apply_fingerprint_decisions(
    pending_items: list[dict[str, Any]],
    mr_pair: dict[tuple[str, str, str], dict[str, str]],
    row_by_id: dict[str, dict[str, Any]],
) -> None:
    """ g02_fingerprint §9.5

    approved → recording_group_id peer group record_id group
    rejected → sample_status
    """
    for item in pending_items:
        rid = item["record_id"]
        peer = item["peer_record_id"]
        key = _normalize_pair_key(rid, "g02_fingerprint", peer)
        mr_entry = mr_pair.get(key)
        if mr_entry is None:
            continue

        decision = mr_entry["decision"].strip()
        if decision == "approved":
            # recording_group_id peer → rid
            row_a = row_by_id.get(rid)
            row_b = row_by_id.get(peer)
            if row_a is None or row_b is None:
                continue
            target_rgid = row_a["recording_group_id"]
            old_rgid = row_b["recording_group_id"]
            if target_rgid and old_rgid and target_rgid != old_rgid:
                # group group
                for row in row_by_id.values():
                    if row["recording_group_id"] == old_rgid:
                        row["recording_group_id"] = target_rgid
                        row["performer_or_recording_id"] = target_rgid
                        row["grouping_method"] = "fingerprint_merge"
                logger.info(" : %s → %s", old_rgid, target_rgid)
        # rejected →


def _writeback_pair_review_status(
    pending_items: list[dict[str, Any]],
    mr_pair: dict[tuple[str, str, str], dict[str, str]],
    row_by_id: dict[str, dict[str, Any]],
    review_type: str,
) -> None:
    """ rows §16.2

    d09_cross: sample_status=keep → approved, =drop → rejected
    g02_fingerprint/manual_merge: got decision ——
        approved → manual_review_status=approved
        rejected → manual_review_status=rejected
    record_id rejected rejected
    """
    # record_id decision
    id_decisions: dict[str, set[str]] = {}
    for item in pending_items:
        rid = item["record_id"]
        peer = item["peer_record_id"]
        key = _normalize_pair_key(rid, review_type, peer)
        mr_entry = mr_pair.get(key)
        if mr_entry is None:
            continue
        decision = mr_entry["decision"].strip()
        id_decisions.setdefault(rid, set()).add(decision)
        id_decisions.setdefault(peer, set()).add(decision)

    for mid, decisions in id_decisions.items():
        row = row_by_id.get(mid)
        if row is None:
            continue
        row["manual_review_required"] = True
        if review_type == "d09_cross":
            # d09_cross keep/drop
            if row["sample_status"] == "drop":
                row["manual_review_status"] = "rejected"
            else:
                row["manual_review_status"] = "approved"
        else:
            # g02_fingerprint / manual_merge:
            # record_id rejected rejected
            if "rejected" in decisions:
                row["manual_review_status"] = "rejected"
            else:
                row["manual_review_status"] = "approved"


def _apply_d10_near_duplicate(
    audit_rows: list[dict[str, Any]],
    family_abbr: dict[str, str],
) -> None:
    """ D10 §9.4

    family_label unique_recordings_count >= 35 canonical
    take

    canonical (source_subtype_dir ASC, file_name ASC)
    """
    # family_label Stats keep unique recording group
    primary_keep = [
        r for r in audit_rows
        if r["source_dataset"] in _PRIMARY_DATASETS
        and r["sample_status"] == "keep"
    ]

    # family_label → set(recording_group_id)
    family_unique_groups: dict[str, set[str]] = {}
    for row in primary_keep:
        fl = row["family_label"]
        rgid = row.get("recording_group_id", "")
        if rgid:
            family_unique_groups.setdefault(fl, set()).add(rgid)

    d10_count = 0
    for fl, group_ids in family_unique_groups.items():
        unique_count = len(group_ids)
        if unique_count < 35:
            logger.info(" D10 %s: unique_recordings=%d < 35 → take", fl, unique_count)
            continue

        logger.info(" D10 %s: unique_recordings=%d >= 35 → canonical", fl, unique_count)
        # recording_group_id
        group_members: dict[str, list[dict[str, Any]]] = {}
        for row in primary_keep:
            if row["family_label"] != fl:
                continue
            rgid = row.get("recording_group_id", "")
            if rgid:
                group_members.setdefault(rgid, []).append(row)

        for rgid, members in group_members.items():
            if len(members) <= 1:
                continue
            # canonical = (source_subtype_dir ASC, file_name ASC)
            members_sorted = sorted(
                members,
                key=lambda m: (m["source_subtype_dir"], m["file_name"]),
            )
            canonical = members_sorted[0]
            for dup in members_sorted[1:]:
                if dup["sample_status"] == "keep":
                    _append_rule_id(dup, "D10")
                    dup["sample_status"] = "drop"
                    dup["drop_reason"] = "redundant_near_duplicate"
                    dup["duplicate_of"] = canonical["record_id"]
                    d10_count += 1

    logger.info(" D10 : %d ", d10_count)


# Step 6: Quota

def step6_quota(
    audit_rows: list[dict[str, Any]],
    freeze_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """ rows phrase B01 + R01/R02/R03 + B02/B03

    §13 Step 6 + §10
      1. B01 phrase CCMusic
      2. R01 / R02 / R03 Stats CCMusic
      3. B02 / B03

    Args:
        audit_rows: step5 rows
        freeze_config:

    Returns:
        rows
    """
    logger.info("=" * 60)
    logger.info("Step 6: Quota ")
    logger.info("=" * 60)

    thresholds = freeze_config["thresholds"]
    phrase_ratio_cap: float = float(thresholds.get("phrase_ratio_cap", 0.15))
    split_seeds: list[int] = freeze_config["split_seeds"]
    seed_ns: dict[str, int] = freeze_config["seed_namespaces"]
    active_classes: list[str] = freeze_config["active_classes"]

    # 6.1 B01: phrase kept —— §10.1
    # External phrase B01
    # phrase_ratio CCMusic phrase primary kept External
    logger.info("B01: phrase phrase_ratio_cap=%.2f ", phrase_ratio_cap)

    # family_label Stats kept External
    primary_keep_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in audit_rows:
        if row["source_dataset"] not in _PRIMARY_DATASETS:
            continue
        if row["sample_status"] != "keep":
            continue
        fl = row["family_label"]
        primary_keep_by_family.setdefault(fl, []).append(row)

    b01_total_dropped = 0
    for fl in active_classes:
        members = primary_keep_by_family.get(fl, [])
        if not members:
            continue
        total_keep_count = len(members)
        # phrase ——External B01 phrase
        phrase_candidates = [
            r for r in members
            if r["content_type"] == "phrase" and r["source_dataset"] != "External"
        ]
        phrase_count = len(phrase_candidates)

        if phrase_count == 0:
            logger.info(" B01 %s: phrase=0/%d ", fl, total_keep_count)
            continue

        phrase_ratio = phrase_count / total_keep_count
        if phrase_ratio <= phrase_ratio_cap:
            logger.info(
                " B01 %s: phrase=%d/%d (%.2f%%) <= %.2f%% ",
                fl, phrase_count, total_keep_count,
                phrase_ratio * 100, phrase_ratio_cap * 100,
            )
            continue

        # k phrase (phrase-k)/(total-k) <= cap
        # k >= (phrase - cap*total) / (1 - cap)
        k = max(0, math.ceil(
            (phrase_count - phrase_ratio_cap * total_keep_count)
            / (1.0 - phrase_ratio_cap)
        ))
        if k <= 0:
            continue
        # phrase
        k = min(k, phrase_count)

        logger.info(
            " B01 %s: phrase=%d/%d (%.2f%%) > %.2f%% %d ",
            fl, phrase_count, total_keep_count,
            phrase_ratio * 100, phrase_ratio_cap * 100, k,
        )

        # (source_subtype_dir, normalized_base_name, file_name)
        phrase_candidates.sort(
            key=lambda r: (
                r["source_subtype_dir"],
                r["normalized_base_name"],
                r["file_name"],
            )
        )

        # rng.sample
        candidate_indices = list(range(len(phrase_candidates)))

        # split_seeds[0] + seed_namespaces["phrase_downsampling"]
        base_seed = split_seeds[0]
        rng = random.Random(base_seed + seed_ns["phrase_downsampling"])

        drop_indices = set(rng.sample(candidate_indices, k))

        for idx in drop_indices:
            row = phrase_candidates[idx]
            row["sample_status"] = "drop"
            row["drop_reason"] = "phrase_quota_downsampled"
            _append_rule_id(row, "B01")
            b01_total_dropped += 1

        # phrase ratio <= cap
        remaining_phrase = phrase_count - k
        remaining_total = total_keep_count - k
        if remaining_total > 0 and remaining_phrase / remaining_total > phrase_ratio_cap:
            logger.error(
                "B01 : %s %d phrase=%d/%d (%.4f) cap=%.4f "
                " ",
                fl, k, remaining_phrase, remaining_total,
                remaining_phrase / remaining_total, phrase_ratio_cap,
            )
            sys.exit(1)

    logger.info(" B01 %d phrase ", b01_total_dropped)

    # 6.2 R01/R02/R03 kept —— §10.3
    logger.info("R01/R02/R03 ")

    # Stats kept B01 sample_status
    primary_keep_stats: dict[str, dict[str, Any]] = {}
    for fl in active_classes:
        primary_keep_stats[fl] = {"kept_recordings": 0, "unique_groups": set()}

    for row in audit_rows:
        if row["source_dataset"] not in _PRIMARY_DATASETS:
            continue
        if row["sample_status"] != "keep":
            continue
        fl = row["family_label"]
        if fl not in primary_keep_stats:
            continue
        primary_keep_stats[fl]["kept_recordings"] += 1
        rgid = row.get("recording_group_id", "")
        if rgid:
            primary_keep_stats[fl]["unique_groups"].add(rgid)

    # R01: kept_recordings < 25 →
    for fl, stats in primary_keep_stats.items():
        kept = stats["kept_recordings"]
        if kept < 25:
            logger.error(
                "R01 : %s kept_recordings=%d < 25 "
                " freeze_config.yaml / rows ",
                fl, kept,
            )
            sys.exit(1)

    # R02: kept_recordings < 0.40 × median →
    kept_counts = [s["kept_recordings"] for s in primary_keep_stats.values()]
    sorted_counts = sorted(kept_counts)
    n = len(sorted_counts)
    if n % 2 == 1:
        median_kept = float(sorted_counts[n // 2])
    else:
        median_kept = (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2.0
    r02_threshold = 0.40 * median_kept

    for fl, stats in primary_keep_stats.items():
        kept = stats["kept_recordings"]
        if kept < r02_threshold:
            logger.error(
                "R02 : %s kept_recordings=%d < 0.40 × median(%.1f)=%.1f "
                " freeze_config.yaml rows ",
                fl, kept, median_kept, r02_threshold,
            )
            sys.exit(1)

    # R03: unique_groups < 7 → unique_groups >= 3 split 1
    for fl, stats in primary_keep_stats.items():
        n_groups = len(stats["unique_groups"])
        if n_groups < 3:
            logger.error(
                "R03 : %s unique_groups=%d < 3 "
                " train/val/test 1 "
                " freeze_config.yaml rows ",
                fl, n_groups,
            )
            sys.exit(1)
        if n_groups < 7:
            logger.error(
                "R03 : %s unique_groups=%d < 7 "
                " freeze_config.yaml rows ",
                fl, n_groups,
            )
            sys.exit(1)

    logger.info(" R01/R02/R03 median_kept=%.1f ", median_kept)

    # 6.3 B02/B03 —— §10.2
    logger.info("B02/B03 :")

    for fl in active_classes:
        members = primary_keep_by_family.get(fl, [])
        # B01 keep
        kept_members = [r for r in members if r["sample_status"] == "keep"]
        total = len(kept_members)
        if total == 0:
            continue
        # B02: controlled + technique
        ct_count = sum(
            1 for r in kept_members
            if r["content_type"] in ("controlled", "technique")
        )
        ct_ratio = ct_count / total
        if ct_ratio < 0.80:
            logger.warning(
                " B02 %s: controlled+technique %.2f%% < 80%% value ",
                fl, ct_ratio * 100,
            )
        else:
            logger.info(
                " B02 %s: controlled+technique %.2f%%",
                fl, ct_ratio * 100,
            )

    # B03: min/max
    kept_per_family = []
    for fl in active_classes:
        cnt = primary_keep_stats[fl]["kept_recordings"]
        kept_per_family.append(cnt)
    if kept_per_family:
        min_kept = min(kept_per_family)
        max_kept = max(kept_per_family)
        if max_kept > 0:
            ratio = min_kept / max_kept
            if ratio < 0.20:
                logger.warning(
                    " B03: min/max %.2f < 0.20 ",
                    ratio,
                )
            elif ratio < 0.33:
                logger.warning(
                    " B03: min/max %.2f < 0.33 value ",
                    ratio,
                )
            else:
                logger.info(" B03: min/max %.2f", ratio)

    # Stats
    total_keep = sum(1 for r in audit_rows if r["sample_status"] == "keep")
    total_drop = sum(1 for r in audit_rows if r["sample_status"] == "drop")
    logger.info("Step 6 : keep=%d, drop=%d", total_keep, total_drop)

    return audit_rows


# Step 7: Freeze

def step7_freeze(
    audit_rows: list[dict[str, Any]],
    scan_rows: list[dict[str, Any]],
    freeze_config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """ sample_id rows frozen_manifest.csv

    §13 Step 7 + §3.2
      1. sample_id ALL audit row drop
      2.
      3. scan_manifest.csv final audit_manifest.csv frozen_manifest.csv

    Args:
        audit_rows: step6 rows
        scan_rows: step1 scan_manifest row
        freeze_config:
        output_dir: Output dir

    Returns:
        frozen_manifest row keep row
    """
    logger.info("=" * 60)
    logger.info("Step 7: Freeze ")
    logger.info("=" * 60)

    family_abbr: dict[str, str] = freeze_config["family_abbr"]
    active_classes: list[str] = freeze_config["active_classes"]

    # 7.1 sample_id —— §3.2
    logger.info(" sample_id ...")

    # (source_dataset, family_label)
    # (source_subtype_dir, recording_group_id, file_name)
    group_key_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in audit_rows:
        key = (row["source_dataset"], row["family_label"])
        group_key_rows.setdefault(key, []).append(row)

    for (source_ds, fl), members in group_key_rows.items():
        # source CCMusic → ccm ChMusic → chm
        if source_ds == "CCMusic":
            source_prefix = "ccm"
        elif source_ds == "External":
            source_prefix = "ext"
        else:
            source_prefix = "chm"
        abbr = family_abbr.get(fl, fl)

        # (source_subtype_dir, recording_group_id, file_name)
        members.sort(
            key=lambda r: (
                r["source_subtype_dir"],
                r["recording_group_id"],
                r["file_name"],
            )
        )

        for seq, row in enumerate(members, start=1):
            row["sample_id"] = f"{source_prefix}_{abbr}_{seq:04d}"

    logger.info(" sample_id : %d ", len(audit_rows))

    # 7.2
    logger.info(" ...")

    # 1: pending
    pending_rows = [
        r for r in audit_rows
        if r.get("manual_review_status") == "pending"
    ]
    if pending_rows:
        logger.error(
            " : %d pending "
            " manual_review.csv rows ",
            len(pending_rows),
        )
        for pr in pending_rows[:10]:
            logger.error("  - %s (%s)", pr["record_id"], pr["file_name"])
        sys.exit(1)

    # 2: ChMusic >= 4 kept accept_limited
    r05_decisions = freeze_config.get("chmusic_r05_decisions") or []
    r05_accept_limited: set[str] = set()
    for dec in r05_decisions:
        if dec.get("decision") == "accept_limited":
            r05_accept_limited.add(dec["family_label"])

    chm_keep_by_family: dict[str, int] = {}
    for row in audit_rows:
        if row["source_dataset"] != "ChMusic":
            continue
        if row["sample_status"] != "keep":
            continue
        fl = row["family_label"]
        chm_keep_by_family[fl] = chm_keep_by_family.get(fl, 0) + 1

    for fl in active_classes:
        chm_kept = chm_keep_by_family.get(fl, 0)
        if chm_kept < 4 and fl not in r05_accept_limited:
            logger.error(
                " R05 : ChMusic %s kept_recordings=%d < 4 "
                " freeze_config accept_limited ",
                fl, chm_kept,
            )
            sys.exit(1)

    # 3: —— family_label active_classes
    invalid_labels = set()
    for row in audit_rows:
        if row["sample_status"] == "keep":
            if row["family_label"] not in active_classes:
                invalid_labels.add(row["family_label"])
    if invalid_labels:
        logger.error(
            " : family_label active_classes : %s",
            invalid_labels,
        )
        sys.exit(1)

    logger.info(" ")

    # 7.3 manifest
    # 7.3a scan_manifest.csv final
    scan_path = output_dir / "scan_manifest.csv"
    _write_manifest(scan_rows, SCAN_MANIFEST_COLUMNS, scan_path)

    # 7.3b audit_manifest.csv sample_id
    # sample_id
    audit_rows.sort(key=lambda r: r["sample_id"])
    audit_path = output_dir / "audit_manifest.csv"
    _write_manifest(audit_rows, AUDIT_MANIFEST_COLUMNS, audit_path)

    # 7.3c frozen_manifest.csv keep row sample_id
    frozen_rows = [r for r in audit_rows if r["sample_status"] == "keep"]
    frozen_rows.sort(key=lambda r: r["sample_id"])
    frozen_path = output_dir / "frozen_manifest.csv"
    _write_manifest(frozen_rows, FROZEN_MANIFEST_COLUMNS, frozen_path)

    logger.info(
        "Step 7 : audit=%d, frozen=%d",
        len(audit_rows), len(frozen_rows),
    )

    return frozen_rows


# Step 8: Split

def step8_split(
    frozen_rows: list[dict[str, Any]],
    freeze_config: dict[str, Any],
    output_dir: Path,
) -> list[list[dict[str, Any]]]:
    """ 3 rows group-aware split_manifest_seed{0,1,2}.csv

    CCMusic group-aware greedy D
    ChMusic split=external_test

    Args:
        frozen_rows: step7 frozen_manifest row
        freeze_config:
        output_dir: Output dir

    Returns:
        3 seed split_manifest row
    """
    logger.info("=" * 60)
    logger.info("Step 8: Split ")
    logger.info("=" * 60)

    split_seeds: list[int] = freeze_config["split_seeds"]
    seed_ns: dict[str, int] = freeze_config["seed_namespaces"]
    active_classes: list[str] = freeze_config["active_classes"]

    target_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    # split
    split_priority = {"train": 0, "val": 1, "test": 2}

    # CCMusic / ChMusic
    primary_frozen = [r for r in frozen_rows if r["source_dataset"] in _PRIMARY_DATASETS]
    chm_frozen = [r for r in frozen_rows if r["source_dataset"] == "ChMusic"]

    all_split_rows: list[dict[str, Any]] = []

    for seed_idx, base_seed in enumerate(split_seeds):
        logger.info("--- Seed %d (base_seed=%d) ---", seed_idx, base_seed)
        split_seed_val = base_seed + seed_ns["split"]

        # 8.1 : group-aware greedy split D
        # family_label
        primary_by_family: dict[str, list[dict[str, Any]]] = {}
        for row in primary_frozen:
            fl = row["family_label"]
            primary_by_family.setdefault(fl, []).append(row)

        # family
        primary_split_assignments: dict[str, str] = {}  # record_id → split

        for fl in active_classes:
            family_rows = primary_by_family.get(fl, [])
            if not family_rows:
                continue

            # unique recording_group_id
            group_to_rows: dict[str, list[dict[str, Any]]] = {}
            for row in family_rows:
                rgid = row["recording_group_id"]
                group_to_rows.setdefault(rgid, []).append(row)

            # recording_group_id
            sorted_groups = sorted(group_to_rows.keys())

            # shuffle
            rng = random.Random(split_seed_val)
            rng.shuffle(sorted_groups)

            # group
            group_sizes: dict[str, int] = {
                g: len(group_to_rows[g]) for g in sorted_groups
            }
            total_N = sum(group_sizes.values())

            # Greedy
            split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
            group_assignments: dict[str, str] = {}
            remaining_groups = list(sorted_groups)

            for i, group_id in enumerate(sorted_groups):
                g_size = group_sizes[group_id]
                unplaced_after = len(sorted_groups) - i - 1 # group

                best_candidate: Optional[tuple[float, float, int, str]] = None

                for split_name in ("train", "val", "test"):
                    # >= split
                    trial_counts = dict(split_counts)
                    trial_counts[split_name] += g_size

                    # split
                    empty_after = sum(
                        1 for s in ("train", "val", "test")
                        if trial_counts[s] == 0
                    )
                    if unplaced_after < empty_after:
                        continue #

                    # target_error and max_abs_deviation
                    # total_N D: N
                    max_dev = 0.0
                    for s in ("train", "val", "test"):
                        dev = abs(trial_counts[s] / total_N - target_ratios[s]) if total_N > 0 else 0.0
                        if dev > max_dev:
                            max_dev = dev
                    target_error = max_dev

                    candidate = (target_error, max_dev, split_priority[split_name], split_name)
                    if best_candidate is None or candidate < best_candidate:
                        best_candidate = candidate

                if best_candidate is None:
                    # R03 >= 7
                    logger.error("Split : %s group=%s", fl, group_id)
                    sys.exit(1)

                chosen_split = best_candidate[3]
                split_counts[chosen_split] += g_size
                group_assignments[group_id] = chosen_split

            # group
            group_assignments = _local_repair_split(
                group_assignments, group_sizes, total_N, target_ratios, split_priority,
            )

            for group_id, split_name in group_assignments.items():
                for row in group_to_rows[group_id]:
                    primary_split_assignments[row["record_id"]] = split_name

            final_counts = {"train": 0, "val": 0, "test": 0}
            for g, s in group_assignments.items():
                final_counts[s] += group_sizes[g]
            logger.info(
                "  %s: train=%d val=%d test=%d (N=%d)",
                fl, final_counts["train"], final_counts["val"],
                final_counts["test"], total_N,
            )

        # 8.2 split_manifest row
        seed_rows: list[dict[str, Any]] = []

        # rows CCMusic + External
        for row in primary_frozen:
            split_row = dict(row)
            split_row["split_seed"] = seed_idx
            split_row["split"] = primary_split_assignments.get(row["record_id"], "train")
            seed_rows.append(split_row)

        # ChMusic row external_test
        for row in chm_frozen:
            split_row = dict(row)
            split_row["split_seed"] = seed_idx
            split_row["split"] = "external_test"
            seed_rows.append(split_row)

        # 8.3 §11.3
        _check_split_hard_gates(seed_rows, active_classes, seed_idx)

        # sample_id
        seed_rows.sort(key=lambda r: r["sample_id"])

        split_path = output_dir / f"split_manifest_seed{seed_idx}.csv"
        _write_manifest(seed_rows, SPLIT_MANIFEST_COLUMNS, split_path)

        all_split_rows.append(seed_rows)

    logger.info("Step 8 : 3 seed split_manifest")

    return all_split_rows


def _local_repair_split(
    assignments: dict[str, str],
    group_sizes: dict[str, int],
    total_N: int,
    target_ratios: dict[str, float],
    split_priority: dict[str, int],
) -> dict[str, str]:
    """ group D Step 8

    (group, from_split, to_split)
    split rows

    Args:
        assignments: group_id → split
        group_sizes: group_id →
        total_N:
        target_ratios:
        split_priority: split

    Returns:
        assignments
    """
    def _compute_max_deviation(counts: dict[str, int]) -> float:
        """ """
        if total_N == 0:
            return 0.0
        return max(
            abs(counts[s] / total_N - target_ratios[s])
            for s in ("train", "val", "test")
        )

    current_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for g, s in assignments.items():
        current_counts[s] += group_sizes[g]
    current_dev = _compute_max_deviation(current_counts)

    # value 1%
    if current_dev <= 0.01:
        return assignments

    improved = True
    while improved:
        improved = False
        best_move: Optional[tuple[float, str, str, str]] = None  # (dev, group, from, to)

        for group_id, from_split in list(assignments.items()):
            g_size = group_sizes[group_id]
            for to_split in ("train", "val", "test"):
                if to_split == from_split:
                    continue
                # from_split
                trial_counts = dict(current_counts)
                trial_counts[from_split] -= g_size
                trial_counts[to_split] += g_size
                if trial_counts[from_split] <= 0:
                    continue # split
                trial_dev = _compute_max_deviation(trial_counts)
                if trial_dev < current_dev:
                    candidate = (trial_dev, group_id, from_split, to_split)
                    if best_move is None or candidate[0] < best_move[0]:
                        best_move = candidate

        if best_move is not None:
            dev, group_id, from_s, to_s = best_move
            g_size = group_sizes[group_id]
            assignments[group_id] = to_s
            current_counts[from_s] -= g_size
            current_counts[to_s] += g_size
            current_dev = dev
            improved = True

    return assignments


def _check_split_hard_gates(
    split_rows: list[dict[str, Any]],
    active_classes: list[str],
    seed_idx: int,
) -> None:
    """ §11.3

    - train/val/test >= 1 group
    - recording_group_id split

    Args:
        split_rows: seed split_manifest row
        active_classes:
        seed_idx:
    """
    # 1: (family_label, split) 1 group
    for fl in active_classes:
        for split_name in ("train", "val", "test"):
            groups_in_split = set()
            for row in split_rows:
                if (row["family_label"] == fl
                        and row["split"] == split_name
                        and row["source_dataset"] in _PRIMARY_DATASETS):
                    groups_in_split.add(row["recording_group_id"])
            if not groups_in_split:
                logger.error(
                    "Split seed %d : %s/%s group ",
                    seed_idx, fl, split_name,
                )
                sys.exit(1)

    # 2: recording_group_id split
    rgid_splits: dict[str, set[str]] = {}
    for row in split_rows:
        rgid = row.get("recording_group_id", "")
        if rgid:
            rgid_splits.setdefault(rgid, set()).add(row["split"])
    for rgid, splits in rgid_splits.items():
        if len(splits) > 1:
            logger.error(
                "Split seed %d : recording_group_id=%s split: %s",
                seed_idx, rgid, splits,
            )
            sys.exit(1)


# Step 9: Segment

def step9_segment(
    split_rows: list[list[dict[str, Any]]],
    freeze_config: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> list[list[dict[str, Any]]]:
    """ seed rowsResample WAV

    §13 Step 9 + §12
      1. 24kHz Resample（soxr_hq）
      2. 5 s 3-5 s
      3. active_frame_ratio < value
      4. CCMusic per_recording_segment_cap
      5. WAV segments_seed{n}/<split>/

    Args:
        split_rows: step8 3 seed split_manifest row
        freeze_config:
        config: config.yaml
        output_dir: Output dir

    Returns:
        3 seed segment_manifest row
    """
    import librosa
    import soundfile as sf
    from tqdm import tqdm

    logger.info("=" * 60)
    logger.info("Step 9: Segment ")
    logger.info("=" * 60)

    target_sr = 24000
    segment_duration = 5.0
    segment_samples = int(target_sr * segment_duration)  # 120000

    thresholds = freeze_config["thresholds"]
    default_afr_threshold: float = float(thresholds.get("active_frame_ratio", 0.20))
    concentration_cap_floor: int = int(thresholds.get("concentration_cap_floor", 6))
    active_classes: list[str] = freeze_config["active_classes"]

    all_segment_rows: list[list[dict[str, Any]]] = []

    # 24kHz Resample record_id → y_24k ( seed /Resample)
    resample_cache: dict[str, np.ndarray] = {}

    for seed_idx, seed_split_rows in enumerate(split_rows):
        logger.info("--- Seed %d ---", seed_idx)

        # Output dir rerun
        seg_root = output_dir / f"segments_seed{seed_idx}"
        if seg_root.exists():
            import shutil
            shutil.rmtree(seg_root)
            logger.info(" : %s", seg_root)
        for split_dir in ("train", "val", "test", "external_test"):
            (seg_root / split_dir).mkdir(parents=True, exist_ok=True)

        # Stats CCMusic kept
        # Stats kept
        primary_kept_per_class: dict[str, int] = {}
        for row in seed_split_rows:
            if row["source_dataset"] in _PRIMARY_DATASETS:
                fl = row["family_label"]
                primary_kept_per_class[fl] = primary_kept_per_class.get(fl, 0) + 1

        segment_rows: list[dict[str, Any]] = []

        for row in tqdm(seed_split_rows, desc=f"Seed {seed_idx} ", unit=" "):
            source_path = Path(row["source_path"])
            sample_id = row["sample_id"]
            family_label = row["family_label"]
            source_dataset = row["source_dataset"]
            split_name = row["split"]

            # per-family active_frame_ratio value
            afr_threshold = _resolve_threshold(
                family_label, source_dataset,
                "active_frame_ratio", default_afr_threshold,
                freeze_config,
            )

            # 9.1 Resample 24kHz seed
            record_id = row["record_id"]
            if record_id in resample_cache:
                y_24k = resample_cache[record_id]
            else:
                try:
                    y, sr = librosa.load(str(source_path), sr=None, mono=True)
                    y_24k = librosa.resample(
                        y, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq",
                    )
                except Exception as e:
                    logger.warning(" /Resample : %s — %s", source_path, e)
                    continue
                resample_cache[record_id] = y_24k

            total_samples = len(y_24k)
            duration_24k = total_samples / target_sr

            # 9.2
            raw_segments: list[dict[str, Any]] = []

            if duration_24k < 3.0:
                # D04 Skip
                continue
            elif duration_24k < 5.0:
                # 3-5 s 5 s 1
                padded = np.zeros(segment_samples, dtype=y_24k.dtype)
                padded[:total_samples] = y_24k
                raw_segments.append({
                    "data": padded,
                    "start_sec": 0.0,
                    "end_sec": segment_duration,
                    "is_padded": True,
                })
            else:
                # >= 5 s
                num_full_segments = total_samples // segment_samples
                for seg_i in range(num_full_segments):
                    start_sample = seg_i * segment_samples
                    end_sample = start_sample + segment_samples
                    seg_data = y_24k[start_sample:end_sample]
                    start_sec = start_sample / target_sr
                    end_sec = end_sample / target_sr
                    raw_segments.append({
                        "data": seg_data,
                        "start_sec": round(start_sec, 6),
                        "end_sec": round(end_sec, 6),
                        "is_padded": False,
                    })
                # 5 s

            # 9.3 §12.3
            filtered_segments: list[dict[str, Any]] = []
            for seg in raw_segments:
                afr = _compute_active_frame_ratio_24k(seg["data"], target_sr)
                seg["active_frame_ratio"] = afr
                if afr >= afr_threshold:
                    filtered_segments.append(seg)

            # 9.4 §12.4
            if source_dataset in _PRIMARY_DATASETS and filtered_segments:
                kept_in_class = primary_kept_per_class.get(family_label, 0)
                family_cap_floor = int(_resolve_threshold(
                    family_label, source_dataset,
                    "concentration_cap_floor", float(concentration_cap_floor),
                    freeze_config,
                ))
                cap = max(family_cap_floor, math.ceil(kept_in_class * 0.10))
                if len(filtered_segments) > cap:
                    # N start_time_sec ——
                    filtered_segments = filtered_segments[:cap]

            # 9.5 WAV segment_manifest row
            for seg_i, seg in enumerate(filtered_segments):
                segment_id = f"{sample_id}_s{seg_i:02d}"
                seg_filename = f"{segment_id}.wav"
                seg_rel_path = f"{split_name}/{seg_filename}"
                seg_abs_path = seg_root / seg_rel_path

                # WAV PCM_16
                sf.write(
                    str(seg_abs_path),
                    seg["data"],
                    samplerate=target_sr,
                    subtype="PCM_16",
                )

                segment_rows.append({
                    "record_id": row["record_id"],
                    "sample_id": sample_id,
                    "source_dataset": source_dataset,
                    "family_label": family_label,
                    "split_seed": seed_idx,
                    "split": split_name,
                    "segment_id": segment_id,
                    "segment_index": seg_i,
                    "start_time_sec": seg["start_sec"],
                    "end_time_sec": seg["end_sec"],
                    "segment_path": seg_rel_path,
                    "is_padded": seg["is_padded"],
                    "overlap_ratio": 0.0,
                    "selection_method": "all_after_energy_filter_and_concentration_cap",
                    "active_frame_ratio": seg["active_frame_ratio"],
                })

        # (sample_id, segment_index)
        segment_rows.sort(key=lambda r: (r["sample_id"], r["segment_index"]))

        # segment_manifest
        seg_manifest_path = output_dir / f"segment_manifest_seed{seed_idx}.csv"
        _write_manifest(segment_rows, SEGMENT_MANIFEST_COLUMNS, seg_manifest_path)

        logger.info(
            " Seed %d: %d ",
            seed_idx, len(segment_rows),
        )
        all_segment_rows.append(segment_rows)

    logger.info("Step 9 ")

    return all_segment_rows


def _compute_active_frame_ratio_24k(
    y: np.ndarray,
    sr: int = 24000,
) -> float:
    """ 24kHz Resample §12.3

    §7.5 25ms / 10ms hop / -40 dBFS value
    active_frame_ratio = /

    Args:
        y: 24kHz Mono float waveform
        sr: Sample rate 24000

    Returns:
        active_frame_ratio (0.0 ~ 1.0)
    """
    import librosa

    frame_length = int(0.025 * sr)  # 600 samples @ 24kHz
    hop_length = int(0.010 * sr)    # 240 samples @ 24kHz

    if len(y) < frame_length:
        return 0.0

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=0))
    rms_db = 20.0 * np.log10(rms_per_frame + 1e-10)

    active_frames = rms_db >= -40.0
    return float(active_frames.sum()) / len(active_frames)


# Step 10: Final Checks + Export

def step10_final_checks(
    split_rows_per_seed: list[list[dict[str, Any]]],
    segment_rows_per_seed: list[list[dict[str, Any]]],
    freeze_config: dict[str, Any],
    output_dir: Path,
) -> None:
    """ rows frozen_summary.json

    §15
      1. pending
      2.
      3.
      4.
      5. recording_group_id split
      6. (family_label, split) >= 1
      7. ChMusic >= 4 kept accept_limited
      8. Stats manifest

    Args:
        split_rows_per_seed: step8 3 seed split_manifest row
        segment_rows_per_seed: step9 3 seed segment_manifest row
        freeze_config:
        output_dir: Output dir
    """
    from importlib.metadata import version as pkg_version

    logger.info("=" * 60)
    logger.info("Step 10: ")
    logger.info("=" * 60)

    active_classes: list[str] = freeze_config["active_classes"]
    split_seeds: list[int] = freeze_config["split_seeds"]
    target_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

    # seed 0 split_rows frozen_manifest seed
    seed0_split_rows = split_rows_per_seed[0]

    # R05
    r05_decisions = freeze_config.get("chmusic_r05_decisions") or []
    r05_accept_limited: set[str] = set()
    for dec in r05_decisions:
        if dec.get("decision") == "accept_limited":
            r05_accept_limited.add(dec["family_label"])

    # external_test
    ext_exemptions = freeze_config.get("external_test_exemptions") or []
    ext_exempt_set: set[tuple[str, int]] = set()
    for ex in ext_exemptions:
        fl = ex.get("family_label", "")
        seed = ex.get("seed")
        if seed is not None:
            ext_exempt_set.add((fl, seed))
        else:
            # seed seed
            for si in range(len(split_seeds)):
                ext_exempt_set.add((fl, si))

    errors: list[str] = []

    # 10.1 1: pending
    for row in seed0_split_rows:
        if row.get("manual_review_status") == "pending":
            errors.append(
                f" 1 : {row['record_id']} manual_review_status=pending"
            )

    # 10.2 2:
    for row in seed0_split_rows:
        if row["family_label"] not in active_classes:
            errors.append(
                f" 2 : {row['record_id']} family_label={row['family_label']} "
                f" active_classes "
            )

    # 10.3 3:
    for row in seed0_split_rows:
        if row.get("is_decodable") is not True:
            errors.append(
                f" 3 : {row['record_id']} is_decodable=false frozen_manifest"
            )

    # 10.4 4:
    sha_seen: dict[str, str] = {}
    for row in seed0_split_rows:
        sha = row.get("audio_sha256", "")
        if not sha:
            continue
        if sha in sha_seen:
            errors.append(
                f" 4 : audio_sha256 ——{row['record_id']} {sha_seen[sha]}"
            )
        else:
            sha_seen[sha] = row["record_id"]

    # 10.5 5: recording_group_id split seed
    for seed_idx, seed_rows in enumerate(split_rows_per_seed):
        rgid_splits: dict[str, set[str]] = {}
        for row in seed_rows:
            rgid = row.get("recording_group_id", "")
            if rgid:
                rgid_splits.setdefault(rgid, set()).add(row["split"])
        for rgid, splits in rgid_splits.items():
            if len(splits) > 1:
                errors.append(
                    f" 5 seed {seed_idx} : "
                    f"recording_group_id={rgid} split: {splits}"
                )

    # 10.6 6: (family_label, split) >= 1
    for seed_idx, seg_rows in enumerate(segment_rows_per_seed):
        seg_coverage: dict[tuple[str, str], int] = {}
        for seg in seg_rows:
            key = (seg["family_label"], seg["split"])
            seg_coverage[key] = seg_coverage.get(key, 0) + 1

        for fl in active_classes:
            for split_name in ("train", "val", "test"):
                count = seg_coverage.get((fl, split_name), 0)
                if count < 1:
                    errors.append(
                        f" 6 seed {seed_idx} : "
                        f"{fl}/{split_name} =0"
                    )
            # external_test
            ext_count = seg_coverage.get((fl, "external_test"), 0)
            if ext_count < 1 and (fl, seed_idx) not in ext_exempt_set:
                errors.append(
                    f" 6 seed {seed_idx} : "
                    f"{fl}/external_test =0 "
                )

    # 10.7 7: ChMusic >= 4 kept accept_limited
    chm_kept: dict[str, int] = {}
    for row in seed0_split_rows:
        if row["source_dataset"] == "ChMusic":
            fl = row["family_label"]
            chm_kept[fl] = chm_kept.get(fl, 0) + 1
    for fl in active_classes:
        cnt = chm_kept.get(fl, 0)
        if cnt < 4 and fl not in r05_accept_limited:
            errors.append(
                f" 7 : ChMusic {fl} kept={cnt} < 4 accept_limited"
            )

    # 10.8 segment_manifest ↔ got WAV file
    logger.info(" 8: ")

    for seed_idx, seg_rows in enumerate(segment_rows_per_seed):
        seg_root = output_dir / f"segments_seed{seed_idx}"
        # 8a: manifest segment_path file
        for seg in seg_rows:
            seg_abs = seg_root / seg["segment_path"]
            if not seg_abs.is_file():
                errors.append(
                    f" 8 seed {seed_idx} : "
                    f"manifest {seg['segment_path']} filenot found"
                )
        # 8b: manifest extra wav
        manifest_paths: set[str] = {seg["segment_path"] for seg in seg_rows}
        for split_dir in ("train", "val", "test", "external_test"):
            split_path = seg_root / split_dir
            if not split_path.is_dir():
                continue
            for wav_file in split_path.iterdir():
                if wav_file.suffix.lower() == ".wav":
                    rel = f"{split_dir}/{wav_file.name}"
                    if rel not in manifest_paths:
                        errors.append(
                            f" 8 seed {seed_idx} : "
                            f" extrafile {rel} manifest "
                        )

    # 8
    if errors:
        logger.error(" %d errors:", len(errors))
        for i, err in enumerate(errors, 1):
            logger.error("  [%d] %s", i, err)
        sys.exit(1)

    logger.info(" ")

    # 10.9 frozen_summary.json §14.3
    logger.info(" frozen_summary.json ...")

    # --- per_class_stats ---
    per_class_stats: dict[str, dict[str, Any]] = {}
    for fl in active_classes:
        # seed0 Stats
        fl_rows = [r for r in seed0_split_rows if r["family_label"] == fl]
        ccm_rows_fl = [r for r in fl_rows if r["source_dataset"] == "CCMusic"]
        ext_rows_fl = [r for r in fl_rows if r["source_dataset"] == "External"]
        chm_rows_fl = [r for r in fl_rows if r["source_dataset"] == "ChMusic"]
        primary_rows_fl = ccm_rows_fl + ext_rows_fl

        unique_groups = set(r["recording_group_id"] for r in primary_rows_fl)
        content_counts: dict[str, int] = {}
        for r in primary_rows_fl:
            ct = r.get("content_type", "")
            content_counts[ct] = content_counts.get(ct, 0) + 1

        per_class_stats[fl] = {
            "ccmusic_kept_recordings": len(ccm_rows_fl),
            "external_kept_recordings": len(ext_rows_fl),
            "chmusic_kept_recordings": len(chm_rows_fl),
            "primary_unique_groups": len(unique_groups),
            "content_type_distribution": content_counts,
        }

    # --- drop_reason_counts audit_manifest —— frozen seed0_split_rows drop row
    # audit_manifest.csv ---
    drop_reason_counts: dict[str, int] = {}
    audit_path = output_dir / "audit_manifest.csv"
    if audit_path.is_file():
        with audit_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for ar in reader:
                if ar.get("sample_status") == "drop":
                    dr = ar.get("drop_reason", "unknown")
                    drop_reason_counts[dr] = drop_reason_counts.get(dr, 0) + 1

    # --- split_deviation_log ---
    split_deviation_log: list[dict[str, Any]] = []
    for seed_idx, seed_rows in enumerate(split_rows_per_seed):
        primary_rows_seed = [r for r in seed_rows if r["source_dataset"] in _PRIMARY_DATASETS]
        total_N = len(primary_rows_seed)
        if total_N == 0:
            continue
        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        for r in primary_rows_seed:
            s = r["split"]
            if s in split_counts:
                split_counts[s] += 1
        deviations: dict[str, float] = {}
        for s in ("train", "val", "test"):
            actual_ratio = split_counts[s] / total_N
            deviations[s] = round(actual_ratio - target_ratios[s], 6)
        split_deviation_log.append({
            "seed_index": seed_idx,
            "base_seed": split_seeds[seed_idx],
            "total_primary": total_N,
            "split_counts": split_counts,
            "deviation_from_target": deviations,
        })

    # --- post_segmentation_checks ---
    post_seg_checks: list[dict[str, Any]] = []
    for seed_idx, seg_rows in enumerate(segment_rows_per_seed):
        seg_per_class: dict[str, dict[str, int]] = {}
        for seg in seg_rows:
            fl = seg["family_label"]
            s = seg["split"]
            seg_per_class.setdefault(fl, {})
            seg_per_class[fl][s] = seg_per_class[fl].get(s, 0) + 1
        post_seg_checks.append({
            "seed_index": seed_idx,
            "total_segments": len(seg_rows),
            "per_class_per_split": seg_per_class,
        })

    # --- audio_processing_config ---
    audio_config: dict[str, str] = {}
    for pkg_name in ("librosa", "soundfile", "soxr"):
        try:
            audio_config[pkg_name] = pkg_version(pkg_name)
        except Exception:
            audio_config[pkg_name] = "unknown"
    audio_config["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # frozen_summary
    frozen_summary: dict[str, Any] = {
        "freeze_config_snapshot": freeze_config,
        "classes": active_classes,
        "per_class_stats": per_class_stats,
        "drop_reason_counts": drop_reason_counts,
        "split_deviation_log": split_deviation_log,
        "post_segmentation_checks": post_seg_checks,
        "audio_processing_config": audio_config,
    }

    summary_path = output_dir / "frozen_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(frozen_summary, f, ensure_ascii=False, indent=2)
    logger.info(" Write %s", summary_path)

    logger.info("Step 10 ")


# main

def main(config_path: Optional[str] = None) -> None:
    """ rows Step 0 → Step 10

    Args:
        config_path: config.yaml rows
    """
    # rows
    if config_path is None:
        import argparse
        parser = argparse.ArgumentParser(
            description=" Step 0-10 "
        )
        parser.add_argument(
            "config",
            type=str,
            help="config.yaml file ",
        )
        args = parser.parse_args()
        config_path = args.config

    cfg_path = Path(config_path).resolve()

    # Step 0:
    preflight_result = step0_preflight(cfg_path)

    config = preflight_result["config"]
    freeze_config = preflight_result["freeze_config"]
    manual_review = preflight_result["manual_review"]
    output_dir = preflight_result["output_dir"]

    # manual_review config Step 5
    config["_manual_review"] = manual_review

    # Step 1: + record_id
    scan_rows = step1_scan(config, freeze_config, output_dir)

    # Step 2: Metadata extraction
    audit_rows = step2_metadata(scan_rows, freeze_config, output_dir)

    # Step 3:
    audit_rows = step3_auto_qc(audit_rows, freeze_config)

    # Step 4:
    audit_rows = step4_apply_manual_review(audit_rows, manual_review, freeze_config, output_dir)

    # Step 5: +
    audit_rows = step5_dedup_group(audit_rows, freeze_config, config, output_dir)

    # Step 6:
    audit_rows = step6_quota(audit_rows, freeze_config)

    # Step 7:
    frozen_rows = step7_freeze(audit_rows, scan_rows, freeze_config, output_dir)

    # Step 8:
    split_rows_per_seed = step8_split(frozen_rows, freeze_config, output_dir)

    # Step 9:
    segment_rows_per_seed = step9_segment(
        split_rows_per_seed, freeze_config, config, output_dir,
    )

    # Step 10:
    step10_final_checks(
        split_rows_per_seed, segment_rows_per_seed, freeze_config, output_dir,
    )

    logger.info("=" * 60)
    logger.info(" Step 0 ~ Step 10 rows ")
    logger.info("Output dir: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
