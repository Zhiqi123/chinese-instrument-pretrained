"""Split commercial album WAVs into individual tracks by silence gaps."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_duration(wav_path: Path) -> float:
    """Return file duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(wav_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _detect_silences(
    wav_path: Path,
    silence_thresh_db: float,
    min_silence_dur: float,
) -> list[tuple[float, float]]:
    """Detect silence intervals using ffmpeg silencedetect.

    Returns list of (start_sec, end_sec) for each silence region.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(wav_path),
            "-af", f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_dur}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    # silencedetect outputs to stderr
    stderr = result.stderr

    starts: list[float] = []
    ends: list[float] = []
    for line in stderr.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        m_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_start:
            starts.append(float(m_start.group(1)))
        if m_end:
            ends.append(float(m_end.group(1)))

    # Pair up: each silence region has a start and end
    silences = list(zip(starts, ends[:len(starts)]))
    return silences


def _compute_segments(
    total_duration: float,
    silences: list[tuple[float, float]],
    min_track_dur: float,
) -> list[tuple[float, float]]:
    """Convert silence regions into track segments (start, end).

    Merges short segments (< min_track_dur) with the previous track.
    """
    # Cut points are the midpoints of each silence region
    cut_points = [(s + e) / 2 for s, e in silences]

    # Build raw segments
    boundaries = [0.0] + cut_points + [total_duration]
    raw_segments = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]

    # Merge short segments into previous
    merged: list[tuple[float, float]] = []
    for start, end in raw_segments:
        dur = end - start
        if merged and dur < min_track_dur:
            # Extend previous segment
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Check first segment after merge (could still be short if it was the
    # very first segment and had no predecessor to merge into)
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_track_dur:
        _, second_end = merged[1]
        merged[0] = (merged[0][0], second_end)
        merged.pop(1)

    return merged


def _extract_segment(
    wav_path: Path,
    output_path: Path,
    start: float,
    end: float,
) -> None:
    """Extract a segment using ffmpeg (stream copy for WAV, no re-encode)."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c", "copy",
            str(output_path),
        ],
        capture_output=True, text=True, check=True,
    )


# main logic


def _sanitize_stem(name: str) -> str:
    """Remove label/artist prefix for cleaner output filenames.

    '龙音唱片-李静古筝独奏专辑' → '李静古筝独奏专辑'
    '刘德海-琵琶独奏精选'       → '刘德海琵琶独奏精选'
    """
    # Remove everything before and including the first '-' or '—'
    cleaned = re.sub(r"^[^-—]+-", "", name)
    return cleaned


def split_album(
    wav_path: Path,
    *,
    silence_thresh_db: float = -40,
    min_silence_dur: float = 2.0,
    min_track_dur: float = 10.0,
    dry_run: bool = False,
) -> list[Path]:
    """Split a single album WAV into track files.
    Returns list of output paths (empty if dry_run).
    """
    logger.info("Processing: %s", wav_path.name)

    total_dur = _get_duration(wav_path)
    logger.info("  Total duration: %.1fs (%.1f min)", total_dur, total_dur / 60)

    silences = _detect_silences(wav_path, silence_thresh_db, min_silence_dur)
    logger.info("  Detected %d silence regions", len(silences))

    segments = _compute_segments(total_dur, silences, min_track_dur)
    logger.info("  %d tracks after merging", len(segments))

    for i, (s, e) in enumerate(segments, 1):
        logger.info("    track_%02d: %7.1fs → %7.1fs  (%.1fs)", i, s, e, e - s)

    if dry_run:
        return []

    # Create output directory
    tracks_dir = wav_path.parent / "tracks"
    tracks_dir.mkdir(exist_ok=True)

    stem = _sanitize_stem(wav_path.stem)
    outputs: list[Path] = []
    for i, (start, end) in enumerate(segments, 1):
        out_path = tracks_dir / f"{stem}_track_{i:02d}.wav"
        _extract_segment(wav_path, out_path, start, end)
        outputs.append(out_path)
        logger.info("  ✓ %s (%.1fs)", out_path.name, end - start)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split commercial album WAVs into individual tracks by silence gaps",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="External data root (containing guzheng/ pipa/ subdirs)",
    )
    parser.add_argument(
        "--silence-thresh", type=float, default=-40,
        help="Silence threshold dB (default: -40)",
    )
    parser.add_argument(
        "--min-silence-dur", type=float, default=2.0,
        help="Minimum silence duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--min-track-dur", type=float, default=10.0,
        help="Minimum track duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print detected segments only, do not write files",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        logger.error("Directory not found: %s", data_dir)
        sys.exit(1)

    total_tracks = 0
    for family_dir in sorted(data_dir.iterdir()):
        if not family_dir.is_dir():
            continue
        wav_files = sorted(family_dir.glob("*.wav"))
        # Skip tracks/ subdirectory files
        wav_files = [f for f in wav_files if f.parent.name != "tracks"]
        if not wav_files:
            continue

        logger.info("── %s ──", family_dir.name)
        for wav in wav_files:
            outputs = split_album(
                wav,
                silence_thresh_db=args.silence_thresh,
                min_silence_dur=args.min_silence_dur,
                min_track_dur=args.min_track_dur,
                dry_run=args.dry_run,
            )
            total_tracks += len(outputs)

    if args.dry_run:
        logger.info("Dry-run complete, no files written")
    else:
        logger.info("Done: split %d tracks total", total_tracks)


if __name__ == "__main__":
    main()
