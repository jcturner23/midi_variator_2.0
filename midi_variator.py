#!/usr/bin/env python3
"""Generate configurable melodic variations for MIDI files."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mido
import yaml


SCALE_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass
class NoteEvent:
    note: int
    velocity: int
    channel: int
    onset_beats: float
    duration_beats: float
    quantized_onset_beats: float = 0.0
    micro_onset_delta: float = 0.0
    quantized_duration_beats: float = 0.0
    micro_duration_delta: float = 0.0

    def clone(self) -> "NoteEvent":
        return copy.deepcopy(self)


@dataclass
class Phrase:
    start_idx: int
    end_idx: int
    start_beats: float
    end_beats: float
    fingerprint: Dict[str, List[float]]
    repeats_with: List[int] = field(default_factory=list)


@dataclass
class ScaleCandidate:
    root_pc: int
    scale_name: str
    score: float
    coverage: float

    @property
    def label(self) -> str:
        return f"{PITCH_CLASS_NAMES[self.root_pc]} {self.scale_name}"


@dataclass
class TrackAnalysis:
    track_index: int
    track_name: str
    notes: List[NoteEvent]
    phrases: List[Phrase]
    rhythmic_motifs: List[List[float]]
    scale_candidates: List[ScaleCandidate]
    out_of_scale_note_indices: List[int]


@dataclass
class TrackData:
    track_index: int
    track_name: str
    notes: List[NoteEvent]
    passthrough_messages: List[Tuple[int, mido.Message]]


def parse_grid_to_beats(grid: str) -> float:
    token = grid.strip().upper()
    if not token.startswith("1/"):
        raise ValueError(f"Unsupported quantize grid format: {grid!r}")
    denominator_token = token[2:]
    triplet = denominator_token.endswith("T")
    if triplet:
        denominator_token = denominator_token[:-1]
    denominator = int(denominator_token)
    if denominator <= 0:
        raise ValueError("Grid denominator must be positive.")
    beats = 4.0 / float(denominator)
    if triplet:
        beats *= 2.0 / 3.0
    return beats


def quantize_to_grid(value_beats: float, grid_beats: float) -> float:
    if grid_beats <= 0:
        return value_beats
    return round(value_beats / grid_beats) * grid_beats


def flatten(values: Iterable[Sequence[float]]) -> List[float]:
    merged: List[float] = []
    for value in values:
        merged.extend(value)
    return merged


def nearest_pitch_in_scale(note: int, root_pc: int, scale_name: str) -> int:
    allowed = {(root_pc + interval) % 12 for interval in SCALE_INTERVALS[scale_name]}
    if note % 12 in allowed:
        return note
    for distance in range(1, 12):
        up = note + distance
        down = note - distance
        if up % 12 in allowed:
            return up
        if down % 12 in allowed:
            return down
    return note


def build_scale_pitch_classes(root_pc: int, scale_name: str) -> set[int]:
    return {(root_pc + interval) % 12 for interval in SCALE_INTERVALS[scale_name]}


def expand_related_scales(root_pc: int, scale_name: str, include_relative: bool, include_adjacent: bool) -> List[Tuple[int, str]]:
    related: List[Tuple[int, str]] = [(root_pc, scale_name)]
    if include_relative:
        if scale_name == "major":
            related.append(((root_pc + 9) % 12, "natural_minor"))
        elif scale_name in {"natural_minor", "harmonic_minor", "melodic_minor"}:
            related.append(((root_pc + 3) % 12, "major"))
    if include_adjacent:
        related.extend(
            [
                ((root_pc + 7) % 12, scale_name),
                ((root_pc + 5) % 12, scale_name),
                (root_pc, "dorian" if scale_name == "natural_minor" else "mixolydian"),
            ]
        )
    # preserve order, unique pairs only
    seen: set[Tuple[int, str]] = set()
    unique: List[Tuple[int, str]] = []
    for item in related:
        if item not in seen and item[1] in SCALE_INTERVALS:
            unique.append(item)
            seen.add(item)
    return unique


def read_midi_tracks(midi_path: Path) -> Tuple[mido.MidiFile, List[TrackData]]:
    mid = mido.MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat
    tracks: List[TrackData] = []
    for track_idx, track in enumerate(mid.tracks):
        abs_ticks = 0
        active: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        notes: List[NoteEvent] = []
        passthrough: List[Tuple[int, mido.Message]] = []
        track_name = f"track_{track_idx}"
        for msg in track:
            abs_ticks += msg.time
            if msg.type == "track_name":
                track_name = msg.name
            if msg.type == "note_on" and msg.velocity > 0:
                key = (msg.note, getattr(msg, "channel", 0))
                active.setdefault(key, []).append((abs_ticks, msg.velocity))
            elif msg.type in ("note_off", "note_on"):
                velocity = 0 if msg.type == "note_off" else msg.velocity
                if msg.type == "note_on" and velocity > 0:
                    continue
                key = (msg.note, getattr(msg, "channel", 0))
                if key in active and active[key]:
                    start_ticks, on_velocity = active[key].pop(0)
                    duration_ticks = max(1, abs_ticks - start_ticks)
                    onset_beats = start_ticks / ticks_per_beat
                    duration_beats = duration_ticks / ticks_per_beat
                    notes.append(
                        NoteEvent(
                            note=msg.note,
                            velocity=on_velocity,
                            channel=getattr(msg, "channel", 0),
                            onset_beats=onset_beats,
                            duration_beats=duration_beats,
                        )
                    )
            else:
                passthrough.append((abs_ticks, msg.copy(time=0)))
        # Close any hanging notes at end of track.
        track_end_ticks = abs_ticks
        for (note, channel), stack in active.items():
            for start_ticks, on_velocity in stack:
                duration_ticks = max(1, track_end_ticks - start_ticks)
                notes.append(
                    NoteEvent(
                        note=note,
                        velocity=on_velocity,
                        channel=channel,
                        onset_beats=start_ticks / ticks_per_beat,
                        duration_beats=duration_ticks / ticks_per_beat,
                    )
                )
        notes.sort(key=lambda note: (note.onset_beats, note.note))
        tracks.append(
            TrackData(
                track_index=track_idx,
                track_name=track_name,
                notes=notes,
                passthrough_messages=passthrough,
            )
        )
    return mid, tracks


def compute_scale_candidates(notes: List[NoteEvent], allowed_scale_names: List[str], top_n: int) -> Tuple[List[ScaleCandidate], List[int]]:
    if not notes:
        return [], []
    weighted_pcs: Dict[int, float] = {}
    total_weight = 0.0
    for note in notes:
        weight = max(0.01, note.duration_beats)
        weighted_pcs[note.note % 12] = weighted_pcs.get(note.note % 12, 0.0) + weight
        total_weight += weight

    candidates: List[ScaleCandidate] = []
    for scale_name in allowed_scale_names:
        if scale_name not in SCALE_INTERVALS:
            continue
        for root in range(12):
            allowed_pcs = build_scale_pitch_classes(root, scale_name)
            in_scale = sum(weight for pc, weight in weighted_pcs.items() if pc in allowed_pcs)
            score = in_scale / max(0.001, total_weight)
            coverage = sum(1.0 for n in notes if n.note % 12 in allowed_pcs) / float(len(notes))
            candidates.append(ScaleCandidate(root_pc=root, scale_name=scale_name, score=score, coverage=coverage))
    candidates.sort(key=lambda candidate: (candidate.score, candidate.coverage), reverse=True)
    selected = candidates[: max(1, top_n)]
    best = selected[0]
    allowed = build_scale_pitch_classes(best.root_pc, best.scale_name)
    out_of_scale_indices = [idx for idx, note in enumerate(notes) if note.note % 12 not in allowed]
    return selected, out_of_scale_indices


def annotate_quantization(notes: List[NoteEvent], grid_beats: float) -> None:
    for note in notes:
        quantized_onset = quantize_to_grid(note.onset_beats, grid_beats)
        quantized_duration = max(grid_beats / 2.0, quantize_to_grid(note.duration_beats, grid_beats / 2.0))
        note.quantized_onset_beats = quantized_onset
        note.micro_onset_delta = note.onset_beats - quantized_onset
        note.quantized_duration_beats = quantized_duration
        note.micro_duration_delta = note.duration_beats - quantized_duration


def build_phrase_fingerprint(notes: Sequence[NoteEvent]) -> Dict[str, List[float]]:
    if not notes:
        return {"intervals": [], "durations": [], "ioi": []}
    intervals: List[float] = []
    durations = [round(n.quantized_duration_beats, 3) for n in notes]
    ioi: List[float] = []
    for idx in range(1, len(notes)):
        intervals.append(float(notes[idx].note - notes[idx - 1].note))
        ioi.append(round(notes[idx].quantized_onset_beats - notes[idx - 1].quantized_onset_beats, 3))
    return {"intervals": intervals, "durations": durations, "ioi": ioi}


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    size = min(len(a), len(b))
    if size == 0:
        return 0.0
    vec_a = a[:size]
    vec_b = b[:size]
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(y * y for y in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def compare_phrase_fingerprints(first: Dict[str, List[float]], second: Dict[str, List[float]]) -> float:
    interval_similarity = cosine_similarity(first["intervals"], second["intervals"])
    duration_similarity = cosine_similarity(first["durations"], second["durations"])
    ioi_similarity = cosine_similarity(first["ioi"], second["ioi"])
    return (interval_similarity * 0.45) + (duration_similarity * 0.3) + (ioi_similarity * 0.25)


def detect_phrases(notes: List[NoteEvent], max_rest_beats: float, min_notes: int, similarity_threshold: float) -> List[Phrase]:
    phrases: List[Phrase] = []
    if not notes:
        return phrases
    phrase_start = 0
    for idx in range(1, len(notes)):
        previous = notes[idx - 1]
        current = notes[idx]
        previous_end = previous.quantized_onset_beats + previous.quantized_duration_beats
        rest = current.quantized_onset_beats - previous_end
        if rest >= max_rest_beats:
            if idx - phrase_start >= min_notes:
                phrase_notes = notes[phrase_start:idx]
                phrases.append(
                    Phrase(
                        start_idx=phrase_start,
                        end_idx=idx - 1,
                        start_beats=phrase_notes[0].quantized_onset_beats,
                        end_beats=phrase_notes[-1].quantized_onset_beats + phrase_notes[-1].quantized_duration_beats,
                        fingerprint=build_phrase_fingerprint(phrase_notes),
                    )
                )
            phrase_start = idx
    if len(notes) - phrase_start >= min_notes:
        phrase_notes = notes[phrase_start:]
        phrases.append(
            Phrase(
                start_idx=phrase_start,
                end_idx=len(notes) - 1,
                start_beats=phrase_notes[0].quantized_onset_beats,
                end_beats=phrase_notes[-1].quantized_onset_beats + phrase_notes[-1].quantized_duration_beats,
                fingerprint=build_phrase_fingerprint(phrase_notes),
            )
        )
    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            similarity = compare_phrase_fingerprints(phrases[i].fingerprint, phrases[j].fingerprint)
            if similarity >= similarity_threshold:
                phrases[i].repeats_with.append(j)
                phrases[j].repeats_with.append(i)
    return phrases


def detect_rhythmic_motifs(notes: List[NoteEvent], motif_size: int = 4) -> List[List[float]]:
    if len(notes) < motif_size:
        return []
    motifs: List[List[float]] = []
    for idx in range(len(notes) - motif_size + 1):
        window = notes[idx : idx + motif_size]
        if not window:
            continue
        motif = [round(window[0].quantized_duration_beats, 3)]
        motif.extend(round(window[i].quantized_onset_beats - window[i - 1].quantized_onset_beats, 3) for i in range(1, len(window)))
        motifs.append(motif)
    return motifs


def apply_phrase_repeat(
    notes: List[NoteEvent],
    phrases: List[Phrase],
    config: Dict[str, Any],
    rng: random.Random,
) -> List[NoteEvent]:
    if not config.get("enabled", False) or not phrases:
        return [note.clone() for note in notes]
    probability = float(config.get("probability", 0.5))
    transforms = list(config.get("transform_on_repeat", ["none"]))
    result = [note.clone() for note in notes]
    offset_push = 0.0
    for phrase in phrases:
        if rng.random() > probability:
            continue
        phrase_notes = [note.clone() for note in notes[phrase.start_idx : phrase.end_idx + 1]]
        phrase_length = phrase.end_beats - phrase.start_beats
        repeat_start = phrase.end_beats + offset_push
        chosen_transform = rng.choice(transforms) if transforms else "none"
        for note in phrase_notes:
            note.quantized_onset_beats = (note.quantized_onset_beats - phrase.start_beats) + repeat_start
            if chosen_transform == "octave_up":
                note.note += 12
            elif chosen_transform == "octave_down":
                note.note -= 12
            elif chosen_transform == "diatonic_sequence":
                note.note += 2
            note.onset_beats = note.quantized_onset_beats + note.micro_onset_delta
            note.duration_beats = max(0.05, note.quantized_duration_beats + note.micro_duration_delta)
            result.append(note)
        offset_push += phrase_length
    result.sort(key=lambda note: (note.quantized_onset_beats, note.note))
    return result


def apply_time_scale(
    notes: List[NoteEvent],
    phrases: List[Phrase],
    config: Dict[str, Any],
    rng: random.Random,
) -> List[NoteEvent]:
    if not config.get("enabled", False):
        return [note.clone() for note in notes]
    factors = [float(value) for value in config.get("factors", [1.0])]
    if not factors:
        return [note.clone() for note in notes]
    preserve_note_ends = bool(config.get("preserve_note_ends", True))
    updated = [note.clone() for note in notes]
    targets = phrases if phrases else [Phrase(0, len(notes) - 1, notes[0].quantized_onset_beats, notes[-1].quantized_onset_beats, {})] if notes else []
    for phrase in targets:
        factor = rng.choice(factors)
        if abs(factor - 1.0) < 1e-6:
            continue
        origin = phrase.start_beats
        for idx in range(phrase.start_idx, phrase.end_idx + 1):
            if idx < 0 or idx >= len(updated):
                continue
            note = updated[idx]
            relative_start = note.quantized_onset_beats - origin
            relative_duration = note.quantized_duration_beats
            note.quantized_onset_beats = origin + (relative_start / factor)
            note.quantized_duration_beats = max(0.05, relative_duration / factor)
            if preserve_note_ends:
                note.duration_beats = max(0.05, note.quantized_duration_beats + note.micro_duration_delta)
            else:
                note.duration_beats = note.quantized_duration_beats
            note.onset_beats = note.quantized_onset_beats + note.micro_onset_delta
    updated.sort(key=lambda note: (note.quantized_onset_beats, note.note))
    return updated


def apply_octave_shift(notes: List[NoteEvent], phrases: List[Phrase], config: Dict[str, Any], rng: random.Random) -> List[NoteEvent]:
    if not config.get("enabled", False):
        return [note.clone() for note in notes]
    choices = [int(choice) for choice in config.get("choices", [0])]
    probability = float(config.get("per_phrase_probability", 0.3))
    shifted = [note.clone() for note in notes]
    if phrases:
        for phrase in phrases:
            if rng.random() > probability:
                continue
            shift = rng.choice(choices)
            for idx in range(phrase.start_idx, phrase.end_idx + 1):
                if 0 <= idx < len(shifted):
                    shifted[idx].note += shift
    else:
        shift = rng.choice(choices) if choices else 0
        for note in shifted:
            note.note += shift
    return shifted


def apply_scale_remap(
    notes: List[NoteEvent],
    scale_candidates: List[ScaleCandidate],
    analysis_cfg: Dict[str, Any],
    mode_cfg: Dict[str, Any],
    rng: random.Random,
) -> List[NoteEvent]:
    if not mode_cfg.get("enabled", False) or not scale_candidates:
        return [note.clone() for note in notes]
    borrowing_probability = float(mode_cfg.get("borrowing_probability", 0.0))
    snap_out_of_scale = bool(mode_cfg.get("snap_out_of_scale", True))
    base = scale_candidates[0]
    include_relative = bool(analysis_cfg.get("scale", {}).get("include_relative_keys", True))
    include_adjacent = bool(analysis_cfg.get("scale", {}).get("include_adjacent_scales", True))
    candidates = expand_related_scales(base.root_pc, base.scale_name, include_relative, include_adjacent)
    remapped: List[NoteEvent] = []
    for original in notes:
        note = original.clone()
        root_pc, scale_name = candidates[0]
        if len(candidates) > 1 and rng.random() < borrowing_probability:
            root_pc, scale_name = rng.choice(candidates[1:])
        allowed = build_scale_pitch_classes(root_pc, scale_name)
        if snap_out_of_scale or note.note % 12 not in allowed:
            note.note = nearest_pitch_in_scale(note.note, root_pc, scale_name)
        remapped.append(note)
    return remapped


def apply_rhythm_displacement(notes: List[NoteEvent], grid_beats: float, config: Dict[str, Any], rng: random.Random) -> List[NoteEvent]:
    if not config.get("enabled", False):
        return [note.clone() for note in notes]
    max_steps = int(config.get("max_grid_shift_steps", 1))
    preserve_micro = bool(config.get("preserve_microtiming_delta", True))
    displaced: List[NoteEvent] = []
    for source in notes:
        note = source.clone()
        step = rng.randint(-max_steps, max_steps)
        note.quantized_onset_beats += step * grid_beats
        if preserve_micro:
            note.onset_beats = note.quantized_onset_beats + note.micro_onset_delta
        else:
            note.onset_beats = note.quantized_onset_beats
            note.micro_onset_delta = 0.0
        displaced.append(note)
    displaced.sort(key=lambda n: (n.quantized_onset_beats, n.note))
    return displaced


def apply_syncopation(notes: List[NoteEvent], grid_beats: float, config: Dict[str, Any], rng: random.Random) -> List[NoteEvent]:
    if not config.get("enabled", False):
        return [note.clone() for note in notes]
    probability = float(config.get("probability", 0.2))
    max_shift = int(config.get("max_grid_shift_steps", 1))
    shifted: List[NoteEvent] = []
    for source in notes:
        note = source.clone()
        if rng.random() < probability:
            step = rng.choice([-max_shift, max_shift])
            note.quantized_onset_beats += step * grid_beats * 0.5
            note.onset_beats = note.quantized_onset_beats + note.micro_onset_delta
        shifted.append(note)
    shifted.sort(key=lambda n: (n.quantized_onset_beats, n.note))
    return shifted


def apply_density_transform(notes: List[NoteEvent], config: Dict[str, Any], rng: random.Random) -> List[NoteEvent]:
    if not config.get("enabled", False):
        return [note.clone() for note in notes]
    add_probability = float(config.get("add_probability", 0.15))
    drop_probability = float(config.get("drop_probability", 0.1))
    max_notes = int(config.get("max_generated_notes", 64))
    transformed: List[NoteEvent] = []
    generated = 0
    for source in notes:
        if rng.random() < drop_probability:
            continue
        note = source.clone()
        transformed.append(note)
        if generated < max_notes and rng.random() < add_probability:
            generated_note = note.clone()
            generated_note.note = max(0, min(127, generated_note.note + rng.choice([-2, 2, 4, -4])))
            generated_note.quantized_onset_beats += generated_note.quantized_duration_beats * 0.5
            generated_note.onset_beats = generated_note.quantized_onset_beats + generated_note.micro_onset_delta
            generated_note.duration_beats = max(0.05, generated_note.duration_beats * 0.8)
            transformed.append(generated_note)
            generated += 1
    transformed.sort(key=lambda n: (n.quantized_onset_beats, n.note))
    return transformed


def apply_generation_constraints(notes: List[NoteEvent], constraints: Dict[str, Any]) -> List[NoteEvent]:
    constrained: List[NoteEvent] = []
    pitch_range = constraints.get("pitch_range", [0, 127])
    min_pitch, max_pitch = int(pitch_range[0]), int(pitch_range[1])
    max_interval = int(constraints.get("max_interval_semitones", 24))
    previous_note: Optional[int] = None
    for source in notes:
        note = source.clone()
        note.note = max(min_pitch, min(max_pitch, note.note))
        if previous_note is not None:
            leap = note.note - previous_note
            if abs(leap) > max_interval:
                note.note = previous_note + max_interval if leap > 0 else previous_note - max_interval
                note.note = max(min_pitch, min(max_pitch, note.note))
        note.duration_beats = max(0.01, note.duration_beats)
        note.quantized_duration_beats = max(0.01, note.quantized_duration_beats)
        constrained.append(note)
        previous_note = note.note
    return constrained


def run_analysis(track: TrackData, config: Dict[str, Any]) -> TrackAnalysis:
    analysis_cfg = config.get("analysis", {})
    scale_cfg = analysis_cfg.get("scale", {})
    phrase_cfg = analysis_cfg.get("phrase", {})
    quant_grid = parse_grid_to_beats(analysis_cfg.get("quantize_grid", "1/16"))
    notes = [note.clone() for note in track.notes]
    annotate_quantization(notes, quant_grid)
    scale_candidates, out_of_scale = compute_scale_candidates(
        notes=notes,
        allowed_scale_names=list(scale_cfg.get("candidate_scales", ["major", "natural_minor"])),
        top_n=int(scale_cfg.get("top_n", 3)),
    )
    phrases = detect_phrases(
        notes=notes,
        max_rest_beats=float(phrase_cfg.get("max_rest_beats_to_split", 0.75)),
        min_notes=int(phrase_cfg.get("min_notes", 4)),
        similarity_threshold=float(phrase_cfg.get("similarity_threshold", 0.8)),
    )
    motifs = detect_rhythmic_motifs(notes)
    return TrackAnalysis(
        track_index=track.track_index,
        track_name=track.track_name,
        notes=notes,
        phrases=phrases,
        rhythmic_motifs=motifs,
        scale_candidates=scale_candidates,
        out_of_scale_note_indices=out_of_scale,
    )


def apply_mode_chain(analysis: TrackAnalysis, config: Dict[str, Any], rng: random.Random) -> List[NoteEvent]:
    variation_cfg = config.get("variation", {})
    analysis_cfg = config.get("analysis", {})
    constraints_cfg = config.get("musical_constraints", {})
    grid_beats = parse_grid_to_beats(analysis_cfg.get("quantize_grid", "1/16"))
    mode_order = list(
        variation_cfg.get(
            "mode_order",
            ["phrase_repeat", "time_scale", "octave_shift", "scale_remap", "rhythm_displacement"],
        )
    )
    current = [note.clone() for note in analysis.notes]
    for mode in mode_order:
        if mode == "phrase_repeat":
            current = apply_phrase_repeat(current, analysis.phrases, variation_cfg.get("phrase_repeat", {}), rng)
        elif mode == "time_scale":
            current = apply_time_scale(current, analysis.phrases, variation_cfg.get("time_scale", {}), rng)
        elif mode == "octave_shift":
            current = apply_octave_shift(current, analysis.phrases, variation_cfg.get("octave_shift", {}), rng)
        elif mode == "scale_remap":
            current = apply_scale_remap(
                current,
                analysis.scale_candidates,
                analysis_cfg,
                variation_cfg.get("scale_remap", {}),
                rng,
            )
        elif mode == "rhythm_displacement":
            current = apply_rhythm_displacement(current, grid_beats, variation_cfg.get("rhythm_displacement", {}), rng)
        elif mode == "syncopation":
            current = apply_syncopation(current, grid_beats, variation_cfg.get("syncopation", {}), rng)
        elif mode == "density_transform":
            current = apply_density_transform(current, variation_cfg.get("density_transform", {}), rng)
    constrained = apply_generation_constraints(current, constraints_cfg)
    return constrained


def write_variation_midi(
    source_midi: mido.MidiFile,
    tracks: List[TrackData],
    generated_tracks: Dict[int, List[NoteEvent]],
    output_path: Path,
) -> None:
    output = mido.MidiFile(type=source_midi.type, ticks_per_beat=source_midi.ticks_per_beat)
    ticks_per_beat = source_midi.ticks_per_beat
    for source_track in tracks:
        output_track = mido.MidiTrack()
        generated_notes = generated_tracks.get(source_track.track_index, source_track.notes)
        absolute_events: List[Tuple[int, int, mido.Message]] = []
        for abs_ticks, passthrough in source_track.passthrough_messages:
            absolute_events.append((abs_ticks, 0, passthrough.copy(time=0)))
        for note in generated_notes:
            start_tick = max(0, int(round(note.onset_beats * ticks_per_beat)))
            end_tick = max(start_tick + 1, int(round((note.onset_beats + note.duration_beats) * ticks_per_beat)))
            absolute_events.append((start_tick, 1, mido.Message("note_on", note=note.note, velocity=note.velocity, channel=note.channel, time=0)))
            absolute_events.append((end_tick, 2, mido.Message("note_off", note=note.note, velocity=0, channel=note.channel, time=0)))
        absolute_events.sort(key=lambda row: (row[0], row[1]))
        last_tick = 0
        for absolute_tick, _, message in absolute_events:
            delta = max(0, absolute_tick - last_tick)
            output_track.append(message.copy(time=delta))
            last_tick = absolute_tick
        output.tracks.append(output_track)
    output.save(str(output_path))


def build_analysis_report(config: Dict[str, Any], analyses: List[TrackAnalysis], output_midi: Path) -> Dict[str, Any]:
    report_tracks: List[Dict[str, Any]] = []
    for analysis in analyses:
        report_tracks.append(
            {
                "track_index": analysis.track_index,
                "track_name": analysis.track_name,
                "note_count": len(analysis.notes),
                "detected_scales": [
                    {
                        "label": candidate.label,
                        "score": round(candidate.score, 4),
                        "coverage": round(candidate.coverage, 4),
                    }
                    for candidate in analysis.scale_candidates
                ],
                "phrase_count": len(analysis.phrases),
                "phrases": [
                    {
                        "start_idx": phrase.start_idx,
                        "end_idx": phrase.end_idx,
                        "start_beats": round(phrase.start_beats, 4),
                        "end_beats": round(phrase.end_beats, 4),
                        "repeats_with": phrase.repeats_with,
                    }
                    for phrase in analysis.phrases
                ],
                "rhythmic_motif_count": len(analysis.rhythmic_motifs),
                "out_of_scale_indices": analysis.out_of_scale_note_indices,
            }
        )
    return {
        "output_midi": str(output_midi),
        "analysis_grid": config.get("analysis", {}).get("quantize_grid", "1/16"),
        "tracks": report_tracks,
    }


def validate_generated_notes(track_name: str, notes: List[NoteEvent]) -> None:
    for idx, note in enumerate(notes):
        if note.duration_beats <= 0:
            raise ValueError(f"{track_name}: negative/zero duration at note {idx}")
        if note.note < 0 or note.note > 127:
            raise ValueError(f"{track_name}: invalid midi note {note.note} at note {idx}")
        if note.onset_beats < -0.001:
            raise ValueError(f"{track_name}: negative onset at note {idx}")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in (".yaml", ".yml"):
            return yaml.safe_load(handle) or {}
        if path.suffix.lower() == ".json":
            return json.load(handle)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def merge_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = copy.deepcopy(config)
    io_cfg = merged.setdefault("io", {})
    gen_cfg = merged.setdefault("generation", {})
    var_cfg = merged.setdefault("variation", {})
    if args.input:
        io_cfg["input_midi"] = args.input
    if args.output_dir:
        io_cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        gen_cfg["random_seed"] = args.seed
    if args.modes:
        var_cfg["mode_order"] = [token.strip() for token in args.modes.split(",") if token.strip()]
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    missing = []
    io_cfg = config.get("io", {})
    if "input_midi" not in io_cfg:
        missing.append("io.input_midi")
    if "output_dir" not in io_cfg:
        missing.append("io.output_dir")
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    grid = config.get("analysis", {}).get("quantize_grid", "1/16")
    parse_grid_to_beats(grid)
    scales = config.get("analysis", {}).get("scale", {}).get("candidate_scales", ["major", "natural_minor"])
    unknown = [name for name in scales if name not in SCALE_INTERVALS]
    if unknown:
        raise ValueError(f"Unknown scale names in analysis.scale.candidate_scales: {unknown}")


def run(config: Dict[str, Any]) -> None:
    validate_config(config)
    io_cfg = config.get("io", {})
    generation_cfg = config.get("generation", {})
    include_tracks = set(io_cfg.get("include_tracks", []))
    outputs_per_track = int(generation_cfg.get("outputs_per_track", 1))
    deterministic = bool(generation_cfg.get("deterministic", False))
    base_seed = int(generation_cfg.get("random_seed", 0))

    midi_path = Path(io_cfg["input_midi"]).expanduser().resolve()
    output_dir = Path(io_cfg["output_dir"]).expanduser().resolve()
    output_prefix = io_cfg.get("output_prefix", "variation")
    ensure_output_dir(output_dir)

    source_midi, source_tracks = read_midi_tracks(midi_path)
    selected_tracks = [
        track
        for track in source_tracks
        if (not include_tracks or track.track_index in include_tracks) and len(track.notes) > 0
    ]
    analyses = [run_analysis(track, config) for track in selected_tracks]

    if not analyses:
        raise ValueError("No tracks selected for variation (check include_tracks and source MIDI note content).")

    for iteration in range(outputs_per_track):
        iteration_seed = base_seed if deterministic else base_seed + iteration
        rng = random.Random(iteration_seed)
        generated: Dict[int, List[NoteEvent]] = {}
        for analysis in analyses:
            notes = apply_mode_chain(analysis, config, rng)
            validate_generated_notes(analysis.track_name, notes)
            generated[analysis.track_index] = notes

        output_midi = output_dir / f"{output_prefix}_{iteration + 1:02d}.mid"
        write_variation_midi(source_midi, source_tracks, generated, output_midi)
        report = build_analysis_report(config, analyses, output_midi)
        report_path = output_dir / f"{output_prefix}_{iteration + 1:02d}.analysis.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Wrote: {output_midi}")
        print(f"Wrote: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MIDI track variations with musical analysis.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config.")
    parser.add_argument("--input", help="Override input midi path.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--modes", help="Comma-separated override for variation.mode_order.")
    parser.add_argument("--seed", type=int, help="Override generation.random_seed.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    config = merge_cli_overrides(config, args)
    run(config)


if __name__ == "__main__":
    main()
