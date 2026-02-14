# midi_variator_2.0

Python MIDI variation tool with scale, rhythm, and phrase analysis.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Copy and edit one of the config templates:

- `variation_config.example.yaml`
- `variation_config.example.json`

Run:

```bash
python midi_variator.py --config variation_config.example.yaml
```

Optional CLI overrides:

```bash
python midi_variator.py \
  --config variation_config.example.yaml \
  --input /path/to/input.mid \
  --output-dir out \
  --modes phrase_repeat,time_scale,octave_shift,scale_remap,rhythm_displacement \
  --seed 123
```

## What it does

- Analyzes each selected track for:
  - likely scales/keys from pitch-class fit,
  - quantized rhythm and preserved microtiming offsets,
  - phrase boundaries and repeated phrases by melodic/rhythmic fingerprint.
- Applies composable variation modes:
  - phrase repetition/transformation,
  - double-time/half-time phrase scaling,
  - octave shifts,
  - scale-aware remapping with optional relative/adjacent borrowing,
  - rhythm displacement, optional syncopation, optional density transform.
- Writes:
  - one or more varied MIDI files,
  - matching `.analysis.json` reports with detected scales, phrase map, and summary metadata.

## Microtiming behavior

Onsets are quantized to a grid for rhythm analysis, while each note stores its
early/late onset delta. During generation, transformed notes keep that
microtiming offset when `preserve_microtiming_delta` is enabled.
