import unittest

from midi_variator import (
    NoteEvent,
    apply_generation_constraints,
    annotate_quantization,
    compute_scale_candidates,
    parse_grid_to_beats,
    quantize_to_grid,
)


class MidiVariatorTests(unittest.TestCase):
    def test_parse_grid_triplet(self) -> None:
        self.assertAlmostEqual(parse_grid_to_beats("1/8T"), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(parse_grid_to_beats("1/16"), 0.25, places=6)

    def test_quantize_and_microtiming_preserved(self) -> None:
        note = NoteEvent(note=60, velocity=100, channel=0, onset_beats=1.12, duration_beats=0.37)
        annotate_quantization([note], 0.25)
        self.assertAlmostEqual(note.quantized_onset_beats, quantize_to_grid(1.12, 0.25))
        self.assertAlmostEqual(note.onset_beats, note.quantized_onset_beats + note.micro_onset_delta)

    def test_scale_candidates_prefers_c_major(self) -> None:
        notes = [
            NoteEvent(note=pitch, velocity=90, channel=0, onset_beats=float(i), duration_beats=0.5)
            for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72])
        ]
        candidates, _ = compute_scale_candidates(notes, ["major", "natural_minor"], 2)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].root_pc, 0)
        self.assertEqual(candidates[0].scale_name, "major")

    def test_generation_constraints_no_negative_duration(self) -> None:
        notes = [
            NoteEvent(note=120, velocity=90, channel=0, onset_beats=0.0, duration_beats=0.0),
            NoteEvent(note=10, velocity=90, channel=0, onset_beats=0.5, duration_beats=0.2),
        ]
        annotate_quantization(notes, 0.25)
        out = apply_generation_constraints(notes, {"pitch_range": [36, 96], "max_interval_semitones": 12})
        self.assertEqual(out[0].note, 96)
        self.assertEqual(out[1].note, 84)
        self.assertGreater(out[0].duration_beats, 0.0)
        self.assertGreater(out[1].duration_beats, 0.0)


if __name__ == "__main__":
    unittest.main()
