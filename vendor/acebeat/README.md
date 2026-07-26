# acebeat

A small, MIT-licensed, dependency-free BPM (tempo) and musical key detector,
written to replace a vendored AGPL-3.0 dependency (Essentia) that was
previously used for this purpose in [acestep.cpp](../..).

## Clean-room statement

This library was written from published, public-domain DSP techniques and
academic descriptions only:

- Cooley-Tukey FFT (Cooley & Tukey, 1965) for the spectral front-end.
- Spectral-flux onset detection + autocorrelation for tempo estimation
  (the general family described in e.g. Ellis, "Beat Tracking by Dynamic
  Programming," 2007).
- A 12-bin pitch-class (chroma) profile, refined with sub-harmonic
  summation and per-track tuning-frequency calibration (Gomez, "Tonal
  Description of Music Audio Signals," 2006), correlated against
  major/minor key templates built from published, corpus-derived profile
  weights (Faraldo, Jorda & Herrera, "A Multi-Profile Method for Key
  Estimation in EDM," AES 142nd Convention, 2017 -- chosen over the older
  Krumhansl & Kessler 1990 profile because this library's real target
  corpus is EDM/pop/hip-hop-heavy, not classical solo tones).

No Essentia source code was read, copied, translated, or referenced while
writing this library. It is not intended to bit-match Essentia's output —
only to provide a reasonable, independently-verifiable estimate using
well-known, decades-old, public techniques.

## Scope

`acebeat` intentionally does **not** decode audio files. Callers are
expected to provide a mono `float` buffer at a known sample rate (WAV/MP3
decoding, resampling, and channel downmixing are the caller's
responsibility, and are already solved elsewhere in acestep.cpp). The one
exception is `tools/acebeat-cli`, a small standalone validation tool with
its own minimal file reader, which is not part of the library itself.

## Status

Under active development — see `../../TRAINING_DEV.md` in the parent repo
for the phased implementation log.
