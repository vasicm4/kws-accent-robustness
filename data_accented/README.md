# Serbian-accented test set

19 speakers with a Serbian accent saying the 10 target words
(`yes no up down left right on off stop go`), cut into 510 one-second clips.
The audio is **not** in this repository; only this description is. See
[Consent and privacy](#consent-and-privacy).

| | |
|---|---|
| Speakers | 19 (`spk01`–`spk19`) |
| Clips | 510: 16 speakers × 3 reps × 10 words, 3 speakers × 1 rep × 10 words |
| Format | 16 kHz mono 16-bit WAV, 1 s, word centred |
| Layout | `clips/<word>/<speaker>_<word>_<rep>.wav`, same as Speech Commands |

## Collection protocol

Each speaker recorded themselves on their own phone, saying each word 1 or 3
times in a row in the fixed order above (`yes yes yes no no no …`). One
speaker sent one file per word. Recordings came in as phone voice-recorder
files (AAC), WhatsApp voice notes (Opus) and WhatsApp audio (AAC). Five
recordings (`spk07`–`spk11`) had background noise or talking and were cleaned
up by hand in Audacity and re-exported as WAV before processing.

## How the clips were made

`scripts/prepare_accented.py` turns the raw recordings into clips, with no
manual cutting:

1. Decode every file to 16 kHz mono (PyAV).
2. Find words by short-time energy, with a threshold set per recording
   between its own noise floor and speech level.
3. Drop breaths, clicks and record-button taps; re-join words split at a
   stop consonant (`stop`, `up`).
4. Accept the recording only if this gives exactly 10 or 30 words (1 or 3
   reps) and no "word" is longer than 1.6 s; otherwise flag it for review.
5. Label words by position, and cut a 1 s window around each from the
   original audio (the speaker's real background, not zero padding).

`--verify` cross-checks every clip with Whisper (run locally). Whisper
disagreements were only used to choose what to listen to, never to drop
clips: Whisper also struggles with accented speech, so filtering on it would
remove exactly the hard cases this set exists to measure. The check caught
two recordings whose labels had shifted by one word; these are why step 4
rejects over-long words.

```
uv run --extra accented python scripts/prepare_accented.py --src <raw recordings dir> [--verify]
```

Outputs, all git-ignored: `clips/`, `clips.csv` (one row per clip: path,
speaker, label, rep, word duration, Whisper transcript), `sources.csv` (one
row per recording) and `metadata.csv` (one row per speaker).
`private/file_map.csv` maps raw filenames to speaker IDs.

## Consent and privacy

Every speaker filled in a written consent form (Google Form, in Serbian)
covering what is recorded, how it is stored and used, and the right to
withdraw. Publication of the recordings was a separate, optional question.

- Speakers are identified only by `spkXX`. The link between names and IDs
  exists only in the private consent responses, never in this repository.
- Raw recordings and clips are kept off GitHub.
- Recordings are deleted by 31 December 2027 unless the speaker agreed to
  publication.
- Only clips from speakers who agreed to publication may ever be released
  (CC BY 4.0). Until then, only aggregate results are published.
- A speaker who withdraws: delete their rows from `private/file_map.csv`
  and their raw recording, then re-run the script.

## Known limitations

- **Codec mismatch.** These clips went through lossy phone/WhatsApp codecs;
  Speech Commands is uncompressed WAV. Part of any accuracy drop may come
  from the codec rather than the accent.
- **Small and uneven.** 19 speakers, 3 of whom said each word once. Per-word
  and per-speaker numbers rest on few clips.
- **Hand-cleaned recordings.** `spk07`–`spk11` were edited in Audacity, so
  their background differs from the rest.
