# The EEG fixture

One real, public electroencephalography (EEG) recording with its sidecars, used by the recipe tests as the input of `nemar-floor`.
The synthetic smoke input of the runtime image is in `tests/fixtures/smoke/`; this one is the scientific fixture.

## Source and attribution

Source: "A COVID-19 survivors and close contacts EEG dataset", OpenNeuro ds007823, version 1.0.1 (doi:10.18112/openneuro.ds007823.v1.0.1); NEMAR record on007823 (doi:10.82901/nemar.on007823), data-plane version tag `v1.0.0`.

Authors: Ana Calzada-Reyes, Eduardo Aubert-Vázquez, Lidice Galán-García, Maria Luisa Bringas-Vega, Trinidad Virués-Alba, Lidia Charroó-Ruiz, Yanely Acosta-Imás, Mitchell Valdés-Sosa, Laura Perez-Mayo, Joel Gutiérrez-Gil, Antonio Caballero-Moreno, Miguel Angel Alvarez, Norge Santiesteban, Javier Vicente Sánchez-Lopez, Annette Valdés-Virués, Elba Elvira Varona-Galindo, Elizabeth Méndez-Parra, Joviana Castro-Valiente, Leyanis Ramos-Hernández, Mabel Whilby-Santiesteban, Thelma Luz Carrillo-Alfonso, Shahwar Yasir, Yu Kin, Peng Ren, Dezhong Yao, Luo Cheng, Roberto Rodriguez-Labrada, Pedro Valdés-Sosa (Cuban Neuroscience Center and collaborators).

Licence: CC0 1.0 Universal (a public-domain dedication, declared in `dataset_description.json`); attribution is given as a courtesy.

Reference: Calzada-Reyes A. et al., "EEG Signatures of COVID-19 Survival compared to close contacts and the Cuban EEG normative database", Clinical Neurophysiology Practice 2026;11:174-186, https://doi.org/10.1016/j.cnp.2026.02.007

Used here unmodified, except that `participants.tsv`, `participants.json` and `datacite.yml` are not copied.
The first two hold the rows of the other 168 participants, which the floor never reads, and the third is bibliographic metadata of no use to a test.

## What is here

```text
manifest.json                the file list in the shape of the data plane's manifest (see below)
dataset/                     the Brain Imaging Data Structure (BIDS) tree, byte for byte as published
  dataset_description.json
  README.md  CHANGES
  sub-CUCOV076/sub-CUCOV076_scans.tsv
  sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.edf       the recording (European Data Format)
  sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_eeg.json
  sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_channels.tsv
  sub-CUCOV076/eeg/sub-CUCOV076_electrodes.tsv
  sub-CUCOV076/eeg/sub-CUCOV076_coordsystem.json
  sub-CUCOV076/eeg/sub-CUCOV076_task-COVID_events.tsv
```

The recording is resting EEG with eyes closed: 22 channels at 200 Hz for 282 s (56,400 samples).
Nineteen channels are scalp EEG with 10-20 locations in `electrodes.tsv` (the Montreal Neurological Institute 305 coordinate system, in millimetres), two further EEG-typed channels (`Pg1`, `Pg2`) have no location, and one is `ECG` (electrocardiography).
The declared power line frequency is 60 Hz, and the acquisition filters declared in `channels.tsv` are 0.5 and 70 Hz.
The files total 2,492,895 bytes, under the 5,000,000 the phase allows.

## How it was checked

All ten files were downloaded from NEMAR's public data plane (`on007823`, version `v1.0.0`) and each matched its manifest entry: the Secure Hash Algorithm 256 (SHA-256) for the recording and the git blob Secure Hash Algorithm 1 (SHA-1) for the text files.
The recording's SHA-256 is `b2029c3f30ee2d84839ee97be5abe39ee4b1960bcb6881ca2ac600ae7dc603d3`, equal to the checksum the data plane publishes.
`manifest.json` is computed from the committed bytes by `scripts/make_eeg_fixture_manifest.py` (`--check` fails when it differs), and `tests/recipes/test_fixture.py` compares it with the tree, with the published checksum and with the size limit.
The data-plane stand-in of the end-to-end test serves `dataset/` and builds its own manifest from the same bytes.

To re-create the tree, fetch each path of `manifest.json` from `<data plane>/on007823/v1.0.0/<path>`, compare it with its entry, and run the script.
