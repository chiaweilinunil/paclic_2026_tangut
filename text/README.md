# Corpus transcriptions

`build_dataset.py` reads one UTF-8 `*.txt` file per work from this directory,
one transcription line per line of the file. Tangut characters are Unicode
(U+17000–U+187FF and the Tangut Components block); lacunae, editorial brackets
and illegibility marks are stripped during extraction.

The transcriptions are **not distributed here**. They come from a separate
ongoing corpus project; each file derives from a published edition with its own
attribution and reuse terms. See the paper for the list of works and their
sources.

With this directory empty, `build_dataset.py` exits with a message rather than
producing a partial dataset. To reproduce the released build, place the
transcriptions here and run:

    python src/build_dataset/build_dataset.py

`dataset/build_config.json` records a SHA-256 digest of every input file, so a
rebuild can be checked against the released one file by file.
