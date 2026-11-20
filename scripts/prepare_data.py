import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"
SLICE_BYTES = {"enwik8-1MB": 1_000_000, "enwik8-10MB": 10_000_000,
               "enwik8-100MB": 100_000_000}
DICTIONARY_WORDS = "4096"
DICTIONARY_MIN_FREQ = "4"


def build_preprocessor(source, binary):
    subprocess.run(
        ["cc", "-O3", "-fno-math-errno", "-fno-trapping-math", "-w",
         '-DCONFIG_VERSION="tufazip v2"', str(source), "-lm", "-o", str(binary)],
        check=True)


def prepare(data_dir, preprocess_source, slices):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    enwik8 = data_dir / "enwik8"
    if not enwik8.exists():
        archive = data_dir / "enwik8.zip"
        urllib.request.urlretrieve(ENWIK8_URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extract("enwik8", data_dir)
    binary = data_dir / "preprocess"
    build_preprocessor(preprocess_source, binary)
    for name in slices:
        raw = data_dir / name
        meta_path = data_dir / f"{name}.json"
        if meta_path.exists():
            continue
        raw.write_bytes(enwik8.read_bytes()[:SLICE_BYTES[name]])
        subprocess.run(
            [str(binary), "c", f"{raw}.voc", str(raw), f"{raw}.pp",
             DICTIONARY_WORDS, DICTIONARY_MIN_FREQ],
            check=True, stdout=subprocess.DEVNULL)
        tokens = np.frombuffer(Path(f"{raw}.pp").read_bytes(), dtype=">u2")
        meta_path.write_text(json.dumps({
            "raw_bytes": SLICE_BYTES[name], "tokens": len(tokens),
            "vocab_size": int(tokens.max()) + 1,
            "voc_bytes": Path(f"{raw}.voc").stat().st_size}))
    return {name: json.loads((data_dir / f"{name}.json").read_text()) for name in slices}


if __name__ == "__main__":
    print(prepare(sys.argv[1], sys.argv[2], sys.argv[3:]))
