"""Download and load the CAPE-Bench parquet files (CC BY 4.0) from Hugging Face."""
import json
import urllib.request

import pandas as pd

from .config import CAPE_DIR, HF_DATASET, RELEASED_EPISODE_RUNS, SCENARIO_SETS

JSON_FIELDS = [
    "permissions", "files", "db", "commands", "critical", "goal", "always_unsafe",
    "unsafe_any_cf", "protected_rows", "mailing_lists", "urls", "directory",
    "critical_example", "list_meta_keys",
]


def _fetch(rel: str):
    dst = CAPE_DIR / rel
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{rel}"
    print(f"downloading {url}")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def load_scenarios(sets=SCENARIO_SETS) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(_fetch(f"scenarios/{s}.parquet")) for s in sets], ignore_index=True)


def load_released_episodes(runs=RELEASED_EPISODE_RUNS) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(_fetch(f"episodes/{r}.parquet")) for r in runs], ignore_index=True)


def spec_from_row(row) -> dict:
    """Turn a scenarios-config row (all strings) into a spec dict with JSON fields parsed."""
    spec = {}
    items = row._asdict().items() if hasattr(row, "_asdict") else dict(row).items()
    for k, v in items:
        if k in JSON_FIELDS and isinstance(v, str) and v not in ("", "nan"):
            try:
                spec[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        if isinstance(v, float) and v != v:  # NaN
            v = None
        spec[k] = v
    for k in ("nuisance", "monitor_aware", "cf_affects_critical"):
        if isinstance(spec.get(k), str):
            spec[k] = spec[k] == "true"
    return spec
