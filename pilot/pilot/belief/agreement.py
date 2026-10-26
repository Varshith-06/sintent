"""Agreement between the blind hand labels and the automatic belief check.

Fill the last column of results/<run>/hand_label_sheet.csv with yes / no / unclear, then:
    python -m pilot.belief.agreement --run pilot_main

x-F items: the hand label "yes" should match class "unknowing", "no" any other class.
x- items had no forged approval, so the expected hand label is "no" (a sanity check on the labeler).
Writes results/<run>/belief_agreement.json.
"""
from __future__ import annotations

import argparse
import csv
import json

from ..config import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    d = RESULTS_DIR / args.run
    key = json.loads((d / "hand_label_key.json").read_text())
    auto = json.loads((d / "belief_check.json").read_text())
    with open(d / "hand_label_sheet.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    labels = {r[0]: r[2].strip().lower() for r in rows}
    missing = [i for i, v in labels.items() if v not in ("yes", "no", "unclear")]
    if missing:
        raise SystemExit(f"items without a yes/no/unclear label: {', '.join(missing)}")

    xf, xm, items = [], [], []
    for i, eid in key.items():
        hand = labels[str(i)]
        if "|x-F|" in eid:
            cls = auto.get(eid, {}).get("class", "missing")
            items.append({"item": i, "episode_id": eid, "hand": hand, "auto": cls})
            if hand != "unclear":
                xf.append((hand == "yes") == (cls == "unknowing"))
        else:
            items.append({"item": i, "episode_id": eid, "hand": hand, "auto": "x- (no forgery)"})
            if hand != "unclear":
                xm.append(hand == "no")
    out = {"x-F agreement": sum(xf) / len(xf) if xf else None, "x-F labeled": len(xf),
           "x- labeled 'no' (sanity)": sum(xm) / len(xm) if xm else None, "x- labeled": len(xm),
           "unclear": sum(v == "unclear" for v in labels.values()), "items": items}
    (d / "belief_agreement.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "items"})


if __name__ == "__main__":
    main()
