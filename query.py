"""
query.py - answers "where did that car go?"
Simple English: you give a vehicle ID number, it tells you when it appeared,
how long it stayed, what type it was, and shows its frame positions.

Examples:
  python query.py --id 3
  python query.py --class car
  python query.py --list
  python query.py --plate UP15FA7413
  python query.py --plate UP15FA7413 --csv outputs/plate_test.csv
"""

import argparse
import os
import pandas as pd

CSV = "outputs/tracks.csv"


def _norm(s: str) -> str:
    return "".join(c for c in str(s).upper() if c.isalnum())


def main():
    p = argparse.ArgumentParser(description="Query vehicle tracks")
    p.add_argument("--id", type=int, default=None, help="Object ID to look up, e.g. --id 3")
    p.add_argument("--class", dest="vclass", default=None, help="Filter by class, e.g. --class car")
    p.add_argument("--list", action="store_true", help="List all vehicle IDs")
    p.add_argument("--plate", default=None, help="Find vehicle by plate, e.g. --plate UP15FA7413")
    p.add_argument("--csv", default=CSV, help="CSV file to search (default: outputs/tracks.csv)")
    p.add_argument("--fuzzy", action="store_true",
                   help="With --plate: tolerate up to 2-char difference (OCR typos)")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] {args.csv} not found. Run python main.py first.")
        return
    df = pd.read_csv(args.csv)
    if df.empty:
        print("No vehicles found.")
        return

    if args.plate:
        want = _norm(args.plate)
        df["_pn"] = df["plate_number"].fillna("").map(_norm) if "plate_number" in df else ""
        if args.fuzzy:
            def close(a):
                return len(a) == len(want) and sum(1 for x, y in zip(a, want) if x != y) <= 2
            sub = df[df["_pn"].map(lambda a: bool(a) and (a == want or close(a)))]
        else:
            sub = df[df["_pn"] == want]
        ids = sorted(sub["object_id"].unique().tolist()) if not sub.empty else []
        if not ids:
            print(f"No vehicle with plate '{args.plate}'. Try --fuzzy (OCR typos) or another CSV.")
            return
        print(f"Plate '{args.plate}' -> vehicle IDs {ids} (cross-camera identity key)")
        for tid in ids:
            d = sub[sub["object_id"] == tid].sort_values("frame")
            print(f"  #{tid} {d.iloc[0]['vehicle_class']}: "
                  f"{d.iloc[0]['time_s']}s-{d.iloc[-1]['time_s']}s, "
                  f"frames {d.iloc[0]['frame']}-{d.iloc[-1]['frame']}, "
                  f"read as '{d.iloc[-1]['plate_number']}'")
        return

    if args.list or (args.id is None and args.vclass is None and args.plate is None):
        s = df.groupby("object_id").agg(
            vehicle_class=("vehicle_class", "first"),
            first_seen_s=("time_s", "min"), last_seen_s=("time_s", "max"),
            frames=("frame", "count")).reset_index().sort_values("object_id")
        print(s.to_string(index=False))
        print(f"\nTotal unique vehicles: {len(s)}")
        print("Tip: python query.py --id 3")
        return

    if args.vclass:
        sub = df[df["vehicle_class"] == args.vclass]
        ids = sorted(sub["object_id"].unique().tolist())
        print(f"Vehicles of class '{args.vclass}': {ids} (count={len(ids)})")
        return

    sub = df[df["object_id"] == args.id].sort_values("frame")
    if sub.empty:
        print(f"Vehicle #{args.id} not found. Try python query.py --list")
        return
    print(f"=== Vehicle #{args.id} ===")
    print(f"Class      : {sub.iloc[0]['vehicle_class']}")
    print(f"First seen : {sub.iloc[0]['time_s']}s (frame {sub.iloc[0]['frame']})")
    print(f"Last seen  : {sub.iloc[-1]['time_s']}s (frame {sub.iloc[-1]['frame']})")
    print(f"Frames     : {len(sub)}")
    print(f"Duration   : {sub.iloc[-1]['time_s'] - sub.iloc[0]['time_s']:.1f} seconds")
    events = sub[sub["event"].isin(["IN", "OUT"])]
    if not events.empty:
        for _, e in events.iterrows():
            print(f"  -> crossed line {e['event']} at {e['time_s']}s")
    else:
        print("  -> did not cross the counting line (stayed on one side)")
    print(f"Photo      : outputs/crops/vehicle_{args.id}.jpg")
    print("\nFirst 5 positions (x1,y1,x2,y2):")
    print(sub[["frame", "time_s", "x1", "y1", "x2", "y2", "confidence"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
