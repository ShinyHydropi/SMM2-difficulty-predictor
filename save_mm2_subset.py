"""
Save a subset of the Mario Maker 2 level dataset as train/val/test Parquet files.

Usage:
    python save_mm2_subset.py --max-levels 100000 --min-attempts 100 --out-dir mm2_subset

Reload later with:
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={
        "train": "mm2_subset/train.parquet",
        "val":   "mm2_subset/val.parquet",
        "test":  "mm2_subset/test.parquet",
    })

Notes:
- Splits are assigned by hashing each level's data_id, so they are
  deterministic and reproducible (same level always lands in the same split),
  even though the data is streamed and never fully in memory.
- The stream is shuffled (seeded) before filtering and before the
  --max-levels cutoff, so the saved subset is a random sample.
- Verify the column names below against the dataset card; edit KEEP_COLUMNS
  if you need other fields.
"""

import argparse
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

DATASET = "TheGreatRambler/mm2_level"

# Columns written to disk. level_data is the raw level bytes (the heavy part).
KEEP_COLUMNS = ["data_id", "clear_rate", "attempts", "clears", "gamestyle", "level_data"]


def assign_split(data_id, seed, train_frac, val_frac):
    """Deterministically map a level to train/val/test using a stable hash."""
    h = hashlib.md5(f"{seed}-{data_id}".encode()).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF  # uniform in [0, 1]
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


class SplitWriter:
    """Buffers rows and appends them to a Parquet file in chunks."""

    def __init__(self, path, chunk_size):
        self.path = path
        self.chunk_size = chunk_size
        self.buffer = []
        self.writer = None
        self.count = 0

    def add(self, row):
        self.buffer.append(row)
        self.count += 1
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)
        self.buffer = []

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="mm2_subset")
    p.add_argument("--max-levels", type=int, default=100_000,
                   help="Total levels to keep across all splits.")
    p.add_argument("--min-attempts", type=int, default=500,
                   help="Drop levels with fewer attempts (clear rate is noisy).")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-buffer", type=int, default=50_000,
                   help="Streaming shuffle buffer size. Larger = more random "
                        "but more RAM and a slower start.")
    p.add_argument("--chunk-size", type=int, default=2_000,
                   help="Rows buffered in memory before writing to disk.")
    args = p.parse_args()

    assert 0 < args.train_frac + args.val_frac < 1, "train+val fractions must be < 1"
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(
        DATASET,
        split="train",
        streaming=True,
        trust_remote_code=True,  # the dataset uses a loading script
    )

    # Shuffle BEFORE any filtering or limiting, so the levels we keep are a
    # random sample rather than just the first ones in the dataset's order.
    # Streaming shuffle is approximate: it samples from a buffer, and also
    # shuffles the order of the underlying shards.
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    writers = {
        name: SplitWriter(os.path.join(args.out_dir, f"{name}.parquet"), args.chunk_size)
        for name in ("train", "val", "test")
    }

    kept, seen = 0, 0
    try:
        for row in ds:
            seen += 1
            if seen % 10_000 == 0:
                print(f"scanned {seen:,} | kept {kept:,}", flush=True)

            attempts = row.get("attempts")
            clears = row.get("clears")
            if attempts is None or attempts < args.min_attempts:
                continue

            # Use the provided clear_rate if present, otherwise compute it.
            clear_rate = row.get("clear_rate")
            if clear_rate is None and clears is not None:
                clear_rate = clears / attempts
            if clear_rate is None or row.get("level_data") is None:
                continue
            row["clear_rate"] = clear_rate

            split = assign_split(row["data_id"], args.seed, args.train_frac, args.val_frac)
            writers[split].add({k: row.get(k) for k in KEEP_COLUMNS})

            kept += 1
            if kept >= args.max_levels:
                break
    finally:
        for w in writers.values():
            w.close()

    counts = {name: w.count for name, w in writers.items()}
    meta = {
        "dataset": DATASET,
        "scanned": seen,
        "kept": kept,
        "counts": counts,
        "min_attempts": args.min_attempts,
        "fractions": {"train": args.train_frac, "val": args.val_frac,
                      "test": 1 - args.train_frac - args.val_frac},
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "columns": KEEP_COLUMNS,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Scanned {seen:,} levels, kept {kept:,}.")
    print(f"Split sizes: {counts}")
    print(f"Saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()