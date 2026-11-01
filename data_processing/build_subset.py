"""
Build a train/val/test Parquet subset of the Mario Maker 2 level dataset
    - Splits come from hashing each level's data_id (deterministic, same level always lands in the same split)
    - Rows are filtered on attempts before the seeded stream shuffle and the --max-levels cutoff (saved subset is a random sample)
    - clear_rate is stored as clears / attempts (0 to 1), not the dataset's percent column
    - Levels with a clear condition the encoder doesn't know are dropped (about 8% of levels, encoding.levelToInput can't read them)
    - Edit KEEP_COLUMNS to keep other fields
    - Run it from the repo root with:
        python -m data_processing.build_subset --max-levels 100000 --min-attempts 500 --out-dir mm2_subset
    - Reload later with:
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files={
            "train": "mm2_subset/train.parquet",
            "val": "mm2_subset/val.parquet",
            "test": "mm2_subset/test.parquet"
        })
"""

# Imports
import argparse
import hashlib
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from level import Level
from .encoding import parse




# ---------------------------------
# Constants
# ---------------------------------
DATASET = "TheGreatRambler/mm2_level"
KEEP_COLUMNS = ["data_id", "clear_rate", "attempts", "clears", "gamestyle", "level_data"]  # level_data is the raw bytes (the heavy part)
SPLITS = ("train", "val", "test")
PROGRESS_EVERY = 10_000  # Rows between progress prints




# ---------------------------------
# hasKnownClearCondition
# ---------------------------------
def hasKnownClearCondition(row):
    """
    True if the level's clear condition is one the encoder can one-hot (Kaitai returns a plain int for unknown ones)

    Args:
        row: Dataset row with level_data
    """
    return isinstance(parse(row).clear_condition, Level.ClearCondition)




# ---------------------------------
# assignSplit
# ---------------------------------
def assignSplit(dataId, seed, trainFraction, valFraction):
    """
    Deterministically map a level to train, val, or test with a stable hash

    Args:
        dataId: The level's data_id
        seed: Hash salt (changing it reshuffles every split)
        trainFraction: Share of levels that go to train
        valFraction: Share of levels that go to val (test gets the rest)
    """
    digest = hashlib.md5(f"{seed}-{dataId}".encode()).hexdigest()
    uniform = int(digest[:8], 16) / 0xFFFFFFFF  # Uniform in [0, 1]
    if uniform < trainFraction:
        return "train"
    if uniform < trainFraction + valFraction:
        return "val"

    return "test"




# ---------------------------------
# SplitWriter
# ---------------------------------
class SplitWriter:
    """Buffers rows and appends them to a Parquet file in chunks"""

    def __init__(self, path, chunkSize):
        self.path = path
        self.chunkSize = chunkSize
        self.buffer = []
        self.writer = None
        self.count = 0

    def add(self, row):
        """Buffer one row, flushing to disk once the chunk is full"""
        self.buffer.append(row)
        self.count += 1
        if len(self.buffer) >= self.chunkSize:
            self.flush()

    def flush(self):
        """Write the buffered rows (the first flush fixes the file's schema)"""
        if not self.buffer:
            return

        table = pa.Table.from_pylist(self.buffer)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)
        self.buffer = []

    def close(self):
        """Flush what's left and close the file"""
        self.flush()
        if self.writer is not None:
            self.writer.close()




# ---------------------------------
# main
# ---------------------------------
def main():
    """
    Stream the dataset, keep the qualifying levels, and write the splits and metadata.json

    Raises:
        ValueError: If train and val fractions don't leave room for test
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="mm2_subset")
    parser.add_argument("--max-levels", type=int, default=100_000, help="Total levels to keep across all splits")
    parser.add_argument("--min-attempts", type=int, default=500, help="Drop levels with fewer attempts (clear rate is noisy)")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=5_000,
        help="Streaming shuffle buffer size (larger is more random but uses more RAM and starts slower)"
    )
    parser.add_argument("--chunk-size", type=int, default=2_000, help="Rows buffered in memory before writing to disk")
    args = parser.parse_args()

    if not 0 < args.train_frac + args.val_frac < 1:
        raise ValueError("train-frac + val-frac must be between 0 and 1")

    os.makedirs(args.out_dir, exist_ok=True)
    minAttempts = max(args.min_attempts, 1)  # Clear rate divides by attempts

    dataset = load_dataset(DATASET, split="train", streaming=True)

    # Filtering before the shuffle (the buffer then holds only qualifying rows, not the ~96% that get dropped)
    dataset = dataset.filter(
        lambda row: row["attempts"] is not None and row["attempts"] >= minAttempts
        and row["clears"] is not None and row["level_data"] is not None
    )

    # Shuffling before the max-levels cutoff (the kept levels are a random sample, not the first ones)
    # Streaming shuffle is approximate (samples from a buffer and shuffles shard order)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    writers = {name: SplitWriter(os.path.join(args.out_dir, f"{name}.parquet"), args.chunk_size) for name in SPLITS}

    kept = dropped = 0
    try:
        for row in dataset:
            if not hasKnownClearCondition(row):
                dropped += 1
                continue

            row["clear_rate"] = row["clears"] / row["attempts"]  # Replacing the dataset's percent column
            split = assignSplit(row["data_id"], args.seed, args.train_frac, args.val_frac)
            writers[split].add({column: row.get(column) for column in KEEP_COLUMNS})

            kept += 1
            if kept % PROGRESS_EVERY == 0:
                print(f"kept {kept:,}", flush=True)
            if kept >= args.max_levels:
                break
    finally:
        for writer in writers.values():
            writer.close()

    counts = {name: writer.count for name, writer in writers.items()}
    metadata = {
        "dataset": DATASET,
        "kept": kept,
        "dropped_unknown_clear_condition": dropped,
        "counts": counts,
        "min_attempts": minAttempts,
        "fractions": {"train": args.train_frac, "val": args.val_frac, "test": 1 - args.train_frac - args.val_frac},
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "columns": KEEP_COLUMNS
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nDone. Kept {kept:,} levels (dropped {dropped:,} with an unknown clear condition).")
    print(f"Split sizes: {counts}")
    print(f"Saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
