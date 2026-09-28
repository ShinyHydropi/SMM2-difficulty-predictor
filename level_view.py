import zlib
from datasets import load_dataset

ds = load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train")
row = next(iter(ds))
with open("level.bcd", "wb") as f:
    f.write(zlib.decompress(row["level_data"]))