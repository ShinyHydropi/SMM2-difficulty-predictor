"""
Converts an extracted level from the dataset into a .bcd file
    - Upload level.bcd to https://tgrcode.com/level_viewer to view the level
    - .bcd is the game's raw level format (the dataset stores it zlib compressed)
"""


# Imports
import zlib
from datasets import load_dataset


"""
Writes one level from the dataset to level.bcd in the current directory
    - streaming=True avoids downloading the full dataset
    - level_data is zlib compressed, so it is decompressed before writing
    - "wb" is needed since the .bcd file is raw binary
"""
ds = load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train")
row = next(iter(ds))
with open("level.bcd", "wb") as f:
    f.write(zlib.decompress(row["level_data"]))
