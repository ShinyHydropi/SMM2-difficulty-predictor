from datasets import load_dataset
from kaitaistruct import KaitaiStream
from io import BytesIO
from level import Level
import zlib
from data_processing.level_tensor import level_to_input


ds = iter(load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train"))
i = 1
row = None
level_data = None
for j in range(i):
    row = next(ds)
    level_data = row["level_data"]
    if j == i:
        level = Level(KaitaiStream(BytesIO(zlib.decompress(level_data))))
        bricks = level_to_input(level)[0][0, :, :]
        for r in reversed(bricks):
            for tile in r:
                if tile == 0:
                    print("  ", end="")
                if tile > 0:
                    print("██", end="")
            print()

        with open(f"level{i}.bcd", "wb") as f:
            f.write(zlib.decompress(row["level_data"]))
