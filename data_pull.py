"""
Inspects the parsed object data of a level from the dataset
    - Parses level_data using the Kaitai Struct classes in level.py
    - Object x and y values are raw units rather than tile indices
"""


# Imports
from pprint import pprint
from datasets import load_dataset
from kaitaistruct import KaitaiStream
from io import BytesIO
from level import Level
import zlib


"""
Streams one level from the dataset and parses it
    - streaming=True avoids downloading the full dataset
    - level_data is zlib compressed, so it is decompressed before parsing
"""
ds = load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train")
level_data = next(iter(ds))["level_data"]
level = Level(KaitaiStream(BytesIO(zlib.decompress(level_data))))


"""
Prints every object in the level overworld and collects the unique grid positions
    - Used to check how object coordinates map onto the tile grid
    - x_s and y_s hold the unique tile columns and rows that contain objects
    - level.overworld.objects is a fixed size (limitation of Kaitai struct)
    - Must iterate by object_count or null objects will be included
"""
x_s = set()
y_s = set()

for i in range(level.overworld.object_count):
    object = level.overworld.objects[i]

    # Printing level objects as dictionaries
    pprint(vars(object))
    print("\n")

    # Printing basic level data
    # print("X: %d Y: %d ID: %s" % (object.x, object.y, object.id))

    # Converting raw coordinates to tile indices (each tile is 160 pixels wide, offset by 80 to its center)
    x_s.add((object.x + 80) / 160)
    y_s.add((object.y + 80) / 160)

# Printing the unique tile positions in order
print(sorted(x_s))
print(sorted(y_s))
