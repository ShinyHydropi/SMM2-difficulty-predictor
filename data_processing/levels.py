"""
Reading SMM2 levels: sampling the dataset, parsing level_data, and turning raw positions into tile cells
    - parse reads only the real entries of each array, with level.py's field names and enums
    - Positions follow the checks in extract.py's inventory (see the SMM2 Level Encoding Spec)
    - Run anything in data_processing from the repo root as a module: python -m data_processing.extract ...
"""

# Imports
import math
import struct
import time
import zlib
from collections import namedtuple
from itertools import islice
from types import SimpleNamespace
from datasets import load_dataset
from huggingface_hub import HfApi
from level import Level


DATASET = "TheGreatRambler/mm2_level"
SAMPLE_SHARDS = 8  # Parquet shards a sample is spread over (the dataset has 196, ordered by upload date)
TILE = 160  # Raw object units per tile (16 px tile * 10)
PIXELS_PER_TILE = 16  # Area boundaries are in pixels
NIGHT_BIT = 0x2  # Bit of map.unk_flag that turns on night mode (Course Format.md)
PIPE_DIRECTION_MASK = 0x60  # Pipe flag bits for direction (Toost)
PIPE_RIGHT, PIPE_LEFT, PIPE_UP, PIPE_DOWN = 0x00, 0x20, 0x40, 0x60


"""
Sampling
    - Rows are stored oldest first, so the first N rows are all mid-2019 levels missing every later part
    - sample_rows takes the first rows of evenly spaced shards instead, so a sample spans all upload dates
    - Run time is about rows, not shards (~0.2 s a row with the old Kaitai parse, mostly parsing)
"""
def sample_rows(count, shards=SAMPLE_SHARDS):
    """
    Yield about count rows spread evenly over evenly spaced shards, printing progress per shard

    Args:
        count: Total rows to yield
        shards: How many shards to spread them over (more = finer date coverage)

    Returns:
        Generator of dataset rows (dicts keyed by column name)
    """
    files = sorted(f for f in HfApi().list_repo_files(DATASET, repo_type="dataset") if f.startswith("data/train-"))
    shards = min(shards, len(files))
    picked = [files[i * len(files) // shards] for i in range(shards)]
    per_shard = math.ceil(count / shards)
    remaining = count

    for index, path in enumerate(picked, start=1):
        wanted = min(per_shard, remaining)
        print(f"Shard {index}/{shards} ({path.split('/')[-1][:20]}): reading {wanted} rows...", end=" ", flush=True)
        started = time.time()

        shard = load_dataset(DATASET, data_files=path, streaming=True, split="train")
        for row in islice(shard, wanted):
            yield row

        print(f"done in {time.time() - started:.0f}s", flush=True)
        remaining -= wanted
        if remaining <= 0:
            return


"""
Parsing
    - Kaitai's Level also builds every padding entry (2,600 object slots per area), ~0.1-0.2 s a level
    - parse reads only the real entries straight from the bytes, so it does a small fraction of that work
    - Offsets and field order follow Course Format.md and level.py; clear_condition is read unsigned so its enum resolves
"""
HEADER = struct.Struct("<BBhhhhbbbbBBIiiiiIqi")  # start_y through game_version (0x00 to 0x33)
AREA_HEADER = struct.Struct("<8B16i")  # theme through ice_count (0x48 bytes)
OBJECT = struct.Struct("<iihBBiiihhhh")
GROUND = struct.Struct("<4B")
TRACK = struct.Struct("<HBBBBHHH")
ICICLE = struct.Struct("<4B")
CLEAR_PIPE = struct.Struct("<BBH")
CLEAR_PIPE_NODE = struct.Struct("<8B")

HEADER_SIZE, AREA_SIZE = 0x200, 0x2DEE0
GAMESTYLE_OFFSET, NAME_OFFSET, DESCRIPTION_OFFSET = 0xF1, 0xF4, 0x136
OBJECTS_OFFSET, CLEAR_PIPES_OFFSET, GROUND_OFFSET, TRACKS_OFFSET, ICICLES_OFFSET = 0x48, 0x15CCC, 0x247A4, 0x28624, 0x2CC74
CLEAR_PIPE_SIZE = CLEAR_PIPE.size + 36 * CLEAR_PIPE_NODE.size

HEADER_FIELDS = ["start_y", "goal_y", "goal_x", "timer", "clear_condition_magnitude", "year", "month", "day", "hour",
                 "minute", "autoscroll_speed", "clear_condition_category", "clear_condition", "unk_gamever",
                 "unk_management_flags", "clear_attempts", "clear_time", "unk_creation_id", "unk_upload_id",
                 "game_version"]
AREA_FIELDS = ["theme", "autoscroll_type", "boundary_type", "orientation", "liquid_end_height", "liquid_mode",
               "liquid_speed", "liquid_start_height", "boundary_right", "boundary_top", "boundary_left",
               "boundary_bottom", "unk_flag", "object_count", "sound_effect_count", "snake_block_count",
               "clear_pipe_count", "piranha_creeper_count", "exclamation_mark_block_count", "track_block_count", "unk1",
               "ground_count", "track_count", "ice_count"]
HEADER_ENUMS = {"autoscroll_speed": Level.AutoscrollSpeed, "clear_condition_category": Level.ClearConditionCategory,
                "clear_condition": Level.ClearCondition, "game_version": Level.GameVersion}
AREA_ENUMS = {"theme": Level.Map.Theme, "autoscroll_type": Level.Map.AutoscrollType,
              "boundary_type": Level.Map.BoundaryType, "orientation": Level.Map.Orientation,
              "liquid_mode": Level.Map.LiquidMode, "liquid_speed": Level.Map.LiquidSpeed}

Obj = namedtuple("Obj", "x y unk1 width height flag cflag ex id cid lid sid")
Ground = namedtuple("Ground", "x y id background_id")
Track = namedtuple("Track", "unk1 flags x y type lid unk2 unk3")
Icicle = namedtuple("Icicle", "x y type unk1")
ClearPipeNode = namedtuple("ClearPipeNode", "type index x y width height unk1 direction")


def parse(row):
    """
    Parse one dataset row's level_data

    Args:
        row: A dataset row

    Returns:
        A level with the same attributes as level.py's Level (only the real array entries are kept)
    """
    return parse_bytes(zlib.decompress(row["level_data"]))


def parse_bytes(data):
    """
    Parse decompressed level bytes (the contents of a .bcd file)

    Args:
        data: Decompressed level_data

    Returns:
        A level with the same attributes as level.py's Level (only the real array entries are kept)
    """
    level = SimpleNamespace(**_with_enums(HEADER_FIELDS, HEADER.unpack_from(data, 0), HEADER_ENUMS))
    level.gamestyle = _enum(Level.Gamestyle, struct.unpack_from("<h", data, GAMESTYLE_OFFSET)[0])
    level.name = _utf16(data[NAME_OFFSET:NAME_OFFSET + 66])
    level.description = _utf16(data[DESCRIPTION_OFFSET:DESCRIPTION_OFFSET + 202])
    level.overworld = _parse_area(data, HEADER_SIZE)
    level.subworld = _parse_area(data, HEADER_SIZE + AREA_SIZE)
    return level


def _parse_area(data, base):
    """
    Parse one area (overworld or subworld) starting at byte offset base

    Args:
        data: Decompressed level_data
        base: Byte offset of the area header

    Returns:
        A map with level.py's Map attributes and its real objects, ground, tracks, icicles, and clear pipes
    """
    area = SimpleNamespace(**_with_enums(AREA_FIELDS, AREA_HEADER.unpack_from(data, base), AREA_ENUMS))

    area.objects = []
    for values in _records(OBJECT, data, base + OBJECTS_OFFSET, area.object_count):
        obj = Obj(*values)
        area.objects.append(obj._replace(id=_enum(Level.Obj.ObjId, obj.id)))

    area.ground = [Ground(*v) for v in _records(GROUND, data, base + GROUND_OFFSET, area.ground_count)]
    area.tracks = [Track(*v) for v in _records(TRACK, data, base + TRACKS_OFFSET, area.track_count)]
    area.icicles = [Icicle(*v) for v in _records(ICICLE, data, base + ICICLES_OFFSET, area.ice_count)]

    area.clear_pipes = []
    for i in range(area.clear_pipe_count):
        offset = base + CLEAR_PIPES_OFFSET + i * CLEAR_PIPE_SIZE
        index, node_count, unk = CLEAR_PIPE.unpack_from(data, offset)
        nodes = [ClearPipeNode(*v) for v in _records(CLEAR_PIPE_NODE, data, offset + CLEAR_PIPE.size, node_count)]
        area.clear_pipes.append(SimpleNamespace(index=index, node_count=node_count, unk=unk, nodes=nodes))

    return area


def _records(layout, data, offset, count):
    """Unpack count fixed-size records of the given struct layout starting at offset"""
    return [layout.unpack_from(data, offset + i * layout.size) for i in range(count)]


def _with_enums(fields, values, enums):
    """Zip field names to values, turning enum fields into their Level enums (raw int if out of range)"""
    return {field: _enum(enums[field], value) if field in enums else value for field, value in zip(fields, values)}


def _enum(enum_class, value):
    """Enum member for value, or value itself if the enum has no such member (same as Kaitai)"""
    try:
        return enum_class(value)
    except ValueError:
        return value


def _utf16(raw):
    """Decode a null-padded UTF-16LE text field"""
    return raw.decode("utf-16-le").split("\x00")[0]


"""
Areas and their arrays
    - The overworld is always horizontal (1000 of 1000 sampled); subworlds are either
    - Horizontal areas are 27 tiles tall and up to 240 wide; vertical ones are 48 wide and up to 168 tall
"""
def used_areas(level):
    """
    Areas a level actually uses

    Args:
        level: A parsed level

    Returns:
        List of (name, map) pairs (overworld always, subworld only if it has anything in it)
    """
    areas = [("overworld", level.overworld)]
    if has_content(level.subworld):
        areas.append(("subworld", level.subworld))
    return areas


def has_content(level_map):
    """True if the area has any objects, ground tiles, or tracks (an unused subworld is all zeros)"""
    return level_map.object_count + level_map.ground_count + level_map.track_count > 0


def is_night(level_map):
    """True if the area is in night mode"""
    return bool(level_map.unk_flag & NIGHT_BIT)


def area_size(level_map):
    """
    Width and height of an area in tiles, from its boundary (stored in pixels)

    Args:
        level_map: level.overworld or level.subworld

    Returns:
        (width, height) in tiles
    """
    width = (level_map.boundary_right - level_map.boundary_left) // PIXELS_PER_TILE
    height = (level_map.boundary_top - level_map.boundary_bottom) // PIXELS_PER_TILE
    return width, height


def objects(level_map):
    return level_map.objects[:level_map.object_count]


def ground(level_map):
    return level_map.ground[:level_map.ground_count]


def tracks(level_map):
    return level_map.tracks[:level_map.track_count]


def icicles(level_map):
    return level_map.icicles[:level_map.ice_count]


def clear_pipes(level_map):
    return level_map.clear_pipes[:level_map.clear_pipe_count]


def enum_name(value):
    """Name of an enum value, or the raw int if the value isn't in the enum"""
    return getattr(value, "name", value)


"""
Positions
    - One-tile objects: column = x // 160, row = y // 160 (22 of 35,819 overlapped ground, 800+ for any shift)
    - y is the bottom row (2,000 levels: ~2,600 overlapping cells vs ~14,500 for a centered y)
    - x is the horizontal center for sprites, the left column for LEFT_ANCHORED parts (chosen per ID from the inventory)
    - Pipes grow from their base in their own direction; PIPE_CELLS comes from Toost's pipe drawing code
"""
LEFT_ANCHORED = {
    Level.Obj.ObjId.mushroom_platform, Level.Obj.ObjId.semisolid_platform, Level.Obj.ObjId.bridge,
    Level.Obj.ObjId.fast_conveyor_belt, Level.Obj.ObjId.half_collision_platform, Level.Obj.ObjId.conveyor_belt,
    Level.Obj.ObjId.spike_block
}

# Direction -> (column offsets, row offsets) from (x // 160, y // 160) for a pipe of length n (stored as height)
PIPE_CELLS = {
    PIPE_UP: lambda n: (range(0, 2), range(0, n)),
    PIPE_DOWN: lambda n: (range(-1, 1), range(-n + 1, 1)),
    PIPE_RIGHT: lambda n: (range(0, n), range(-1, 1)),
    PIPE_LEFT: lambda n: (range(-n + 1, 1), range(0, 2))
}


def footprint(obj):
    """
    Tile cells an object covers

    Args:
        obj: A parsed object

    Returns:
        List of (column, row) cells
    """
    if obj.id == Level.Obj.ObjId.pipe:
        columns, rows = PIPE_CELLS[obj.flag & PIPE_DIRECTION_MASK](obj.height)
        column, row = obj.x // TILE, obj.y // TILE
        return [(column + dx, row + dy) for dx in columns for dy in rows]

    left = obj.x // TILE if obj.id in LEFT_ANCHORED else (obj.x - 80 * obj.width) // TILE
    bottom = obj.y // TILE
    return [(left + dx, bottom + dy) for dx in range(obj.width) for dy in range(obj.height)]
