"""Helpers for reading and sampling parsed Super Mario Maker 2 levels."""

from itertools import islice
from io import BytesIO
import zlib

from kaitaistruct import KaitaiStream

from level import Level


TILE = 160
SAMPLE_SHARDS = 8
PIPE_DIRECTION_MASK = 0x60
PIPE_RIGHT, PIPE_LEFT, PIPE_UP, PIPE_DOWN = 0, 0x20, 0x40, 0x60
LEFT_ANCHORED = {
    Level.Obj.ObjId(value) for value in (14, 16, 71, 17, 53, 94)
}


def enum_name(value):
    return value.name if hasattr(value, "name") else value


def parse(row):
    data = zlib.decompress(row["level_data"])
    return Level(KaitaiStream(BytesIO(data)))


def sample_rows(count, shards=SAMPLE_SHARDS):
    """Sample a streaming dataset across its shards."""
    from datasets import load_dataset

    if count < 0 or shards < 1:
        raise ValueError("count must be nonnegative and shards must be positive")
    dataset = load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train")
    for shard_index in range(min(shards, count)):
        shard_count = (count - shard_index + shards - 1) // shards
        yield from islice(dataset.shard(num_shards=shards, index=shard_index), shard_count)


def used_areas(level):
    areas = [("overworld", level.overworld)]
    if level.subworld is not None and all(size > 0 for size in area_size(level.subworld)):
        areas.append(("subworld", level.subworld))
    return areas


def area_size(level_map):
    return level_map.boundary_right // 16, level_map.boundary_top // 16


def is_night(level_map):
    return level_map.unk_flag == 1


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


def footprint(obj):
    """Return the tile cells occupied by an object's rectangular footprint."""
    width, height = max(obj.width, 1), max(obj.height, 1)
    left = obj.x // TILE if obj.id in LEFT_ANCHORED else (obj.x - 80 * width) // TILE
    bottom = obj.y // TILE
    return {(left + dx, bottom + dy) for dx in range(width) for dy in range(height)}
