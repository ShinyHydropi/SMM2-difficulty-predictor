"""
Dataset analysis
    - inventory: every level element that appears (all 133 object IDs, unseen ones included), plus position and area checks
    - characteristics: one row per level of top-level (non-grid) characteristics and the clear-rate target
    - objects, view: single-level inspection (raw objects, ASCII map of the SOLID channel)
    - Tables land in data_processing/output/
    - Run from the repo root (count defaults to 2000, shards to 8, more shards = finer date coverage but slower):
        python -m data_processing.analysis inventory 2000
        python -m data_processing.analysis characteristics 2000 8
        python -m data_processing.analysis objects
        python -m data_processing.analysis view 1
"""

# Imports
import csv
import sys
import zlib
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from pprint import pprint
from .encoding import (
    ALT_FORM_BIT,
    BOUNDARY_UNITS_PER_TILE,
    LEFT_ANCHORED,
    PARACHUTE_BIT,
    RAW_OFFSET,
    TILE,
    UNIDENTIFIED,
    WINGS_BIT,
    NO_CHILD,
    ObjId,
    levelToInput,
    parse
)




# ---------------------------------
# Constants
# ---------------------------------
# Dataset and sampling
DATASET = "TheGreatRambler/mm2_level"
SAMPLE_SHARDS = 8
DEFAULT_COUNT = 2000
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MIN_ATTEMPTS = 100  # Clear rate is too noisy below this

# Pipe direction flags (this module masks 0x60, encoding.py masks 0x7F)
PIPE_DIRECTION_MASK = 0x60
PIPE_DIRECTIONS = {0: "right", 0x20: "left", 0x40: "up", 0x60: "down"}

# Anchor check (which x rule overlaps terrain least)
X_RULES = {
    "center": lambda obj, width: (obj.x - RAW_OFFSET * width) // TILE,
    "left": lambda obj, width: obj.x // TILE
}
SKIP_ANCHOR_CHECK = {ObjId.slight_slope, ObjId.steep_slope}  # Slopes fill a triangle, not their box
MIN_ANCHOR_CELLS = 30  # IDs with fewer checked cells are too thin to call
LEFT_ANCHORED_NAMES = {ObjId(value).name for value in LEFT_ANCHORED}

# Dataset columns (leakage columns come from other players' results)
INPUT_COLUMNS = [
    "gamestyle", "theme", "tag1", "tag2", "timer", "autoscroll_speed", "clear_condition",
    "clear_condition_magnitude"
]
CREATOR_COLUMNS = ["upload_time", "upload_attempts"]  # Creator's own clear check (team decision)
LEAKAGE_COLUMNS = [
    "clears", "attempts", "clear_rate", "difficulty", "world_record", "plays", "likes", "boos",
    "num_comments", "unique_players_and_versus", "versus_matches", "coop_matches", "weekly_likes",
    "weekly_plays", "first_completer_pid", "record_holder_pid"
]
HEADER_FIELDS = ["start_y", "goal_x", "goal_y", "clear_condition_category"]
AREA_FIELDS = [
    "orientation", "theme", "autoscroll_type", "liquid_mode", "liquid_speed", "liquid_start_height",
    "liquid_end_height", "object_count", "ground_count"
]




# ---------------------------------
# enumName
# ---------------------------------
def enumName(value):
    """
    Name of a Kaitai enum member, or the raw value when Kaitai returned a plain int

    Args:
        value: Enum member or int
    """
    return value.name if hasattr(value, "name") else value




# ---------------------------------
# sampleRows
# ---------------------------------
def sampleRows(count, shards=SAMPLE_SHARDS):
    """
    Sample a streaming dataset evenly across its shards

    Args:
        count: Total rows to yield
        shards: Number of shards to draw from

    Raises:
        ValueError: If count is negative or shards isn't positive
    """
    from datasets import load_dataset

    if count < 0 or shards < 1:
        raise ValueError("count must be nonnegative and shards must be positive")

    dataset = load_dataset(DATASET, streaming=True, split="train")
    for shardIndex in range(min(shards, count)):
        shardCount = (count - shardIndex + shards - 1) // shards
        yield from islice(dataset.shard(num_shards=shards, index=shardIndex), shardCount)




# ---------------------------------
# Area and array accessors
# ---------------------------------
def areaSize(areaMap):
    """
    Area width and height in tiles

    Args:
        areaMap: level.overworld or level.subworld
    """
    return areaMap.boundary_right // BOUNDARY_UNITS_PER_TILE, areaMap.boundary_top // BOUNDARY_UNITS_PER_TILE


def usedAreas(level):
    """
    The areas a level actually uses, as (name, areaMap) pairs

    Args:
        level: Parsed Level

    Returns:
        Overworld, plus the subworld when it has a nonzero size
    """
    areas = [("overworld", level.overworld)]
    if level.subworld is not None and all(size > 0 for size in areaSize(level.subworld)):
        areas.append(("subworld", level.subworld))

    return areas


def isNight(areaMap):
    """
    True if the area is in night mode

    Args:
        areaMap: level.overworld or level.subworld
    """
    return areaMap.unk_flag == 1


def objects(areaMap):
    """Objects actually in use (the array is fixed-size)"""
    return areaMap.objects[:areaMap.object_count]


def ground(areaMap):
    """Ground tiles actually in use"""
    return areaMap.ground[:areaMap.ground_count]


def tracks(areaMap):
    """Track pieces actually in use"""
    return areaMap.tracks[:areaMap.track_count]


def icicles(areaMap):
    """Icicles actually in use"""
    return areaMap.icicles[:areaMap.ice_count]


def clearPipes(areaMap):
    """Clear pipes actually in use"""
    return areaMap.clear_pipes[:areaMap.clear_pipe_count]




# ---------------------------------
# footprint
# ---------------------------------
def footprint(obj):
    """
    Tile cells covered by an object's rectangular footprint

    Args:
        obj: A parsed object

    Returns:
        Set of (column, row) cells
    """
    width, height = max(obj.width, 1), max(obj.height, 1)
    left = obj.x // TILE if obj.id in LEFT_ANCHORED else (obj.x - RAW_OFFSET * width) // TILE
    bottom = obj.y // TILE
    return {(left + dx, bottom + dy) for dx in range(width) for dy in range(height)}




# ---------------------------------
# IdStats
# ---------------------------------
class IdStats:
    """Running counts for one object ID across the sample"""

    def __init__(self):
        self.count = 0
        self.levels = set()
        self.styles = Counter()
        self.themes = Counter()
        self.sizes = Counter()
        self.oddY = Counter()  # y % 160 values other than 80
        self.altForm = 0
        self.airborne = 0
        self.withChild = 0
        self.anchorCells = 0  # Footprint cells checked against ground (objects wider than one tile)
        self.anchorOverlaps = Counter()  # X_RULES name to cells landing on ground




# ---------------------------------
# inventory
# ---------------------------------
def inventory(rows):
    """
    Inventory every level element in a batch of dataset rows
        - Prints the summary and writes output/inventory_objects.csv
        - Saves one .bcd per unidentified or unknown ID to output/ for the level viewer

    Args:
        rows: Iterable of dataset rows
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    stats = defaultdict(IdStats)
    pipeCells, pipeOverlaps = Counter(), Counter()  # Pipe direction to footprint cells, cells on ground
    arrays, orientations, sizes = Counter(), Counter(), defaultdict(Counter)
    groundOutside = groundTotal = halfTileY = 0
    dataIds, saved = [], set()

    for row in rows:
        level = parse(row)
        dataIds.append(row["data_id"])
        style = enumName(level.gamestyle)

        for areaName, areaMap in usedAreas(level):
            orientation = enumName(areaMap.orientation)
            orientations[f"{areaName} {orientation}"] += 1
            width, height = areaSize(areaMap)
            sizes[orientation][(width, height)] += 1

            groundCells = {(tile.x, tile.y) for tile in ground(areaMap)}
            groundTotal += len(groundCells)
            groundOutside += sum(1 for x, y in groundCells if x >= width or y >= height)
            arrays["ground tiles"] += len(groundCells)
            arrays["track pieces"] += len(tracks(areaMap))
            arrays["icicles"] += len(icicles(areaMap))
            arrays["clear pipes"] += len(clearPipes(areaMap))
            arrays["snake blocks"] += areaMap.snake_block_count
            arrays["! block paths"] += areaMap.exclamation_mark_block_count
            arrays["track block paths"] += areaMap.track_block_count
            arrays["piranha creeper paths"] += areaMap.piranha_creeper_count
            arrays["sound effects"] += areaMap.sound_effect_count

            for obj in objects(areaMap):
                name = enumName(obj.id)
                entry = stats[name]
                entry.count += 1
                entry.levels.add(row["data_id"])
                entry.styles[style] += 1
                entry.themes[enumName(areaMap.theme)] += 1
                entry.sizes[(obj.width, obj.height)] += 1
                entry.altForm += bool(obj.flag & ALT_FORM_BIT)
                entry.airborne += bool(obj.flag & (WINGS_BIT | PARACHUTE_BIT))
                entry.withChild += obj.cid != NO_CHILD
                if obj.y % TILE != RAW_OFFSET:
                    entry.oddY[obj.y % TILE] += 1
                    halfTileY += 1

                # Objects can't overlap terrain in the editor (so the right x rule overlaps it least)
                if obj.id == ObjId.pipe:
                    direction = PIPE_DIRECTIONS[obj.flag & PIPE_DIRECTION_MASK]
                    cells = footprint(obj)
                    pipeCells[direction] += len(cells)
                    pipeOverlaps[direction] += sum(cell in groundCells for cell in cells)
                elif obj.lid == -1 and obj.id not in SKIP_ANCHOR_CHECK:
                    _checkAnchor(obj, entry, groundCells)

                unclear = obj.id in UNIDENTIFIED or not isinstance(obj.id, ObjId)
                if unclear and name not in saved:
                    saved.add(name)
                    with open(OUTPUT_DIR / f"unidentified_{name}_{row['data_id']}.bcd", "wb") as file:
                        file.write(zlib.decompress(row["level_data"]))

    _writeInventoryCsv(stats)

    seen = [objId.name for objId in ObjId if stats[objId.name].count]
    print(f"Levels: {len(dataIds)} (data_id {min(dataIds)} to {max(dataIds)})")
    print("Areas by orientation:", dict(orientations))
    for orientation, counter in sizes.items():
        print(f"Area sizes in tiles, {orientation}: largest {max(counter)}, most common {counter.most_common(3)}")
    print(f"Ground tiles outside their area's boundary: {groundOutside} of {groundTotal}")
    print("Secondary arrays:", dict(arrays))
    print(f"Objects at a half-tile y offset: {halfTileY}")

    print(f"\nx rule per ID, objects wider than one tile (cells on ground; IDs with {MIN_ANCHOR_CELLS}+ cells):")
    print(f"   {'ID':24s} {'cells':>7s} {'center':>7s} {'left':>7s}  footprint() uses")
    checked = {name: idStats for name, idStats in stats.items() if idStats.anchorCells >= MIN_ANCHOR_CELLS}
    for name, idStats in sorted(checked.items(), key=lambda item: -item[1].anchorCells):
        uses = "left" if name in LEFT_ANCHORED_NAMES else "center"
        better = min(X_RULES, key=lambda rule: idStats.anchorOverlaps[rule])
        flag = "" if idStats.anchorOverlaps[better] == idStats.anchorOverlaps[uses] else f"   <-- {better} fits better"
        center, left = idStats.anchorOverlaps["center"], idStats.anchorOverlaps["left"]
        print(f"   {name:24s} {idStats.anchorCells:7d} {center:7d} {left:7d}  {uses}{flag}")

    print(
        "Pipe cells on ground under footprint(), by direction:",
        {direction: f"{pipeOverlaps[direction]} of {pipeCells[direction]}" for direction in pipeCells}
    )

    print(f"\nObject IDs seen: {len(seen)} of {len(ObjId)}")
    print("Never seen:", [objId.name for objId in ObjId if not stats[objId.name].count])
    print("Unidentified and unknown IDs:")
    for name in [objId.name for objId in UNIDENTIFIED] + [key for key in stats if isinstance(key, int)]:
        idStats = stats[name]
        print(f"   {name}: {idStats.count} in {len(idStats.levels)} levels, styles {dict(idStats.styles)}, sizes {dict(idStats.sizes)}")
    print(f"Saved for the level viewer: {sorted(map(str, saved))}")




# ---------------------------------
# _checkAnchor
# ---------------------------------
def _checkAnchor(obj, entry, groundCells):
    """
    Count how many of an object's cells land on ground under each x rule

    Args:
        obj: A parsed object
        entry: IdStats to add the counts to
        groundCells: Set of (x, y) ground tiles in the same area
    """
    width, height = obj.width, obj.height
    if width == 1:
        return  # Both rules agree for one-tile-wide objects

    bottom = obj.y // TILE
    entry.anchorCells += width * height
    for rule, column in X_RULES.items():
        left = column(obj, width)
        overlaps = sum((left + dx, bottom + dy) in groundCells for dx in range(width) for dy in range(height))
        entry.anchorOverlaps[rule] += overlaps




# ---------------------------------
# _writeInventoryCsv
# ---------------------------------
def _writeInventoryCsv(stats):
    """
    Write one row per object ID, all 133 first, then any unknown raw IDs

    Args:
        stats: Dict of ID name to IdStats
    """
    names = [objId.name for objId in ObjId] + sorted(key for key in stats if isinstance(key, int))

    with open(OUTPUT_DIR / "inventory_objects.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "id", "name", "count", "levels", "styles", "themes", "top_sizes", "alt_form_share",
            "airborne_share", "with_child_share", "odd_y_offsets", "anchor_cells",
            "anchor_overlap_center", "anchor_overlap_left"
        ])
        for name in names:
            idStats = stats[name]
            objId = ObjId[name].value if isinstance(name, str) else name
            shares = [
                round(total / idStats.count, 3) if idStats.count else ""
                for total in (idStats.altForm, idStats.airborne, idStats.withChild)
            ]
            writer.writerow([
                objId, name, idStats.count, len(idStats.levels), dict(idStats.styles), dict(idStats.themes),
                dict(idStats.sizes.most_common(3)), *shares, dict(idStats.oddY.most_common(3)), idStats.anchorCells,
                idStats.anchorOverlaps["center"], idStats.anchorOverlaps["left"]
            ])

    print(f"Wrote {OUTPUT_DIR / 'inventory_objects.csv'}")




# ---------------------------------
# characteristics
# ---------------------------------
def characteristics(rows):
    """
    Write top-level characteristics for every qualifying row to output/level_characteristics.csv
        - Target is clears / attempts, kept only for levels with more than MIN_ATTEMPTS attempts
        - Leakage columns are refused if they ever reach the output

    Args:
        rows: Iterable of dataset rows

    Raises:
        ValueError: If a leakage column ends up in the output (guards future edits)
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    written = skipped = 0
    writer = None

    with open(OUTPUT_DIR / "level_characteristics.csv", "w", newline="") as file:
        for row in rows:
            if row["attempts"] <= MIN_ATTEMPTS:
                skipped += 1
                continue

            record = _levelRecord(row)
            if writer is None:
                leaked = set(record) & set(LEAKAGE_COLUMNS)
                if leaked:
                    raise ValueError(f"Leakage columns in output: {sorted(leaked)}")
                writer = csv.DictWriter(file, fieldnames=list(record))
                writer.writeheader()

            writer.writerow(record)
            written += 1

    print(
        f"Wrote {written} levels to {OUTPUT_DIR / 'level_characteristics.csv'} "
        f"(skipped {skipped} with {MIN_ATTEMPTS} or fewer attempts)"
    )




# ---------------------------------
# _levelRecord
# ---------------------------------
def _levelRecord(row):
    """
    One level's top-level characteristics as a flat dict

    Args:
        row: A dataset row

    Returns:
        Dict of column name to value (enums decoded to names)
    """
    level = parse(row)
    record = {"data_id": row["data_id"], "target_clear_rate": row["clears"] / row["attempts"]}
    record.update({column: row[column] for column in INPUT_COLUMNS + CREATOR_COLUMNS})
    record.update({field: enumName(getattr(level, field)) for field in HEADER_FIELDS})
    record["has_subworld"] = len(usedAreas(level)) == 2

    for areaName, areaMap in (("overworld", level.overworld), ("subworld", level.subworld)):
        for field in AREA_FIELDS:
            record[f"{areaName}_{field}"] = enumName(getattr(areaMap, field))
        record[f"{areaName}_width"], record[f"{areaName}_height"] = areaSize(areaMap)
        record[f"{areaName}_night"] = isNight(areaMap)

    return record




# ---------------------------------
# showObjects
# ---------------------------------
def showObjects():
    """Print every overworld object of the first streamed level (raw units, 160 per tile, offset 80 to the tile center)"""
    level = parse(next(iter(sampleRows(1))))
    tileXs, tileYs = set(), set()
    for obj in objects(level.overworld):
        pprint(vars(obj))
        print("\n")
        tileXs.add((obj.x + RAW_OFFSET) / TILE)
        tileYs.add((obj.y + RAW_OFFSET) / TILE)

    print(sorted(tileXs))
    print(sorted(tileYs))




# ---------------------------------
# showMap
# ---------------------------------
def showMap(index=1):
    """
    Print the overworld SOLID channel of the Nth streamed level and save it as level<N>.bcd

    Args:
        index: 1-based position in the unshuffled stream
    """
    from datasets import load_dataset

    row = next(islice(iter(load_dataset(DATASET, streaming=True, split="train")), index - 1, None))
    solid = levelToInput(parse(row))["ow"][0]  # Overworld grid, channel 0 is SOLID
    for gridRow in reversed(solid):
        print("".join("██" if tile > 0 else "  " for tile in gridRow))

    with open(f"level{index}.bcd", "wb") as file:
        file.write(zlib.decompress(row["level_data"]))




# ---------------------------------
# Command line
# ---------------------------------
ROW_COMMANDS = {"inventory": inventory, "characteristics": characteristics}
COMMANDS = {**ROW_COMMANDS, "objects": showObjects, "view": showMap}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"Usage: python -m data_processing.analysis {{{'|'.join(COMMANDS)}}} [count] [shards]")

    command = sys.argv[1]
    if command in ROW_COMMANDS:
        count = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_COUNT
        shards = int(sys.argv[3]) if len(sys.argv) > 3 else SAMPLE_SHARDS
        COMMANDS[command](sampleRows(count, shards))
    elif command == "view":
        showMap(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    else:
        COMMANDS[command]()
