"""
Extracts tables from a sample of the dataset into data_processing/output/
    - inventory: every level element that appears (all 133 object IDs, unseen ones included), plus position and area checks
    - characteristics: one row per level of top-level (non-grid) characteristics and the clear-rate target
    - Run from the repo root (count defaults to 2000, shards to 8; more shards = finer date coverage, slower):
        python -m data_processing.extract inventory 2000
        python -m data_processing.extract characteristics 2000 8
"""

# Imports
import csv
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from .categories import ALT_FORM_BIT, PARACHUTE_BIT, UNIDENTIFIED, WINGS_BIT, ObjId
from .levels import (
    LEFT_ANCHORED, PIPE_DIRECTION_MASK, PIPE_DOWN, PIPE_LEFT, PIPE_RIGHT, PIPE_UP, SAMPLE_SHARDS, TILE, area_size,
    clear_pipes, enum_name, footprint, ground, icicles, is_night, objects, parse, sample_rows, tracks, used_areas
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MIN_ATTEMPTS = 100  # Clear rate is too noisy below this

# x rules compared per object ID for objects wider than one tile (y is the bottom row under both)
X_RULES = {
    "center": lambda obj, width: (obj.x - 80 * width) // TILE,
    "left": lambda obj, width: obj.x // TILE
}
SKIP_ANCHOR_CHECK = {ObjId.slight_slope, ObjId.steep_slope}  # Slopes fill a triangle, not their box
MIN_ANCHOR_CELLS = 30  # IDs with fewer checked cells are too thin to call
LEFT_ANCHORED_NAMES = {obj_id.name for obj_id in LEFT_ANCHORED}
PIPE_DIRECTIONS = {PIPE_RIGHT: "right", PIPE_LEFT: "left", PIPE_UP: "up", PIPE_DOWN: "down"}

# Dataset columns safe to use as inputs (set by the creator at upload)
INPUT_COLUMNS = ["gamestyle", "theme", "tag1", "tag2", "timer", "autoscroll_speed", "clear_condition",
                 "clear_condition_magnitude"]
CREATOR_COLUMNS = ["upload_time", "upload_attempts"]  # Creator's own clear check (team decision)
LEAKAGE_COLUMNS = [
    "clears", "attempts", "clear_rate", "difficulty", "world_record", "plays", "likes", "boos",
    "num_comments", "unique_players_and_versus", "versus_matches", "coop_matches", "weekly_likes",
    "weekly_plays", "first_completer_pid", "record_holder_pid"
]
HEADER_FIELDS = ["start_y", "goal_x", "goal_y", "clear_condition_category"]
AREA_FIELDS = ["orientation", "theme", "autoscroll_type", "liquid_mode", "liquid_speed", "liquid_start_height",
               "liquid_end_height", "object_count", "ground_count"]


class IdStats:
    """Running counts for one object ID across the sample"""

    def __init__(self):
        self.count = 0
        self.levels = set()
        self.styles = Counter()
        self.themes = Counter()
        self.sizes = Counter()
        self.odd_y = Counter()  # y % 160 values other than 80
        self.alt_form = 0
        self.airborne = 0
        self.with_child = 0
        self.anchor_cells = 0  # Footprint cells checked against ground (objects wider than one tile)
        self.anchor_overlaps = Counter()  # X_RULES name -> cells landing on ground


"""
Inventory
    - Prints the summary and writes output/inventory_objects.csv
    - Saves one .bcd per unidentified or unknown ID to output/ for the level viewer
"""
def inventory(rows):
    """
    Inventory every level element in a batch of dataset rows

    Args:
        rows: Iterable of dataset rows
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    stats = defaultdict(IdStats)
    pipe_cells, pipe_overlaps = Counter(), Counter()  # Pipe direction -> footprint cells, cells on ground
    arrays, orientations, sizes = Counter(), Counter(), defaultdict(Counter)
    ground_outside = ground_total = half_tile_y = 0
    data_ids, saved = [], set()

    for row in rows:
        level = parse(row)
        data_ids.append(row["data_id"])
        style = enum_name(level.gamestyle)

        for area_name, level_map in used_areas(level):
            orientation = enum_name(level_map.orientation)
            orientations[f"{area_name} {orientation}"] += 1
            width, height = area_size(level_map)
            sizes[orientation][(width, height)] += 1

            ground_cells = {(tile.x, tile.y) for tile in ground(level_map)}
            ground_total += len(ground_cells)
            ground_outside += sum(1 for x, y in ground_cells if x >= width or y >= height)
            arrays["ground tiles"] += len(ground_cells)
            arrays["track pieces"] += len(tracks(level_map))
            arrays["icicles"] += len(icicles(level_map))
            arrays["clear pipes"] += len(clear_pipes(level_map))
            arrays["snake blocks"] += level_map.snake_block_count
            arrays["! block paths"] += level_map.exclamation_mark_block_count
            arrays["track block paths"] += level_map.track_block_count
            arrays["piranha creeper paths"] += level_map.piranha_creeper_count
            arrays["sound effects"] += level_map.sound_effect_count

            for obj in objects(level_map):
                name = enum_name(obj.id)
                entry = stats[name]
                entry.count += 1
                entry.levels.add(row["data_id"])
                entry.styles[style] += 1
                entry.themes[enum_name(level_map.theme)] += 1
                entry.sizes[(obj.width, obj.height)] += 1
                entry.alt_form += bool(obj.flag & ALT_FORM_BIT)
                entry.airborne += bool(obj.flag & (WINGS_BIT | PARACHUTE_BIT))
                entry.with_child += obj.cid != -1
                if obj.y % TILE != 80:
                    entry.odd_y[obj.y % TILE] += 1
                    half_tile_y += 1

                # Objects can't overlap terrain in the editor, so the right x rule overlaps it least
                if obj.id == ObjId.pipe:
                    direction = PIPE_DIRECTIONS[obj.flag & PIPE_DIRECTION_MASK]
                    cells = footprint(obj)
                    pipe_cells[direction] += len(cells)
                    pipe_overlaps[direction] += sum(cell in ground_cells for cell in cells)
                elif obj.lid == -1 and obj.id not in SKIP_ANCHOR_CHECK:
                    _check_anchor(obj, entry, ground_cells)

                unclear = obj.id in UNIDENTIFIED or not isinstance(obj.id, ObjId)
                if unclear and name not in saved:
                    saved.add(name)
                    with open(OUTPUT_DIR / f"unidentified_{name}_{row['data_id']}.bcd", "wb") as f:
                        f.write(zlib.decompress(row["level_data"]))

    _write_inventory_csv(stats)

    seen = [obj_id.name for obj_id in ObjId if stats[obj_id.name].count]
    print(f"Levels: {len(data_ids)} (data_id {min(data_ids)} to {max(data_ids)})")
    print("Areas by orientation:", dict(orientations))
    for orientation, counter in sizes.items():
        print(f"Area sizes in tiles, {orientation}: largest {max(counter)}, most common {counter.most_common(3)}")
    print(f"Ground tiles outside their area's boundary: {ground_outside} of {ground_total}")
    print("Secondary arrays:", dict(arrays))
    print(f"Objects at a half-tile y offset: {half_tile_y}")

    print(f"\nx rule per ID, objects wider than one tile (cells on ground; IDs with {MIN_ANCHOR_CELLS}+ cells):")
    print(f"   {'ID':24s} {'cells':>7s} {'center':>7s} {'left':>7s}  footprint() uses")
    checked = {name: s for name, s in stats.items() if s.anchor_cells >= MIN_ANCHOR_CELLS}
    for name, s in sorted(checked.items(), key=lambda item: -item[1].anchor_cells):
        uses = "left" if name in LEFT_ANCHORED_NAMES else "center"
        better = min(X_RULES, key=lambda rule: s.anchor_overlaps[rule])
        flag = "" if s.anchor_overlaps[better] == s.anchor_overlaps[uses] else f"   <-- {better} fits better"
        print(f"   {name:24s} {s.anchor_cells:7d} {s.anchor_overlaps['center']:7d} {s.anchor_overlaps['left']:7d}"
              f"  {uses}{flag}")
    print("Pipe cells on ground under footprint(), by direction:",
          {direction: f"{pipe_overlaps[direction]} of {pipe_cells[direction]}" for direction in pipe_cells})

    print(f"\nObject IDs seen: {len(seen)} of {len(ObjId)}")
    print("Never seen:", [obj_id.name for obj_id in ObjId if not stats[obj_id.name].count])
    print("Unidentified and unknown IDs:")
    for name in [obj_id.name for obj_id in UNIDENTIFIED] + [n for n in stats if isinstance(n, int)]:
        s = stats[name]
        print(f"   {name}: {s.count} in {len(s.levels)} levels, styles {dict(s.styles)}, sizes {dict(s.sizes)}")
    print(f"Saved for the level viewer: {sorted(map(str, saved))}")


def _check_anchor(obj, entry, ground_cells):
    """
    Count how many of an object's cells land on ground under each x rule

    Args:
        obj: A parsed object
        entry: IdStats to add the counts to
        ground_cells: Set of (x, y) ground tiles in the same area
    """
    width, height = obj.width, obj.height
    if width == 1:
        return  # Both rules agree for one-tile-wide objects

    bottom = obj.y // TILE
    entry.anchor_cells += width * height
    for rule, column in X_RULES.items():
        left = column(obj, width)
        entry.anchor_overlaps[rule] += sum((left + dx, bottom + dy) in ground_cells
                                           for dx in range(width) for dy in range(height))


def _write_inventory_csv(stats):
    """
    Write one row per object ID, all 133 first, then any unknown raw IDs

    Args:
        stats: Dict of ID name -> IdStats
    """
    names = [obj_id.name for obj_id in ObjId] + sorted(n for n in stats if isinstance(n, int))

    with open(OUTPUT_DIR / "inventory_objects.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "count", "levels", "styles", "themes", "top_sizes", "alt_form_share",
                         "airborne_share", "with_child_share", "odd_y_offsets", "anchor_cells",
                         "anchor_overlap_center", "anchor_overlap_left"])
        for name in names:
            s = stats[name]
            obj_id = ObjId[name].value if isinstance(name, str) else name
            shares = [round(n / s.count, 3) if s.count else "" for n in (s.alt_form, s.airborne, s.with_child)]
            writer.writerow([obj_id, name, s.count, len(s.levels), dict(s.styles), dict(s.themes),
                             dict(s.sizes.most_common(3)), *shares, dict(s.odd_y.most_common(3)), s.anchor_cells,
                             s.anchor_overlaps["center"], s.anchor_overlaps["left"]])
    print(f"Wrote {OUTPUT_DIR / 'inventory_objects.csv'}")


"""
Characteristics
    - Target = clears / attempts, kept only for levels with more than MIN_ATTEMPTS attempts
    - Leakage columns come from other players' results and are refused if they ever reach the output
"""
def characteristics(rows):
    """
    Write top-level characteristics for every qualifying row to output/level_characteristics.csv

    Args:
        rows: Iterable of dataset rows

    Raises:
        ValueError: If a leakage column ends up in the output (guards future edits)
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    written = skipped = 0
    writer = None

    with open(OUTPUT_DIR / "level_characteristics.csv", "w", newline="") as f:
        for row in rows:
            if row["attempts"] <= MIN_ATTEMPTS:
                skipped += 1
                continue

            record = _level_record(row)
            if writer is None:
                leaked = set(record) & set(LEAKAGE_COLUMNS)
                if leaked:
                    raise ValueError(f"Leakage columns in output: {sorted(leaked)}")
                writer = csv.DictWriter(f, fieldnames=list(record))
                writer.writeheader()

            writer.writerow(record)
            written += 1

    print(f"Wrote {written} levels to {OUTPUT_DIR / 'level_characteristics.csv'} "
          f"(skipped {skipped} with {MIN_ATTEMPTS} or fewer attempts)")


def _level_record(row):
    """
    One level's top-level characteristics as a flat dict

    Args:
        row: A dataset row

    Returns:
        Dict of column name -> value (enums decoded to names)
    """
    level = parse(row)
    record = {"data_id": row["data_id"], "target_clear_rate": row["clears"] / row["attempts"]}
    record.update({column: row[column] for column in INPUT_COLUMNS + CREATOR_COLUMNS})
    record.update({field: enum_name(getattr(level, field)) for field in HEADER_FIELDS})
    record["has_subworld"] = len(used_areas(level)) == 2

    for area_name, level_map in (("overworld", level.overworld), ("subworld", level.subworld)):
        for field in AREA_FIELDS:
            record[f"{area_name}_{field}"] = enum_name(getattr(level_map, field))
        record[f"{area_name}_width"], record[f"{area_name}_height"] = area_size(level_map)
        record[f"{area_name}_night"] = is_night(level_map)

    return record


COMMANDS = {"inventory": inventory, "characteristics": characteristics}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"Usage: python -m data_processing.extract {{{'|'.join(COMMANDS)}}} [count] [shards]")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    shards = int(sys.argv[3]) if len(sys.argv) > 3 else SAMPLE_SHARDS
    COMMANDS[sys.argv[1]](sample_rows(count, shards))
