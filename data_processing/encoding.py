"""
Level encoding
    - Maps every element stored in SMM2 level_data to gameplay channels (SOLID, HAZARD, ...)
    - Paints those channels into the multi-hot (channels, height, width) grid of each area
    - Packs the scalar fields (theme, liquid, clear condition, ...) into one-hot vectors
    - Reasoning per object ID lives in the CSE 330 Final Project doc (Level Object Categories tab)
"""

# Imports
from enum import IntEnum
from io import BytesIO
import zlib
import torch
from kaitaistruct import KaitaiStream
from level import Level




# ---------------------------------
# Enum aliases and flag bits
# ---------------------------------
ObjId = Level.Obj.ObjId
Gamestyle = Level.Gamestyle
Theme = Level.Map.Theme
LiquidMode = Level.Map.LiquidMode
Orientation = Level.Map.Orientation

ALT_FORM_BIT = 0x4  # Flag bit for an object's alternate form (Toost's renderer)
WINGS_BIT = 0x2  # Flag bit for wings (Toost's renderer)
PARACHUTE_BIT = 0x8000  # Flag bit for a parachute (Toost's renderer)
NO_CHILD = -1  # Value of cid when nothing is inside the object




# ---------------------------------
# Grid geometry
# ---------------------------------
NUM_CH = 12  # Number of channels
NUM_OBJS = 133  # Size of the ObjId enum (IDs 0 to 132)
TILE = 160  # Raw object units per tile
RAW_OFFSET = 80  # Raw position of a tile center (tile index = (raw - 80) // 160, 0-indexed)
BOUNDARY_UNITS_PER_TILE = 16  # Boundary fields count 1/16 tiles (per Toost)
START_AREA_WIDTH = 7  # Start area is always the 7 leftmost columns
PADDED_HEIGHT = 168  # Tallest area (vertical), padded CNN only
PADDED_WIDTH = 240  # Widest area (horizontal), padded CNN only

PIPE, CLEAR_PIPE, CLAW, ICICLE = 9, 93, 105, 118  # Object IDs with their own footprint rules
LEFT_ANCHORED = {14, 16, 71, 17, 53, 94}  # X is the leftmost tile, Y the bottom row
SLOPES = {87, 88}
SKIP_IN_OBJECTS = {CLEAR_PIPE, ICICLE}  # Taken from the clear_pipes and icicles arrays instead
PIPE_FLAG_MASK = 0x7F  # Keeps the direction bits, drops the rest of the flag
SLOPE_FALLING_BIT = 20  # Flag bit index, set means the slope falls left to right

OBJECT_PIPE_DIRECTIONS = {0x00: "R", 0x20: "L", 0x40: "U", 0x60: "D"}  # Flag & PIPE_FLAG_MASK to opening direction
CLEAR_PIPE_DIRECTIONS = "RLUD"  # Indexed by clear pipe node direction (0 to 3)
LIQUID_THEMES = (Theme.castle, Theme.forest)  # Themes where the liquid level comes from the header fields

CLEAR_CONDITIONS = list(Level.ClearCondition)  # Enum order is the one-hot position
GAMESTYLES = list(Level.Gamestyle)
CLEAR_CATEGORY_COUNT = 4  # Category one-hot comes first in the clear condition vector (none, parts, status, actions)
CLEAR_CONDITION_SIZE = 99  # 4 categories + 94 conditions + 1 magnitude (last slot)




# ---------------------------------
# Channel
# ---------------------------------
class Channel(IntEnum):
    """
    Grid channels
        - One channel per gameplay role, so the networks see what an element does rather than which sprite it is
        - Cells are multi-hot (a ? Block holding a mushroom sets both SOLID and HELPER)
        - LIQUID and IN_BOUNDS are painted from area header fields, not from objects
    """

    SOLID = 0  # Collides on all sides (terrain, blocks, pipes, slopes, launchers, moving blocks)
    SEMISOLID = 1  # Passable from below or one side (semisolids, bridges, clouds, one-way walls)
    CONDITIONAL = 2  # Appears, disappears, or switches state (donuts, ON/OFF, hidden, P and blinking blocks)
    MOVEMENT = 3  # Moves Mario or moves under him (lifts, conveyors, tracks, springs, vines, bumpers)
    HAZARD = 4  # Traps (spikes, munchers, grinders, burners, fire bars, launchers)
    ENEMY = 5  # Creatures, bosses included
    HELPER = 6  # Power-ups, rides, wearables, checkpoints
    COLLECTIBLE = 7  # Coins and keys
    INTERACTIVE = 8  # Switches, doors, warp boxes, clear pipes, keys
    AIRBORNE = 9  # Object has wings or a parachute
    LIQUID = 10  # Rows under the area's liquid level
    IN_BOUNDS = 11  # Cells inside the area's boundary (padding stays 0)




# ---------------------------------
# Channel groups
# ---------------------------------
"""
Object ID to channel assignments
    - An ID can sit in more than one group (a launcher is SOLID and a HAZARD)
    - NEUTRAL IDs have no gameplay effect or never appear in SMM2 objects (start and goal come from the header)
    - UNIDENTIFIED IDs get no channel until the inventory finds them and the level viewer says what they are
    - jumping_machine (50) is inferred to be Hop-Chops (3D World only, comes big, only 3D World enemy with no other ID)
"""
GROUPS = {
    Channel.SOLID: [
        ObjId.block, ObjId.question_block, ObjId.hard_block, ObjId.ground, ObjId.pipe,
        ObjId.bullet_bill_blaster, ObjId.donut_block, ObjId.note_block, ObjId.goal_ground,
        ObjId.starting_brick, ObjId.spikes, ObjId.cannon, ObjId.castle_bridge, ObjId.fast_conveyor_belt,
        ObjId.burner, ObjId.muncher, ObjId.ice_block, ObjId.sprint_platform, ObjId.skewer,
        ObjId.snake_block, ObjId.track_block, ObjId.slight_slope, ObjId.steep_slope, ObjId.clear_pipe,
        ObjId.conveyor_belt, ObjId.on_off_block, ObjId.spike_block, ObjId.crate, ObjId.exclamation_block
    ],
    Channel.SEMISOLID: [
        ObjId.mushroom_platform, ObjId.semisolid_platform, ObjId.bridge, ObjId.cloud, ObjId.one_way,
        ObjId.half_collision_platform, ObjId.tree
    ],
    Channel.CONDITIONAL: [
        ObjId.donut_block, ObjId.hidden_block, ObjId.ice_block, ObjId.p_block, ObjId.dotted_line_block,
        ObjId.blinking_block, ObjId.crate, ObjId.on_off_trampoline
    ],
    Channel.MOVEMENT: [
        ObjId.spring, ObjId.lift, ObjId.note_block, ObjId.lava_lift, ObjId.fast_conveyor_belt,
        ObjId.track, ObjId.vine, ObjId.twister, ObjId.sprint_platform, ObjId.donut, ObjId.snake_block,
        ObjId.track_block, ObjId.seesaw, ObjId.conveyor_belt, ObjId.swinging_claw,
        ObjId.mushroom_trampoline, ObjId.exclamation_block, ObjId.on_off_trampoline
    ],
    Channel.HAZARD: [
        ObjId.bullet_bill_blaster, ObjId.fire_bar, ObjId.spikes, ObjId.cannon, ObjId.burner,
        ObjId.muncher, ObjId.saw, ObjId.skewer, ObjId.spike_block, ObjId.icicle
    ],
    Channel.ENEMY: [
        ObjId.goomba, ObjId.koopa, ObjId.piranha_flower, ObjId.hammer_bro, ObjId.thwomp, ObjId.bob_omb,
        ObjId.spiny, ObjId.buzzy_beetle, ObjId.lakitu, ObjId.banzai_bill, ObjId.magikoopa,
        ObjId.spike_top, ObjId.boo, ObjId.shoe_goomba, ObjId.dry_bones, ObjId.blooper, ObjId.skipsqueak,
        ObjId.wiggler, ObjId.cheep_cheep, ObjId.rocky_wrench, ObjId.lava_bubble, ObjId.chain_chomp,
        ObjId.bowser, ObjId.stingby, ObjId.spike_ball, ObjId.boom_boom, ObjId.pokey, ObjId.charvaargh,
        ObjId.ant_trooper, ObjId.bowser_jr, ObjId.jumping_machine, ObjId.monty_mole, ObjId.fish_bone,
        ObjId.angry_sun, ObjId.piranha_creeper, ObjId.mechakoopa, ObjId.porkupuffer, ObjId.bully,
        ObjId.lemmy, ObjId.morton, ObjId.larry, ObjId.wendy, ObjId.iggy, ObjId.roy, ObjId.ludwig
    ],
    Channel.HELPER: [
        ObjId.super_mushroom, ObjId.lakitu_cloud, ObjId.one_up, ObjId.fire_flower, ObjId.super_star,
        ObjId.clown_car, ObjId.big_mushroom, ObjId.shoe_goomba, ObjId.koopa_car, ObjId.smb2_mushroom,
        ObjId.checkpoint_flag, ObjId.super_hammer, ObjId.cannon_box, ObjId.propeller_box,
        ObjId.goomba_mask, ObjId.bullet_bill_mask, ObjId.red_pow_box
    ],
    Channel.COLLECTIBLE: [ObjId.coin, ObjId.big_coin, ObjId.red_coin, ObjId.key],
    Channel.INTERACTIVE: [
        ObjId.p_switch, ObjId.pow, ObjId.door, ObjId.clear_pipe, ObjId.key, ObjId.warp_box,
        ObjId.on_off_block
    ]
}

NEUTRAL = [
    ObjId.goal, ObjId.starting_arrow, ObjId.arrow, ObjId.player, ObjId.reel_camera, ObjId.water_marker,
    ObjId.sound_effect
]

UNIDENTIFIED = [ObjId.cinobio, ObjId.stone, ObjId.cinobic]




# ---------------------------------
# Data-driven overrides
# ---------------------------------
"""
Overrides that come from the data itself
    - ALT_FORM_CHANNELS replace an ID's channels when its flag (or cflag, for contents) has ALT_FORM_BIT set
    - Each alternate form is the only variant the wiki lists for that part (Toost draws it from this bit)
    - Rotten Mushroom and Yoshi are decided by area theme, night mode, and game style instead of a bit
"""
ALT_FORM_CHANNELS = {
    ObjId.coin: frozenset({Channel.SOLID, Channel.CONDITIONAL}),  # Frozen coin
    ObjId.spiny: frozenset({Channel.HELPER}),  # Spiny Shell, worn as a helmet
    ObjId.buzzy_beetle: frozenset({Channel.HELPER}),  # Buzzy Shell, worn as a helmet
    ObjId.dry_bones: frozenset({Channel.HELPER}),  # Dry Bones Shell, wearable
    ObjId.spike_ball: frozenset({Channel.HAZARD}),  # Lone spike ball or snowball instead of the Spike enemy
    ObjId.p_block: frozenset({Channel.SOLID, Channel.CONDITIONAL}),  # Starts solid
    ObjId.key: frozenset({Channel.COLLECTIBLE, Channel.INTERACTIVE, Channel.ENEMY}),  # Cursed Key (a Phanto chases the carrier)
    ObjId.angry_sun: frozenset({Channel.HELPER})  # Moon
}

ROTTEN_MUSHROOM_CHANNELS = frozenset({Channel.ENEMY})  # 1-Up in a night Ground-theme area
YOSHI_CHANNELS = frozenset({Channel.HELPER})  # Shoe Goomba in SMW and NSMBU is Yoshi's Egg
YOSHI_STYLES = {Gamestyle.smw, Gamestyle.nsmbw}  # The ksy's "nsmbw" is NSMBU




# ---------------------------------
# Channels outside the objects array
# ---------------------------------
"""
Channels for elements stored outside the objects array
    - Ground tiles live in map.ground in tile units and hold nearly all terrain
    - Icicle type 0 falls, type 1 is solid (Course Format.md)
"""
GROUND_TILE_CHANNELS = frozenset({Channel.SOLID})
TRACK_TILE_CHANNELS = frozenset({Channel.MOVEMENT})
CLEAR_PIPE_CHANNELS = frozenset({Channel.SOLID, Channel.INTERACTIVE})
ICICLE_CHANNELS = {
    0: frozenset({Channel.HAZARD}),
    1: frozenset({Channel.SOLID})
}




# ---------------------------------
# UnknownObjectError
# ---------------------------------
class UnknownObjectError(ValueError):
    """Raised when a level contains an object ID missing from level.py's ObjId enum"""




# ---------------------------------
# _buildObjectChannels
# ---------------------------------
def _buildObjectChannels():
    """
    Invert GROUPS into one lookup from object ID to its channels

    Returns:
        Dict mapping every ObjId to a frozenset of Channel (empty for NEUTRAL and UNIDENTIFIED IDs)

    Raises:
        ValueError: If an ObjId sits in none of GROUPS, NEUTRAL, or UNIDENTIFIED (catching enum additions)
    """
    lookup = {objId: set() for objId in NEUTRAL + UNIDENTIFIED}

    for channel, objIds in GROUPS.items():
        for objId in objIds:
            lookup.setdefault(objId, set()).add(channel)

    missing = [objId.name for objId in ObjId if objId not in lookup]
    if missing:
        raise ValueError(f"Object IDs with no group: {missing}")

    return {objId: frozenset(channels) for objId, channels in lookup.items()}


OBJECT_CHANNELS = _buildObjectChannels()




# ---------------------------------
# channelsFor
# ---------------------------------
def channelsFor(obj, gamestyle, theme, night):
    """
    Channels a parsed object contributes to every cell it covers

    Args:
        obj: A Level.Obj from one area's sliced objects
        gamestyle: level.gamestyle
        theme: The area's theme (areaMap.theme)
        night: True if the area is in night mode

    Returns:
        Frozenset of Channel (the object's own, its contents', and AIRBORNE if winged or parachuting)
    """
    channels = set(elementChannels(obj.id, obj.flag, gamestyle, theme, night))

    # Adding the contents where the container sits (mushroom in a ? Block, Goombas out of a pipe)
    if obj.cid != NO_CHILD:
        channels |= elementChannels(obj.cid, obj.cflag, gamestyle, theme, night)

    if (obj.flag | obj.cflag) & (WINGS_BIT | PARACHUTE_BIT):
        channels.add(Channel.AIRBORNE)

    return frozenset(channels)




# ---------------------------------
# elementChannels
# ---------------------------------
def elementChannels(rawId, flag, gamestyle, theme, night):
    """
    Channels for one object ID in its context, before contents and modifiers

    Args:
        rawId: An ObjId member or a raw int (kaitai returns an int for values outside the enum)
        flag: The object's flag (or the container's cflag when rawId is a cid)
        gamestyle: level.gamestyle
        theme: The area's theme
        night: True if the area is in night mode

    Returns:
        Frozenset of Channel

    Raises:
        UnknownObjectError: If rawId isn't a known ObjId
    """
    try:
        objId = ObjId(rawId)
    except ValueError:
        raise UnknownObjectError(f"Unknown object ID {rawId}") from None

    if objId == ObjId.one_up and theme == Theme.overworld and night:
        return ROTTEN_MUSHROOM_CHANNELS
    if objId == ObjId.shoe_goomba and gamestyle in YOSHI_STYLES:
        return YOSHI_CHANNELS
    if flag & ALT_FORM_BIT and objId in ALT_FORM_CHANNELS:
        return ALT_FORM_CHANNELS[objId]

    return OBJECT_CHANNELS[objId]




# ---------------------------------
# parse
# ---------------------------------
def parse(row):
    """
    Decompress a dataset row's level_data and parse it into a Level

    Args:
        row: Dataset row with a zlib-compressed level_data
    """
    data = zlib.decompress(row["level_data"])
    return Level(KaitaiStream(BytesIO(data)))




# ---------------------------------
# _toInt
# ---------------------------------
def _toInt(value):
    """
    Convert a Kaitai enum member or an int to a plain int

    Args:
        value: Kaitai enum member or int
    """
    return int(getattr(value, "value", value))




# ---------------------------------
# _pipeCells
# ---------------------------------
def _pipeCells(col, row, length, direction):
    """
    Cells covered by a 2-thick pipe anchored at (col, row)

    Args:
        col: Anchor tile column
        row: Anchor tile row
        length: The object's height field (the length, for every direction)
        direction: Opening direction, one of "U", "D", "R", "L"

    Returns:
        Inclusive (colStart, colEnd, rowStart, rowEnd)
    """
    if direction == "U":
        return col, col + 1, row, row + length - 1
    if direction == "D":
        return col - 1, col, row - length + 1, row
    if direction == "R":
        return col, col + length - 1, row - 1, row

    return col - length + 1, col, row, row + 1  # Left




# ---------------------------------
# _fillRectangle
# ---------------------------------
def _fillRectangle(grid, channels, colStart, colEnd, rowStart, rowEnd):
    """
    Set each channel to 1 on an inclusive rectangle, clipped to the grid

    Args:
        grid: (channels, height, width) tensor, edited in place
        channels: Iterable of Channel
        colStart: First column
        colEnd: Last column
        rowStart: First row (row 0 is the bottom of the map)
        rowEnd: Last row
    """
    height, width = grid.shape[1], grid.shape[2]
    colStart, colEnd = max(colStart, 0), min(colEnd, width - 1)
    rowStart, rowEnd = max(rowStart, 0), min(rowEnd, height - 1)
    if colStart > colEnd or rowStart > rowEnd:
        return

    for channel in channels:
        grid[channel, rowStart:rowEnd + 1, colStart:colEnd + 1] = 1




# ---------------------------------
# _paintLiquid
# ---------------------------------
def _paintLiquid(grid, areaMap):
    """
    Paint the LIQUID channel from the area's liquid header fields

    Args:
        grid: (channels, height, width) tensor, edited in place
        areaMap: level.overworld or level.subworld
    """
    if areaMap.theme in LIQUID_THEMES:
        if areaMap.liquid_mode == LiquidMode.rising_and_falling:
            grid[Channel.LIQUID, 0:areaMap.liquid_start_height, :] = 0.5
            grid[Channel.LIQUID, 0:areaMap.liquid_end_height, :] += 0.5  # 1 below both heights, 0.5 between them
        else:
            grid[Channel.LIQUID, 0:areaMap.liquid_end_height, :] = 1

    if areaMap.theme == Theme.underwater:
        grid[Channel.LIQUID, :, :] = 1  # Every tile is water




# ---------------------------------
# _paintInBounds
# ---------------------------------
def _paintInBounds(grid, areaMap):
    """
    Paint IN_BOUNDS on every tile inside the area's boundary (padding stays 0)

    Args:
        grid: (channels, height, width) tensor, edited in place
        areaMap: level.overworld or level.subworld
    """
    top = areaMap.boundary_top // BOUNDARY_UNITS_PER_TILE
    right = areaMap.boundary_right // BOUNDARY_UNITS_PER_TILE
    grid[Channel.IN_BOUNDS, 0:top, 0:right] = 1




# ---------------------------------
# _paintTerrain
# ---------------------------------
def _paintTerrain(grid, level, areaMap, area, includeImplicitGround):
    """
    Paint ground tiles, plus the start and goal ground drawn from the level header

    Args:
        grid: (channels, height, width) tensor, edited in place
        level: Parsed Level (header fields for the start and goal)
        areaMap: level.overworld or level.subworld
        area: "overworld" or "subworld"
        includeImplicitGround: True paints the start and goal ground (overworld only)
    """
    for tile in areaMap.ground[:areaMap.ground_count]:
        _fillRectangle(grid, GROUND_TILE_CHANNELS, tile.x, tile.x, tile.y, tile.y)  # Already 0-indexed tile coordinates

    if includeImplicitGround and area == "overworld":
        # Start and goal ground come from the header, not the ground array (as Toost does)
        goalStart = int((level.goal_x - 5) / 10 + 0.5)
        goalEnd = int((level.goal_x - 5) / 10 + 9)
        _fillRectangle(grid, GROUND_TILE_CHANNELS, goalStart, goalEnd, 0, level.goal_y - 1)
        _fillRectangle(grid, GROUND_TILE_CHANNELS, 0, START_AREA_WIDTH - 1, 0, level.start_y - 1)




# ---------------------------------
# _paintClearPipes
# ---------------------------------
def _paintClearPipes(grid, areaMap):
    """
    Paint every clear pipe segment from the clear_pipes array

    Args:
        grid: (channels, height, width) tensor, edited in place
        areaMap: level.overworld or level.subworld
    """
    for pipe in areaMap.clear_pipes[:areaMap.clear_pipe_count]:
        for node in pipe.nodes[:pipe.node_count]:
            direction = CLEAR_PIPE_DIRECTIONS[_toInt(node.direction)]
            _fillRectangle(grid, CLEAR_PIPE_CHANNELS, *_pipeCells(node.x, node.y, node.height, direction))




# ---------------------------------
# _paintIcicles
# ---------------------------------
def _paintIcicles(grid, areaMap):
    """
    Paint every icicle from the icicles array (1 wide, 2 tall)

    Args:
        grid: (channels, height, width) tensor, edited in place
        areaMap: level.overworld or level.subworld
    """
    for icicle in areaMap.icicles[:areaMap.ice_count]:
        _fillRectangle(grid, ICICLE_CHANNELS[_toInt(icicle.type)], icicle.x, icicle.x, icicle.y, icicle.y + 1)




# ---------------------------------
# mapToTensor
# ---------------------------------
def mapToTensor (
    level,
    area="overworld",
    height=None,
    width=None,
    includeImplicitGround=True,
    dtype=torch.float32
):
    """
    Multi-hot (12, height, width) tensor for one area of a parsed level

    Args:
        level: Parsed Level
        area: "overworld" or "subworld"
        height: Rows in the tensor (defaults to the area's own height)
        width: Columns in the tensor (defaults to the area's own width)
        includeImplicitGround: True paints the start and goal ground (overworld only)
        dtype: Tensor dtype

    Returns:
        Tensor indexed [channel, y, x], y = 0 at the bottom of the map (flip dim 1 for image orientation)
    """
    areaMap = getattr(level, area)  # level.overworld or level.subworld
    height = height or areaMap.boundary_top // BOUNDARY_UNITS_PER_TILE
    width = width or areaMap.boundary_right // BOUNDARY_UNITS_PER_TILE
    grid = torch.zeros((NUM_CH, height, width), dtype=dtype)

    _paintLiquid(grid, areaMap)
    _paintInBounds(grid, areaMap)
    _paintTerrain(grid, level, areaMap, area, includeImplicitGround)

    night = areaMap.unk_flag == 1  # Night mode, same test as levelToInput

    for obj in areaMap.objects[:areaMap.object_count]:
        objectId = _toInt(obj.id)
        if not 0 <= objectId < NUM_OBJS or objectId in SKIP_IN_OBJECTS:
            continue

        objWidth, objHeight = max(obj.width, 1), max(obj.height, 1)
        channels = channelsFor(obj, level.gamestyle, areaMap.theme, night)  # Own, contents, overrides, AIRBORNE
        anchorCol, anchorRow = (obj.x - RAW_OFFSET) // TILE, (obj.y - RAW_OFFSET) // TILE

        if objectId == PIPE:
            direction = OBJECT_PIPE_DIRECTIONS.get(obj.flag & PIPE_FLAG_MASK)
            if direction is None:
                continue
            _fillRectangle(grid, channels, *_pipeCells(anchorCol, anchorRow, objHeight, direction))
        elif objectId == CLAW:  # 3 wide, 5 tall, hanging down from Y
            _fillRectangle(grid, channels, anchorCol - 1, anchorCol + 1, anchorRow - 4, anchorRow)
        elif objectId in SLOPES:  # Surface diagonal only, one cell per column
            ascending = ((obj.flag >> SLOPE_FALLING_BIT) & 1) == 0
            for offset in range(objWidth):
                row = anchorRow + ((offset if ascending else objWidth - 1 - offset) * objHeight) // objWidth
                _fillRectangle(grid, channels, anchorCol + offset, anchorCol + offset, row, row)
        elif objectId in LEFT_ANCHORED:
            _fillRectangle(grid, channels, anchorCol, anchorCol + objWidth - 1, anchorRow, anchorRow + objHeight - 1)
        else:  # X is the horizontal center, Y the bottom row
            colStart = (obj.x - RAW_OFFSET * objWidth) // TILE
            _fillRectangle(grid, channels, colStart, colStart + objWidth - 1, anchorRow, anchorRow + objHeight - 1)

    _paintClearPipes(grid, areaMap)
    _paintIcicles(grid, areaMap)

    return grid




# ---------------------------------
# _oneHot
# ---------------------------------
def _oneHot(size, index):
    """
    One-hot float vector

    Args:
        size: Vector length
        index: Position set to 1 (enum members work)
    """
    vector = torch.zeros(size)
    vector[index] = 1
    return vector




# ---------------------------------
# _nightFlag
# ---------------------------------
def _nightFlag(areaMap):
    """
    Night mode as a one-value tensor (1 if night, else 0)

    Args:
        areaMap: level.overworld or level.subworld
    """
    return torch.tensor([1 if areaMap.unk_flag == 1 else 0])




# ---------------------------------
# _liquidVector
# ---------------------------------
def _liquidVector(areaMap):
    """
    Liquid fields as [start height, end height, 3 mode one-hot slots, speed]

    Args:
        areaMap: level.overworld or level.subworld
    """
    vector = torch.tensor([areaMap.liquid_start_height, areaMap.liquid_end_height, 0, 0, 0, areaMap.liquid_speed])
    vector[areaMap.liquid_mode + 2] = 1  # Mode one-hot sits in slots 2 to 4
    return vector




# ---------------------------------
# _areaScalars
# ---------------------------------
def _areaScalars(areaMap, suffix):
    """
    The per-area scalar inputs, keyed with a suffix

    Args:
        areaMap: level.overworld or level.subworld
        suffix: "ow" or "sw"

    Returns:
        Dict of theme, night, liquid, and autoscroll tensors (keys end in the suffix)
    """
    return {
        f"theme_{suffix}": _oneHot(10, areaMap.theme),
        f"night_{suffix}": _nightFlag(areaMap),
        f"liquid_{suffix}": _liquidVector(areaMap),
        f"autoscroll_{suffix}": _oneHot(5, areaMap.autoscroll_type)
    }




# ---------------------------------
# _clearConditionVector
# ---------------------------------
def _clearConditionVector(level):
    """
    Clear condition as [category one-hot, condition one-hot, magnitude]

    Args:
        level: Parsed Level
    """
    vector = torch.zeros(CLEAR_CONDITION_SIZE)
    vector[level.clear_condition_category] = 1
    vector[CLEAR_CONDITIONS.index(level.clear_condition) + CLEAR_CATEGORY_COUNT] = 1
    vector[-1] = level.clear_condition_magnitude
    return vector




# ---------------------------------
# levelToInput
# ---------------------------------
def levelToInput(level: Level, pad: bool = False):
    """
    Encode a parsed level into the network inputs

    Args:
        level: Parsed Level
        pad: True pads grids to the max area size (padded CNN only), False keeps each area's own size

    Returns:
        Dict of tensors
            - ow: Overworld grid (12, height, width)
            - sw_hor, sw_vert: Subworld grid for its orientation, None when the subworld is absent or the other orientation
            - theme_*, night_*, liquid_*, autoscroll_*: Per-area scalars, suffix ow or sw (sw is None when absent)
            - clear_condition, gamestyle, timer: Per-level scalars
    """
    gridHeight, gridWidth = None, None
    if pad:
        gridHeight = PADDED_HEIGHT
        gridWidth = PADDED_WIDTH

    output = {}
    output["ow"] = mapToTensor(level, width=gridWidth)
    output.update(_areaScalars(level.overworld, "ow"))
    output["clear_condition"] = _clearConditionVector(level)
    output["gamestyle"] = _oneHot(5, GAMESTYLES.index(level.gamestyle))
    output["timer"] = torch.tensor([level.timer])

    for key in ("sw_hor", "sw_vert", "theme_sw", "night_sw", "liquid_sw", "autoscroll_sw"):
        output[key] = None

    if level.subworld is not None:
        if level.subworld.orientation == Orientation.horizontal:
            output["sw_hor"] = mapToTensor(level, "subworld", width=gridWidth)
        else:
            output["sw_vert"] = mapToTensor(level, "subworld", height=gridHeight)
        output.update(_areaScalars(level.subworld, "sw"))

    return output
