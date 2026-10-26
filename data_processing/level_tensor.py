import torch
from categories import (Channel, channels_for, GROUND_TILE_CHANNELS, CLEAR_PIPE_CHANNELS,
                        ICICLE_CHANNELS)
from level import Level

NUM_CH = 12           # number of channels
NUM_OBJS = 133        # size of the obj_id enum (ids 0..132)
TILE = 160            # raw units per tile
RAW_OFFSET = 80       # tile0 = (raw - 80) // 160, i.e. 0-based; same as your (raw+80)/160 minus 1

GROUND, PIPE, CLEAR_PIPE, CLAW, ICICLE = 7, 9, 93, 105, 118  # GROUND is unused now (tiles use GROUND_TILE_CHANNELS)
LEFT_ANCHORED = {14, 16, 71, 17, 53, 94}   # X is the leftmost tile, Y the bottom row
SLOPES = {87, 88}
SKIP_IN_OBJECTS = {CLEAR_PIPE, ICICLE}     # taken from clear_pipes / icicles arrays instead

OBJ_PIPE_DIR = {0x00: "R", 0x20: "L", 0x40: "U", 0x60: "D"}   # flag & 0x7F -> opening direction
CPIPE_DIR = "RLUD"                                            # clear_pipe_node.direction 0..3


def _i(v):
    """Kaitai enum / int -> int."""
    return int(getattr(v, "value", v))


def _pipe_cells(c, r, length, d):
    """Inclusive (col0, col1, row0, row1) of a 2-thick pipe whose anchor tile is (c, r).
    d is the opening direction; `length` is the object's *height* field for every direction."""
    if d == "U":
        return c, c + 1, r, r + length - 1
    if d == "D":
        return c - 1, c, r - length + 1, r
    if d == "R":
        return c, c + length - 1, r - 1, r
    return c - length + 1, c, r, r + 1  # "L"


def _rect(t, channels, c0, c1, r0, r1):
    """Set each Channel in `channels` to 1 on the inclusive rectangle, clipped to the map."""
    h, w = t.shape[1], t.shape[2]
    c0, c1, r0, r1 = max(c0, 0), min(c1, w - 1), max(r0, 0), min(r1, h - 1)
    if c0 > c1 or r0 > r1:
        return
    for ch in channels:
        t[ch, r0:r1 + 1, c0:c1 + 1] = 1


def map_to_tensor(level, area="overworld", height=None, width=None,
                    include_implicit_ground=True, dtype=torch.float32):
    """Multi-hot (12, height, width) tensor for one area of a parsed level.
    Index as t[channel, y, x] with y = 0 at the BOTTOM of the map (flip dim 1 for image orientation)."""
    m = getattr(level, area)                      # level.overworld or level.subworld
    height = height or m.boundary_top // 16       # boundary fields are in 1/16-tile units per toost
    width = width or m.boundary_right // 16
    t = torch.zeros((NUM_CH, height, width), dtype=dtype)
    if m.theme == 2 or m.theme == 9:
        if m.liquid_mode == 2:
            t[10, 0:m.liquid_start_height, :] = 0.5
            t[10, 0:m.liquid_end_height, :] += 0.5
        else:
            t[10, 0:m.liquid_end_height, :] = 1
    if m.theme == 4:
        t[10, :, :] = 1
    t[11, 0:m.boundary_top // 16, 0:m.boundary_right // 16] = 1

    night = m.unk_flag == 1                       # night mode, as in level_to_input

    # --- terrain ---------------------------------------------------------------
    for g in m.ground[:m.ground_count]:
        _rect(t, GROUND_TILE_CHANNELS, g.x, g.x, g.y, g.y)    # ground bytes are already 0-based tile coords

    if include_implicit_ground and area == "overworld":
        # Ground under the start and goal comes from the header, not the ground array (as toost does).
        j0 = int((level.goal_x - 5) / 10 + 0.5)
        j1 = int((level.goal_x - 5) / 10 + 9)
        _rect(t, GROUND_TILE_CHANNELS, j0, j1, 0, level.goal_y - 1)
        _rect(t, GROUND_TILE_CHANNELS, 0, 6, 0, level.start_y - 1)

    # --- objects (+ contained objects) -----------------------------------------
    for o in m.objects[:m.object_count]:
        oid = _i(o.id)
        if not 0 <= oid < NUM_OBJS or oid in SKIP_IN_OBJECTS:
            continue
        w, h = max(o.width, 1), max(o.height, 1)
        # Own channels + contents + alt-form / Rotten Mushroom / Yoshi overrides + AIRBORNE
        chans = channels_for(o, level.gamestyle, m.theme, night)

        c, r = (o.x - RAW_OFFSET) // TILE, (o.y - RAW_OFFSET) // TILE   # anchor tile, 0-based
        if oid == PIPE:
            d = OBJ_PIPE_DIR.get(o.flag & 0x7F)
            if d is None:
                continue
            _rect(t, chans, *_pipe_cells(c, r, h, d))
        elif oid == CLAW:                         # 3 wide x 5 tall, hangs down from Y
            _rect(t, chans, c - 1, c + 1, r - 4, r)
        elif oid in SLOPES:                       # only the surface diagonal, one cell per column
            ascending = ((o.flag >> 20) & 1) == 0
            for k in range(w):
                row = r + ((k if ascending else w - 1 - k) * h) // w
                _rect(t, chans, c + k, c + k, row, row)
        elif oid in LEFT_ANCHORED:
            _rect(t, chans, c, c + w - 1, r, r + h - 1)
        else:                                     # default: X is the horizontal center, Y the bottom row
            c0 = (o.x - RAW_OFFSET * w) // TILE
            _rect(t, chans, c0, c0 + w - 1, r, r + h - 1)

    # --- clear pipes and icicles live in their own arrays ----------------------
    for p in m.clear_pipes[:m.clear_pipe_count]:
        for n in p.nodes[:p.node_count]:
            _rect(t, CLEAR_PIPE_CHANNELS, *_pipe_cells(n.x, n.y, n.height, CPIPE_DIR[_i(n.direction)]))
    for ic in m.icicles[:m.ice_count]:
        _rect(t, ICICLE_CHANNELS[_i(ic.type)], ic.x, ic.x, ic.y, ic.y + 1)   # 1 wide x 2 tall; type 0 falls (HAZARD), 1 is SOLID

    return t

def level_to_input(level: Level, pad: bool = False):
    h, w = None, None
    if pad:
        h = 168
        w = 240
    ow = map_to_tensor(level, width=w)
    theme_ow = torch.zeros(10)
    theme_ow[level.overworld.theme] = 1
    night_ow = torch.tensor([1 if level.overworld.unk_flag == 1 else 0])
    liquid_ow = torch.tensor(
        [level.overworld.liquid_start_height, level.overworld.liquid_end_height, 0, 0, 0, level.overworld.liquid_speed])
    liquid_ow[level.overworld.liquid_mode + 2] = 1
    autoscroll_ow = torch.zeros(5)
    autoscroll_ow[level.overworld.autoscroll_type] = 1
    clear_condition = torch.zeros(99)
    clear_condition[level.clear_condition_category] = 1
    clear_condition[level.clear_condition + 4] = 1
    clear_condition[-1] = level.clear_condition_magnitude
    gamestyle = torch.zeros(5)
    gamestyle[list(Level.Gamestyle).index(level.gamestyle)] = 1
    timer = torch.tensor([level.timer])
    sw_hor = None
    sw_vert = None
    theme_sw = None
    night_sw = None
    liquid_sw = None
    autoscroll_sw = None
    if level.subworld is not None:
        if level.subworld.orientation == Level.Map.Orientation.horizontal:
            sw_hor = map_to_tensor(level, "subworld", width=w)
        else:
            sw_vert = map_to_tensor(level, "subworld", height=h)
        liquid_sw = torch.tensor([level.subworld.liquid_start_height, level.subworld.liquid_end_height, 0, 0, 0,
                                  level.subworld.liquid_speed])
        liquid_sw[level.subworld.liquid_mode + 2] = 1
        autoscroll_sw = torch.zeros(5)
        autoscroll_sw[level.subworld.autoscroll_type] = 1
        theme_sw = torch.zeros(10)
        theme_sw[level.subworld.theme] = 1
        night_sw = torch.tensor([1 if level.subworld.unk_flag == 1 else 0])
    return ow, theme_ow, night_ow, liquid_ow, autoscroll_ow, sw_hor, sw_vert, theme_sw, night_sw, liquid_sw, autoscroll_sw, clear_condition, gamestyle, timer
