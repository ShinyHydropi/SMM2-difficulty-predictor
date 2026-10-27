"""
What every element stored in SMM2 level_data means for gameplay, as grid channels
    - Each Channel is one layer of the multi-hot (channels, height, width) area grid
    - Only elements that exist in level_data get a category (object IDs, flag bits, secondary arrays)
    - Reasoning per ID lives in the SMM2 Level Encoding Spec doc
"""

# Imports
from enum import IntEnum
from level import Level


ObjId = Level.Obj.ObjId
Gamestyle = Level.Gamestyle
Theme = Level.Map.Theme

ALT_FORM_BIT = 0x4  # Flag bit for an object's alternate form (Toost's renderer)
WINGS_BIT = 0x2  # Flag bit for wings (Toost's renderer)
PARACHUTE_BIT = 0x8000  # Flag bit for a parachute (Toost's renderer)
NO_CHILD = -1  # cid value when nothing is inside the object


"""
Grid channels
    - One channel per gameplay role, so the networks see what an element does rather than which sprite it is
    - Cells are multi-hot (a ? Block holding a mushroom sets both SOLID and HELPER)
    - LIQUID and IN_BOUNDS are painted from area header fields, not from objects
"""
class Channel(IntEnum):
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


"""
Object ID to channel assignments
    - An ID can sit in more than one group (a launcher is SOLID and a HAZARD)
    - NEUTRAL IDs have no gameplay effect or never appear in SMM2 objects (start and goal come from the header)
    - UNIDENTIFIED IDs get no channel until the inventory finds them and the level viewer says what they are
    - jumping_machine (50) is inferred to be Hop-Chops: 3D World only, comes big, and the only 3D World enemy with no other ID
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


"""
Overrides that come from the data itself
    - ALT_FORM_CHANNELS replace an ID's channels when its flag (or cflag, for contents) has ALT_FORM_BIT set
    - Each alternate form is the only variant the wiki lists for that part, and Toost draws it from this bit
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
YOSHI_CHANNELS = frozenset({Channel.HELPER})  # shoe_goomba in SMW and NSMBU is Yoshi's Egg
YOSHI_STYLES = {Gamestyle.smw, Gamestyle.nsmbw}  # The ksy's "nsmbw" is NSMBU


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


class UnknownObjectError(ValueError):
    """Raised when a level contains an object ID missing from level.py's ObjId enum"""


def _build_object_channels():
    """
    Invert GROUPS into one lookup from object ID to its channels

    Returns:
        Dict mapping every ObjId to a frozenset of Channel (empty for NEUTRAL and UNIDENTIFIED IDs)

    Raises:
        ValueError: If an ObjId sits in none of GROUPS, NEUTRAL, or UNIDENTIFIED (catches enum additions)
    """
    lookup = {obj_id: set() for obj_id in NEUTRAL + UNIDENTIFIED}

    for channel, obj_ids in GROUPS.items():
        for obj_id in obj_ids:
            lookup.setdefault(obj_id, set()).add(channel)

    missing = [obj_id.name for obj_id in ObjId if obj_id not in lookup]
    if missing:
        raise ValueError(f"Object IDs with no group: {missing}")

    return {obj_id: frozenset(channels) for obj_id, channels in lookup.items()}


OBJECT_CHANNELS = _build_object_channels()


def channels_for(obj, gamestyle, theme, night):
    """
    Channels a parsed object contributes to every cell it covers

    Args:
        obj: A Level.Obj from one area's sliced objects
        gamestyle: level.gamestyle
        theme: The area's theme (level_map.theme)
        night: True if the area is in night mode (level_io.is_night)

    Returns:
        Frozenset of Channel (the object's own, its contents', and AIRBORNE if winged or parachuting)

    Raises:
        UnknownObjectError: If obj.id or obj.cid isn't a known ObjId
    """
    channels = set(element_channels(obj.id, obj.flag, gamestyle, theme, night))

    # Contents count where the container sits (mushroom in a ? Block, Goombas out of a pipe)
    if obj.cid != NO_CHILD:
        channels |= element_channels(obj.cid, obj.cflag, gamestyle, theme, night)

    if (obj.flag | obj.cflag) & (WINGS_BIT | PARACHUTE_BIT):
        channels.add(Channel.AIRBORNE)

    return frozenset(channels)


def element_channels(raw_id, flag, gamestyle, theme, night):
    """
    Channels for one object ID in its context, before contents and modifiers

    Args:
        raw_id: An ObjId member or a raw int (kaitai returns an int for values outside the enum)
        flag: The object's flag (or the container's cflag when raw_id is a cid)
        gamestyle: level.gamestyle
        theme: The area's theme
        night: True if the area is in night mode

    Returns:
        Frozenset of Channel

    Raises:
        UnknownObjectError: If raw_id isn't a known ObjId
    """
    try:
        obj_id = ObjId(raw_id)
    except ValueError:
        raise UnknownObjectError(f"Unknown object ID {raw_id}") from None

    if obj_id == ObjId.one_up and theme == Theme.overworld and night:
        return ROTTEN_MUSHROOM_CHANNELS
    if obj_id == ObjId.shoe_goomba and gamestyle in YOSHI_STYLES:
        return YOSHI_CHANNELS
    if flag & ALT_FORM_BIT and obj_id in ALT_FORM_CHANNELS:
        return ALT_FORM_CHANNELS[obj_id]

    return OBJECT_CHANNELS[obj_id]
