"""
Network data loaders
    - One loader function per network, all reading the train/val/test Parquet files from build_subset.py
    - Each split is encoded once (encoding.levelToInput) and cached on disk, then streamed from the cache with np.memmap
    - The cache is rebuilt when it's missing, older than its Parquet file, or from an older CACHE_VERSION
    - Padded CNN gets every area padded to the max size, the other three get each level at its own size
    - Raw batches are zero-padded to the longest level in the batch (each level's true size is in its scalars)
    - Label is clears / attempts (0 to 1)
    - Levels the encoder can't handle (unknown clear condition) are skipped and counted
    - Run from the repo root so data_processing and level.py import (python -m networks.<network>)

Each batch is a dict
    - overworld: Overworld grid (batch, 12, 27, width)
    - subworldHorizontal: Horizontal subworld grid (batch, 12, 27, width), all zeros if the level has none
    - subworldVertical: Vertical subworld grid (batch, 12, height, 48), all zeros if the level has none
    - scalars: 150 values for the padded CNN (overworld 22, subworld 22, clear condition 99, gamestyle 5, timer 1, autoscroll speed 1), unnormalized
    - scalars: the same 150 plus 4 true (pre-padding) sizes in tiles for the other three, at SIZE_COLUMNS (subworld sizes are 0 if it has none)
    - label: Clear rate (batch,)
"""

# Imports
import math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from data_processing.encoding import NUM_CH, PADDED_HEIGHT, PADDED_WIDTH, levelToInput, parse




# ---------------------------------
# Constants
# ---------------------------------

# Files
DATA_DIR = Path(__file__).resolve().parent.parent / "data_processing" / "mm2_subset"
SPLITS = ("train", "val", "test")
GRIDS_FILE = "grids.bin"
INDEX_FILE = "index.npz"  # Written last, so its existence marks a finished cache
READ_BATCH = 256  # Parquet rows read at a time while building a cache

# Grids
GRID_KEYS = ("overworld", "subworldHorizontal", "subworldVertical")
HORIZONTAL_HEIGHT = 27  # Every horizontal area is this tall
VERTICAL_WIDTH = 48  # Every vertical area is this wide

# Scalars (key from levelToInput, size), in the order they're concatenated
SCALAR_PARTS = (
    ("overworldTheme", 10), ("overworldNight", 1), ("overworldLiquid", 6), ("overworldAutoscroll", 5),
    ("subworldTheme", 10), ("subworldNight", 1), ("subworldLiquid", 6), ("subworldAutoscroll", 5),
    ("clearCondition", 99), ("gamestyle", 5), ("timer", 1), ("autoscrollSpeed", 1)
)
BASE_SCALAR_COUNT = sum(size for _, size in SCALAR_PARTS)  # 150, all the padded CNN gets
SIZE_NAMES = ("overworldWidth", "overworldHeight", "subworldWidth", "subworldHeight")  # Tiles, raw networks only
SIZE_COLUMNS = {name: BASE_SCALAR_COUNT + offset for offset, name in enumerate(SIZE_NAMES)}  # Ex. scalars[:, SIZE_COLUMNS["overworldWidth"]]
RAW_SCALAR_COUNT = BASE_SCALAR_COUNT + len(SIZE_NAMES)  # 154

# Cache
CACHE_VERSION = 3  # Bump whenever the encoding or scalar layout changes (older caches rebuild on next load)

# Batching
BUCKET_POOL = 50  # Batches' worth of levels sorted by length together




# ---------------------------------
# emptyGrids
# ---------------------------------

def emptyGrids(padded):
    """
    All-zero stand-ins for the subworld grids a level doesn't have

    Args:
        padded: True for the padded CNN's full size, False for one zero step along the sequence axis
    """
    width = PADDED_WIDTH if padded else 1
    height = PADDED_HEIGHT if padded else 1
    return {
        "subworldHorizontal": np.zeros((NUM_CH, HORIZONTAL_HEIGHT, width), dtype=np.float16),
        "subworldVertical": np.zeros((NUM_CH, height, VERTICAL_WIDTH), dtype=np.float16)
    }




# ---------------------------------
# encodeRow
# ---------------------------------

def encodeRow(row, padded):
    """
    Encode one dataset row into its grids and scalar vector

    Args:
        row: Dataset row with level_data
        padded: True pads grids to the max area size (padded CNN), False keeps each area's own size

    Returns:
        (grids, scalars) where grids maps each GRID_KEYS name to a float16 array and scalars is a float32 vector
        (150 values, then the 4 level sizes in tiles when padded is False)
    """
    encoded = levelToInput(parse(row), pad=padded)
    empty = emptyGrids(padded)

    grids = {"overworld": np.asarray(encoded["overworld"], dtype=np.float16)}
    for key in ("subworldHorizontal", "subworldVertical"):
        grid = encoded[key]
        if grid is None or 0 in grid.shape:  # No subworld, or one with no area
            grids[key] = empty[key]
        else:
            grids[key] = np.asarray(grid, dtype=np.float16)

    parts = []
    for key, size in SCALAR_PARTS:
        value = encoded[key]
        parts.append(np.zeros(size, dtype=np.float32) if value is None else np.asarray(value, dtype=np.float32))

    if not padded:
        parts.append(np.array(levelSizes(encoded), dtype=np.float32))

    return grids, np.concatenate(parts)




# ---------------------------------
# levelSizes
# ---------------------------------

def levelSizes(encoded):
    """
    True (pre-padding) sizes in tiles of the overworld and subworld

    Args:
        encoded: Output of levelToInput with pad=False

    Returns:
        [overworld width, overworld height, subworld width, subworld height], subworld zeros if the level has none
    """
    overworld = encoded["overworld"].shape
    sizes = [overworld[2], overworld[1], 0, 0]

    for key in ("subworldHorizontal", "subworldVertical"):
        grid = encoded[key]
        if grid is not None and 0 not in grid.shape:
            sizes[2], sizes[3] = grid.shape[2], grid.shape[1]

    return sizes




# ---------------------------------
# getCacheDir
# ---------------------------------

def getCacheDir(dataDir, split, padded):
    """
    Folder holding one split's cache for one network family

    Args:
        dataDir: Folder with the split Parquet files
        split: "train", "val", or "test"
        padded: True for the padded CNN's cache, False for the variable-length one
    """
    return dataDir / "cache" / f"{split}_{'padded' if padded else 'raw'}"




# ---------------------------------
# buildCache
# ---------------------------------

def buildCache(dataDir, split, padded):
    """
    Encode a split's Parquet file into grids.bin (flat float16 grids) and index.npz (offsets, shapes, scalars, labels)

    Args:
        dataDir: Folder with the split Parquet files
        split: "train", "val", or "test"
        padded: True pads grids to the max area size (padded CNN), False keeps each area's own size

    Returns:
        The cache folder
    """
    cacheDir = getCacheDir(dataDir, split, padded)
    cacheDir.mkdir(parents=True, exist_ok=True)
    (cacheDir / INDEX_FILE).unlink(missing_ok=True)  # A half-built cache must not look finished

    offsets, shapes, scalars, labels = [], [], [], []
    position, skipped = 0, 0  # Position counts float16 values written so far
    scalarCount = BASE_SCALAR_COUNT if padded else RAW_SCALAR_COUNT

    parquetFile = pq.ParquetFile(dataDir / f"{split}.parquet")
    print(f"Encoding {split} ({'padded' if padded else 'raw'}), {parquetFile.metadata.num_rows} levels (one time only)", flush=True)

    with open(cacheDir / GRIDS_FILE, "wb") as gridsFile:
        for batch in parquetFile.iter_batches(batch_size=READ_BATCH, columns=["clears", "attempts", "level_data"]):
            for row in batch.to_pylist():
                try:
                    grids, scalarVector = encodeRow(row, padded)
                except ValueError:  # Unknown clear condition (UnknownObjectError is a ValueError too)
                    skipped += 1
                    continue

                offsets.append(position)
                shapes.append([grids[key].shape for key in GRID_KEYS])
                for key in GRID_KEYS:
                    gridsFile.write(grids[key].tobytes())
                    position += grids[key].size

                scalars.append(scalarVector)
                labels.append(row["clears"] / row["attempts"])

    np.savez(
        cacheDir / INDEX_FILE,
        offsets=np.array(offsets, dtype=np.int64),
        shapes=np.array(shapes, dtype=np.int64).reshape(-1, len(GRID_KEYS), 3),
        scalars=np.array(scalars, dtype=np.float32).reshape(-1, scalarCount),
        labels=np.array(labels, dtype=np.float32),
        version=np.array(CACHE_VERSION)
    )
    print(f"Cached {len(offsets)} {split} levels (skipped {skipped})", flush=True)

    return cacheDir




# ---------------------------------
# ensureCache
# ---------------------------------

def ensureCache(dataDir, split, padded):
    """
    Cache folder for a split, building it first if it's missing, older than the Parquet file, or an older version

    Args:
        dataDir: Folder with the split Parquet files
        split: "train", "val", or "test"
        padded: True for the padded CNN's cache, False for the variable-length one

    Raises:
        ValueError: If split isn't train, val, or test
        FileNotFoundError: If the split's Parquet file doesn't exist
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    parquetPath = dataDir / f"{split}.parquet"
    if not parquetPath.exists():
        raise FileNotFoundError(f"{parquetPath} not found (run data_processing/build_subset.py first)")

    cacheDir = getCacheDir(dataDir, split, padded)
    indexPath = cacheDir / INDEX_FILE
    if indexPath.exists() and indexPath.stat().st_mtime >= parquetPath.stat().st_mtime:
        with np.load(indexPath) as index:
            if "version" in index and int(index["version"]) == CACHE_VERSION:
                return cacheDir

    return buildCache(dataDir, split, padded)




# ---------------------------------
# LevelDataset
# ---------------------------------

class LevelDataset(Dataset):
    """One split's cached levels, read from disk one level at a time"""

    def __init__(self, cacheDir):
        index = np.load(cacheDir / INDEX_FILE)
        self.offsets = index["offsets"]
        self.shapes = index["shapes"]
        self.scalars = index["scalars"]
        self.labels = index["labels"]
        self.overworldWidths = self.shapes[:, 0, 2]  # Bucketing key
        self.gridsPath = cacheDir / GRIDS_FILE
        self.grids = None  # Opened on first read (each DataLoader worker needs its own handle)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if self.grids is None:
            self.grids = np.memmap(self.gridsPath, dtype=np.float16, mode="r")

        item = {}
        position = int(self.offsets[index])
        for key, shape in zip(GRID_KEYS, self.shapes[index]):
            size = int(np.prod(shape))
            grid = self.grids[position:position + size].reshape(shape)
            item[key] = torch.from_numpy(grid.astype(np.float32))
            position += size

        item["scalars"] = torch.from_numpy(self.scalars[index])
        item["label"] = torch.tensor(float(self.labels[index]))

        return item




# ---------------------------------
# padToLongest
# ---------------------------------

def padToLongest(grids):
    """
    Stack grids into one batch tensor, zero-padding each to the largest height and width

    Args:
        grids: List of (channels, height, width) tensors
    """
    height = max(grid.shape[1] for grid in grids)
    width = max(grid.shape[2] for grid in grids)
    batch = torch.zeros((len(grids), NUM_CH, height, width), dtype=torch.float32)
    for index, grid in enumerate(grids):
        batch[index, :, :grid.shape[1], :grid.shape[2]] = grid

    return batch




# ---------------------------------
# collate
# ---------------------------------

def collate(items):
    """
    Combine levels into one batch dict (see the module docstring for the keys)

    Args:
        items: List of LevelDataset items
    """
    batch = {key: padToLongest([item[key] for item in items]) for key in GRID_KEYS}
    batch["scalars"] = torch.stack([item["scalars"] for item in items])
    batch["label"] = torch.stack([item["label"] for item in items])

    return batch




# ---------------------------------
# BucketBatchSampler
# ---------------------------------

class BucketBatchSampler(Sampler):
    """
    Batches of similar-length levels (less zero padding per batch)
        - Shuffles levels, sorts each pool of BUCKET_POOL batches by length, cuts it into batches, then shuffles the batches
        - Reshuffles differently every epoch (seed + epoch)
    """

    def __init__(self, lengths, batchSize, shuffle, seed=0):
        self.lengths = np.asarray(lengths)
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batchSize)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        indices = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        poolSize = self.batchSize * BUCKET_POOL

        batches = []
        for start in range(0, len(indices), poolSize):
            pool = indices[start:start + poolSize]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            batches += [pool[first:first + self.batchSize].tolist() for first in range(0, len(pool), self.batchSize)]

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches




# ---------------------------------
# makeLoader
# ---------------------------------

def makeLoader(split, padded, batchSize, dataDir, numWorkers):
    """
    DataLoader over one split's cache (training shuffles, val and test don't)

    Args:
        split: "train", "val", or "test"
        padded: True pads every area to the max size, False keeps each level's own size (bucketed batches)
        batchSize: Levels per batch
        dataDir: Folder with the split Parquet files
        numWorkers: DataLoader worker processes
    """
    dataset = LevelDataset(ensureCache(Path(dataDir), split, padded))
    shuffle = split == "train"

    if padded:
        return DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, collate_fn=collate, num_workers=numWorkers)

    sampler = BucketBatchSampler(dataset.overworldWidths, batchSize, shuffle)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=numWorkers)




# ---------------------------------
# loadCnnPadding
# ---------------------------------

def loadCnnPadding(split, batchSize=32, dataDir=DATA_DIR, numWorkers=0):
    """
    Loader for the padded CNN (every area padded to 12x27x240 horizontal, 12x168x48 vertical)

    Args:
        split: "train", "val", or "test"
        batchSize: Levels per batch
        dataDir: Folder with the split Parquet files
        numWorkers: DataLoader worker processes
    """
    return makeLoader(split, True, batchSize, dataDir, numWorkers)




# ---------------------------------
# loadCnnPooling
# ---------------------------------

def loadCnnPooling(split, batchSize=32, dataDir=DATA_DIR, numWorkers=0):
    """
    Loader for the pooling CNN (each level at its own size, batches zero-padded to the longest level)

    Args:
        split: "train", "val", or "test"
        batchSize: Levels per batch
        dataDir: Folder with the split Parquet files
        numWorkers: DataLoader worker processes
    """
    return makeLoader(split, False, batchSize, dataDir, numWorkers)




# ---------------------------------
# loadGru
# ---------------------------------

def loadGru(split, batchSize=32, dataDir=DATA_DIR, numWorkers=0):
    """
    Loader for the GRU (each level at its own size, batches zero-padded to the longest level)

    Args:
        split: "train", "val", or "test"
        batchSize: Levels per batch
        dataDir: Folder with the split Parquet files
        numWorkers: DataLoader worker processes
    """
    return makeLoader(split, False, batchSize, dataDir, numWorkers)




# ---------------------------------
# loadLstm
# ---------------------------------

def loadLstm(split, batchSize=32, dataDir=DATA_DIR, numWorkers=0):
    """
    Loader for the LSTM (each level at its own size, batches zero-padded to the longest level)

    Args:
        split: "train", "val", or "test"
        batchSize: Levels per batch
        dataDir: Folder with the split Parquet files
        numWorkers: DataLoader worker processes
    """
    return makeLoader(split, False, batchSize, dataDir, numWorkers)