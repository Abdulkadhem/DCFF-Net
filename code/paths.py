"""
The single place this project looks for data and checkpoints.

Nothing else in the codebase should contain an absolute path. Every script
imports from here, so a clone works without editing source files.

Environment variables, all optional:

    DCFF_DATA      directory holding the HuggingFace dataset cache
                   (default: <repo>/data/hf_cache)
    DCFF_WEIGHTS   directory holding checkpoints
                   (default: <repo>/weights)

The datasets are public and are downloaded on first use, so the default is
usually correct and nothing needs to be set.

    from paths import DATA_DIR, WEIGHTS_DIR, find_parquet, REPO_ID
"""
import os
import glob

# repository root, resolved from this file's location
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.abspath(os.environ.get("DCFF_DATA",
                                          os.path.join(ROOT, "data", "hf_cache")))
WEIGHTS_DIR = os.path.abspath(os.environ.get("DCFF_WEIGHTS",
                                             os.path.join(ROOT, "weights")))

# point the HuggingFace libraries at the same cache, unless the user set it
os.environ.setdefault("HF_HOME", DATA_DIR)

# dataset short name -> HuggingFace repository id
REPO_ID = {
    "levir": "ericyu/LEVIRCD_Cropped_256",
    "egy":   "ericyu/EGY_BCD",
    "clcd":  "ericyu/CLCD_Cropped_256",
    "sysu":  "ericyu/SYSU_CD",
    "gvlm":  "ericyu/GVLM_Cropped_256",
    "dsifn": "EVER-Z/torchange_dsifn-cd",
}


def _cache_slug(repo_id):
    """HuggingFace stores 'org/name' as 'datasets--org--name'."""
    return "datasets--" + repo_id.replace("/", "--")


def find_parquet(dataset, split, required=True):
    """
    Locate the cached parquet shards of one split, searching the configured
    cache first and then the user's default HuggingFace cache.

    Returns a sorted list of paths. Reading the parquet directly is much
    faster than going through `datasets` for the evaluation scripts, and it
    avoids re-downloading when the cache is already populated.
    """
    if dataset not in REPO_ID:
        raise KeyError("unknown dataset %r; known: %s"
                       % (dataset, ", ".join(sorted(REPO_ID))))
    slug = _cache_slug(REPO_ID[dataset])
    roots = [DATA_DIR,
             os.path.join(os.path.expanduser("~"), ".cache", "huggingface")]

    hits = []
    for r in roots:
        hits += glob.glob(os.path.join(r, "**", slug, "**", "%s-*.parquet" % split),
                          recursive=True)
        # some splits are stored as 'validation-*'
        if split == "val":
            hits += glob.glob(os.path.join(r, "**", slug, "**", "validation-*.parquet"),
                              recursive=True)
    hits = sorted(set(hits))

    if not hits and required:
        raise FileNotFoundError(
            "no cached parquet for %s/%s.\n"
            "Searched:\n  %s\n"
            "Populate the cache once with:\n"
            "  python -c \"import datasets; datasets.load_dataset('%s')\""
            % (dataset, split, "\n  ".join(roots), REPO_ID[dataset]))
    return hits


def checkpoint(name):
    """Resolve a checkpoint name or path against WEIGHTS_DIR."""
    if os.path.isabs(name) or os.path.exists(name):
        return name
    return os.path.join(WEIGHTS_DIR, name)
