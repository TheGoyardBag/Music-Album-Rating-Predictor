"""Generate reproducible FICTIONAL albums, tags, and ratings. No scraped artist data."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate(path: Path, seed: int = 17):
    rng = np.random.default_rng(seed)
    vocabulary = ["ambient", "electronic", "jazz", "folk", "rock", "pop", "experimental",
                  "acoustic", "instrumental", "melodic", "atmospheric", "energetic", "warm",
                  "dark", "minimal", "dense", "lo-fi", "progressive", "dreamy", "rhythmic"]
    weights = dict(zip(vocabulary, rng.normal(0, 0.35, len(vocabulary))))
    rows = []
    for artist in range(120):
        artist_effect = rng.normal(0, 0.35)
        for album in range(5):
            tags = sorted(rng.choice(vocabulary, size=int(rng.integers(3, 8)), replace=False))
            interaction = 0.8 if {"ambient", "atmospheric"} <= set(tags) else 0
            rating = np.clip(6.0 + sum(weights[tag] for tag in tags) + interaction
                             + artist_effect + rng.normal(0, 0.4), 0, 10)
            rows.append({"album_id": f"demo-{artist:03d}-{album}",
                         "artist": f"Fictional Artist {artist:03d}",
                         "album": f"Synthetic Record {artist:03d}.{album}",
                         "tags": "|".join(tags), "rating": round(float(rating), 2)})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("data/demo_albums.csv"))
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    generate(args.output, args.seed)
    print(f"Wrote synthetic data to {args.output}")
