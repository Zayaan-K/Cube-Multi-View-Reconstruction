import argparse
from pathlib import Path
import numpy as np
from projection.camera import Camera


def generateDataset(count=10000, seed=42):

    if count <= 0:
        raise ValueError("count must be positive")

    cameraA = Camera(position=(-1, 0, 0), angles=(0, 0, 0))
    cameraB = Camera(position=(1, 0, 0), angles=(0, 0, 0))
    rng = np.random.default_rng(seed)
    rows = []


    while len(rows) < count:
        point = rng.uniform((-2, -2, 3), (2, 2, 10))
        pointA = cameraA.project(point)
        pointB = cameraB.project(point)
        if pointA is None or pointB is None:
            continue
        if not all(0 <= u < 256 and 0 <= v < 256
                   for u, v in (pointA, pointB)):
            continue
        rows.append([*pointA, *pointB, *point])

    return np.asarray(rows, dtype=np.float64)


def saveDataset(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, rows, delimiter=",", header="uA,vA,uB,vB,x,y,z", comments="", fmt="%.12f")
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate multiview training data")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("dataset.csv"))
    args = parser.parse_args()

    rows = generateDataset(args.count, args.seed)
    output = saveDataset(rows, args.output)
    print(f"Saved {len(rows)} examples to {output}")
    print(f"Inputs shape: {rows[:, :4].shape}")
    print(f"Targets shape: {rows[:, 4:].shape}")


if __name__ == "__main__":
    main()
