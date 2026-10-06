import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


if __package__:
    from .network import ReconstructionModel
else:
    from network import ReconstructionModel


def load_data(csv_path, batch_size, seed):

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    columns = ("uA", "vA", "uB", "vB", "x", "y", "z")
    if data.dtype.names != columns:
        raise ValueError("CSV header must be uA,vA,uB,vB,x,y,z")
    rows = np.column_stack([np.atleast_1d(data[name]) for name in columns])
    if len(rows) < 10 or not np.isfinite(rows).all():
        raise ValueError("not enough examples")

    #training data = 80%, val data = 10%, test = 10%
    indices = np.random.default_rng(seed).permutation(len(rows))
    
    train_end = int(0.8 * len(rows))
    train_rows = rows[indices[:train_end]]
    validation_end = int(0.9 * len(rows))
    validation_rows = rows[indices[train_end:validation_end]]
    test_rows = rows[indices[validation_end:]]
    
    input_mean = train_rows[:, :4].mean(axis=0)
    input_std = train_rows[:, :4].std(axis=0)
    target_mean = train_rows[:, 4:].mean(axis=0)
    target_std = train_rows[:, 4:].std(axis=0)
    input_std = np.where(input_std < 1e-8, 1.0, input_std)
    target_std = np.where(target_std < 1e-8, 1.0, target_std)

    def make_loader(subset, shuffle):
        inputs = torch.tensor((subset[:, :4] - input_mean) / input_std, dtype=torch.float32)
        targets = torch.tensor((subset[:, 4:] - target_mean) / target_std, dtype=torch.float32)
        return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size,shuffle=shuffle, num_workers=0,)

    normalization = {
        "input_mean": torch.tensor(input_mean, dtype=torch.float32),
        "input_std": torch.tensor(input_std, dtype=torch.float32),
        "target_mean": torch.tensor(target_mean, dtype=torch.float32),
        "target_std": torch.tensor(target_std, dtype=torch.float32),
    }

    return (make_loader(train_rows, True),make_loader(validation_rows, False),make_loader(test_rows, False),normalization,)


def evaluate(model, loader, loss_function):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            loss = loss_function(model(inputs), targets)
            total_loss += loss.item() * len(inputs)
    return total_loss / len(loader.dataset)


def train(args):
    torch.manual_seed(args.seed)
    train_loader, validation_loader, test_loader, normalization = load_data(
        args.data, args.batch_size, args.seed
    )
    model = ReconstructionModel() 
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    print(f"Training: {len(train_loader.dataset)} examples")
    print(f"Validation: {len(validation_loader.dataset)} examples")
    print(f"Test: {len(test_loader.dataset)} examples")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)

        train_loss = total_loss / len(train_loader.dataset)
        validation_loss = evaluate(model, validation_loader, loss_function)
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise RuntimeError("Nonfinite loss; check data or lower the learning rate")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs}: "
                  f"train MSE={train_loss:.6f}, val MSE={validation_loss:.6f}")


    model.load_state_dict(best_state)
    test_loss = evaluate(model, test_loader, loss_function)
    total_distance = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:

            errors = (model(inputs) - targets) * normalization["target_std"]
            total_distance += torch.linalg.vector_norm(errors, dim=1).sum().item()
    mean_distance = total_distance / len(test_loader.dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "normalization": normalization,
        "input_columns": ["uA", "vA", "uB", "vB"],
        "target_columns": ["x", "y", "z"],
        "best_epoch": best_epoch,
        "validation_mse": best_loss,
        "test_mse": test_loss,
        "test_mean_distance": mean_distance,
        "seed": args.seed,
    }, args.output)
    print(f"Best epoch: {best_epoch}")
    print(f"Test MSE (normalized): {test_loss:.6f}")
    print(f"Mean test 3D error: {mean_distance:.6f} world units")
    print(f"Saved model to {args.output.resolve()}")


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train multiview reconstruction")
    parser.add_argument("--data", type=Path, default=folder / "dataset.csv")
    parser.add_argument("--output", type=Path, default=folder / "reconstruction.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    train(args)


if __name__ == "__main__":
    main()
