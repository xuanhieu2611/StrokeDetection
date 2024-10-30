from pathlib import Path
import pickle

def load_dataset(filename):
    with open(Path("..", "data", filename).with_suffix(".pkl"), "rb") as f:
        return pickle.load(f)
