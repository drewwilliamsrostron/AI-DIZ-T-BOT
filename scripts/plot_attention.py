"""Plot histogram of attention weights from a saved batch."""

import sys
import numpy as np
import matplotlib.pyplot as plt


def main(path: str) -> None:
    arr = np.load(path)
    plt.hist(arr.flatten(), bins=50, color="blue", alpha=0.7)
    plt.title("Attention weight distribution")
    plt.xlabel("weight")
    plt.ylabel("count")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_attention.py <file.npy>")
    else:
        main(sys.argv[1])
