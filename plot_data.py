import sys
import matplotlib.pyplot as plt
from tools import (
    load_csv,
)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot_data.py <dataset_file_name.csv>",
              file=sys.stderr)
        sys.exit(1)
    data = load_csv(sys.argv[1])
    if data is None:
        print("Error: Failed to load dataset.", file=sys.stderr)
        sys.exit(1)

    mileages = [float(value) for value in data["km"].tolist()]
    prices = [float(value) for value in data["price"].tolist()]

    plt.scatter(mileages, prices)
    plt.title("Car Price by Mileage")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.show()