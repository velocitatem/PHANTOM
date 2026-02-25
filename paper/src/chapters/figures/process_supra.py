import pandas as pd
import json
import numpy as np
import sys
import os


def process_supra(input_file, output_file):
    print(f"Processing {input_file} -> {output_file}")

    # Read the CSV
    try:
        # The CSV has a weird format: "Step","giddy-deluge-6 - distributions/prices"
        # The header is on line 1.
        # Let's verify the file content format first effectively.
        # The previous read showed standard CSV with quoted fields.
        df = pd.read_csv(input_file, quotechar='"', skipinitialspace=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Prepare for re-binning
    # We need a common set of bins to plot a heatmap (surface)
    # First, let's collect all data to determine range
    all_min = float("inf")
    all_max = float("-inf")

    parsed_data = []

    # The column names might be dynamic, so let's rely on indices
    # Column 0: Step
    # Column 1: JSON blob

    for index, row in df.iterrows():
        try:
            step = int(row.iloc[0])
            json_str = row.iloc[1]

            # Cleaning potential double quotes issue if pandas didn't catch it perfect
            # but pandas read_csv usually handles standard CSV escaping well.

            data = json.loads(json_str)

            bins = np.array(data["bins"])
            values = np.array(data["values"])

            # Update global range
            if bins.min() < all_min:
                all_min = bins.min()
            if bins.max() > all_max:
                all_max = bins.max()

            parsed_data.append({"step": step, "bins": bins, "values": values})
        except Exception as e:
            print(f"Skipping row {index} due to error: {e}")
            continue

    if not parsed_data:
        print("No data parsed.")
        return

    print(f"Found {len(parsed_data)} steps. Range: {all_min} to {all_max}")

    # Define common grid
    # Y-axis (Price)
    # Using 100 bins for resolution
    y_bins_edges = np.linspace(all_min, all_max, 101)
    y_bin_centers = (y_bins_edges[:-1] + y_bins_edges[1:]) / 2

    # Open output file
    with open(output_file, "w") as f:
        # PGFPlots 3D format often prefers no header or a specific header.
        # We will use named columns.
        f.write("step,price,density\n")

        # Sort by step to ensure correct mesh ordering
        parsed_data.sort(key=lambda x: x["step"])

        for item in parsed_data:
            step = item["step"]
            original_bins = item["bins"]
            original_values = item["values"]

            # Re-binning logic
            current_new_hist = np.zeros(len(y_bin_centers))

            for i, (new_start, new_end) in enumerate(
                zip(y_bins_edges[:-1], y_bins_edges[1:])
            ):
                val = 0.0
                # This inner loop is slightly inefficient O(N*M) but N~3000, M~100 -> 300k ops, totally fine.
                for j in range(len(original_values)):
                    b_start = original_bins[j]
                    # Handle cases where values array might be 1 shorter than bins (histogram edges vs centers)
                    # The provided JSON has "bins" array larger than "values" by 1 usually for edges.
                    if j + 1 >= len(original_bins):
                        break

                    b_end = original_bins[j + 1]
                    b_width = b_end - b_start

                    if b_width <= 0:
                        continue

                    # Calculate overlap
                    overlap_start = max(new_start, b_start)
                    overlap_end = min(new_end, b_end)
                    overlap = max(0, overlap_end - overlap_start)

                    if overlap > 0:
                        # Add proportional count
                        val += original_values[j] * (overlap / b_width)

                current_new_hist[i] = val

            # Write row to file for this step
            for price, density in zip(y_bin_centers, current_new_hist):
                # PGFPlots expects x y z
                f.write(f"{step},{price},{density}\n")

            # Add a blank line for PGFPlots matrix format (essential for 'mesh' or 'surf')
            f.write("\n")


if __name__ == "__main__":
    # Resolve relative paths relative to where script is run, or use absolute
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "supra.csv")
    output_path = os.path.join(base_dir, "supra_data.csv")

    process_supra(input_path, output_path)
