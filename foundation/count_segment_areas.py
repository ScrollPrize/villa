import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    """When run on the data server, generates a plot of the cumulative area segmented per scroll over time."""
    full_scrolls_dir = Path("/home/data/public/full-scrolls")
    for scroll_dir in sorted(full_scrolls_dir.iterdir()):
        scroll_name = scroll_dir.name
        paths_dir = list(scroll_dir.iterdir())[0] / "paths"

        areas = []
        times = []
        for path_dir in sorted(paths_dir.iterdir()):
            if "superseded" in path_dir.name:
                continue
            area_file = path_dir / "area_cm2.txt"
            if area_file.exists():
                with open(area_file, "r") as f:
                    area = float(f.read().strip())
                    areas.append(area)
                    # get last modified time of the area file
                    last_modified = area_file.stat().st_mtime
                    # convert to datetime.date
                    last_modified = datetime.datetime.fromtimestamp(last_modified)
                    times.append(last_modified)

        # sort both by date
        times, areas = zip(*sorted(zip(times, areas)))
        # get cumulative area
        areas = [
            sum(areas[:i + 1])
            for i in range(len(areas))
        ]

        # plot
        plt.plot(times, areas, label=scroll_name)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative area (cm^2)")
    plt.title("Segmented area per scroll")
    # save as .png
    plt.savefig(f"areas.png")


if __name__ == "__main__":
    main()
