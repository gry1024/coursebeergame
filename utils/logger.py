"""Append-only CSV logger for training/evaluation metrics."""

import csv
import os


class CSVLogger:
    """Open a CSV with a header and append one row at a time."""

    def __init__(self, path, fieldnames):
        """Open ``path`` in write mode (existing files are overwritten) and write the header."""
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row):
        """Write one row and flush so a crash does not lose buffered data."""
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the file handle."""
        self.file.close()