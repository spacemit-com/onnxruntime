#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file replaces https URLs in deps.txt to local file paths. It runs after we download the dependencies from Azure
# DevOps Artifacts

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import wget
import time
import requests


@dataclass(frozen=True)
class Dep:
    name: str
    url: str
    sha1_hash: str


def parse_arguments():
    parser = argparse.ArgumentParser()
    # The directory that contains downloaded zip files
    parser.add_argument("--new_dir", required=False)
    parser.add_argument("--download", required=False, default="1", type=str)
    return parser.parse_args()


def download_file(url, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    try:
        response = requests.get(url, stream=True, verify=False)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return output_path
    except:
        return None


def main():
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))  # noqa: N806
    REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))  # noqa: N806

    args = parse_arguments()
    new_dir = None
    if args.new_dir:
        new_dir = Path(args.new_dir)
    else:
        BUILD_BINARIESDIRECTORY = os.environ.get("BUILD_BINARIESDIRECTORY")  # noqa: N806
        if BUILD_BINARIESDIRECTORY is None:
            raise NameError("Please specify --new_dir or set the env var BUILD_BINARIESDIRECTORY")
        new_dir = Path(BUILD_BINARIESDIRECTORY) / "deps"

    # Here we intentionally do not check if new_dir exists, because it might be used in a docker container instead.

    deps = []

    csv_file_path = Path(REPO_DIR) / "cmake" / "deps.txt"
    backup_csv_file_path = Path(REPO_DIR) / "cmake" / "deps.txt.bak"
    # prefer to use the backup file
    if backup_csv_file_path.exists():
        csv_file_path = backup_csv_file_path
    else:
        # Make a copy before modifying it
        print(f"Making a copy to {backup_csv_file_path!s}")
        shutil.copy(csv_file_path, backup_csv_file_path)

    print("Reading from %s" % str(csv_file_path))

    auto_download = bool(int(args.download))
    if auto_download:
        if not os.path.isdir(new_dir.as_posix()):
            os.makedirs(new_dir.as_posix())

    # Read the whole file into memory first
    with csv_file_path.open("r", encoding="utf-8") as f:
        depfile_reader = csv.reader(f, delimiter=";")
        for row in depfile_reader:
            if len(row) != 3:
                continue
            # Lines start with "#" are comments
            if row[0].startswith("#"):
                continue
            deps.append(Dep(row[0], row[1], row[2]))

    csv_file_path = Path(REPO_DIR) / "cmake" / "deps.txt"
    print(f"Writing to {csv_file_path!s}")
    # Write updated content back
    with csv_file_path.open("w", newline="", encoding="utf-8") as f:
        depfile_writer = csv.writer(f, delimiter=";")
        for dep in deps:
            if dep.url.startswith("https://"):
                new_url = new_dir / dep.url[8:].replace("/", "_")
                if auto_download:
                    if not os.path.exists(new_url.as_posix()):
                        try_count = 0
                        download_file_path = None
                        print("download file {}".format(new_url.as_posix()))
                        while download_file_path is None and try_count < 10:
                            try:
                                download_file_path = wget.download(dep.url, out=new_url.as_posix())
                            except Exception as e:
                                print("try download {} again. raise error: {}".format(dep.url, e))
                                download_file_path = download_file(dep.url, new_url.as_posix())
                                try_count += 1
                                time.sleep(10)
                    else:
                        print("exist file {}".format(new_url.as_posix()))
                depfile_writer.writerow([dep.name, new_url.as_posix(), dep.sha1_hash])
            else:
                # Write the original thing back
                depfile_writer.writerow([dep.name, dep.url, dep.sha1_hash])


if __name__ == "__main__":
    main()
