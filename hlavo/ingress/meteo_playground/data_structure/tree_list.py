#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag

BASE_URL = "https://opendata.chmi.cz/"  # root of the tree
FILES_OUTPUT = "chmi_files.txt"
DIR_COUNTS_OUTPUT = "chmi_dir_counts.txt"


def list_files_and_counts(base_url: str):
    """
    Recursively list all files reachable from an Apache-style index root and
    count how many files are in each directory's subtree.

    :param base_url: Root URL of the HTTP directory listing
    :return: (files, dir_counts, root_total)
             - files: list of file URLs (strings)
             - dir_counts: dict {dir_url: total_files_in_subtree}
             - root_total: total number of files under base_url
    """
    if not base_url.endswith("/"):
        base_url += "/"

    visited_dirs = set()
    files = []
    dir_counts = {}

    def walk_url(url: str) -> int:
        """
        Recursively walk a directory URL.

        Returns the total number of files in this directory's subtree.
        """
        # Normalize directory URL to always end with '/'
        if not url.endswith("/"):
            url += "/"

        if url in visited_dirs:
            return 0
        visited_dirs.add(url)

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            dir_counts[url] = 0
            return 0

        soup = BeautifulSoup(resp.text, "html.parser")

        total_files_here = 0

        for a in soup.find_all("a"):
            href = a.get("href")
            if not href:
                continue

            # Strip URL fragment
            href, _ = urldefrag(href)

            # Skip parent directory and empty refs
            if href in ("", "/", "../"):
                continue

            full = urljoin(url, href)

            # Directory (ends with '/')
            if href.endswith("/"):
                total_files_here += walk_url(full)
            else:
                # File
                files.append(full)
                total_files_here += 1

        # Store total for this directory
        dir_counts[url] = total_files_here
        return total_files_here

    root_total = walk_url(base_url)
    return files, dir_counts, root_total


if __name__ == "__main__":
    all_files, dir_counts, root_total = list_files_and_counts(BASE_URL)

    # Write all file URLs to a file
    with open(FILES_OUTPUT, "w", encoding="utf-8") as f_out:
        for url in all_files:
            f_out.write(url + "\n")

    # Write directory URL + total number of files in its subtree
    # Example line: "https://opendata.chmi.cz/ 12345"
    with open(DIR_COUNTS_OUTPUT, "w", encoding="utf-8") as f_out:
        for dir_url, count in sorted(dir_counts.items()):
            f_out.write(f"{dir_url} {count}\n")

    print(f"Total files found under {BASE_URL}: {root_total}")
    print(f"File list written to: {FILES_OUTPUT}")
    print(f"Directory counts written to: {DIR_COUNTS_OUTPUT}")
