"""Longest-output variant exporting the fixed registration entry points."""

from pathlib import Path
from runpy import run_path

# Execute the shared checks in a separate namespace for this policy's state.
_policy = run_path(str(Path(__file__).with_name("verify_custom_policy_shared_data.py")))


def initialize_policy():
    return _policy["_initialize_policy"]("longest", -1)


initialize_request = _policy["initialize_request_keyword"]
enable = _policy["enable"]
rank_longest = _policy["rank_longest"]
