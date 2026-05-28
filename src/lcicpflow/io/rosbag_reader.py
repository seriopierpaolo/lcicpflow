"""Optional ROS/rosbag adapter placeholder.

The core ICP-Flow pipeline intentionally has no ROS dependency. Add extraction
logic here when ROS message formats are known for a specific deployment.
"""

from __future__ import annotations


def rosbag_not_available() -> None:
    raise ImportError("ROS/rosbag support is optional and not installed in the core package")
