"""Small shared utilities with no benchmark-stage dependencies."""

from trajsimbench.utils.hardware import hardware_info
from trajsimbench.utils.paths import cache_dir, project_root
from trajsimbench.utils.seeding import seed_everything

__all__ = ["cache_dir", "hardware_info", "project_root", "seed_everything"]
