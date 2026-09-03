"""File-level adapters for optional learned or representation-based methods.

The core package intentionally ships protocol validation only.  Model code,
checkpoints, and their dependency environments stay outside this package.
"""

from .external import (
    ADAPTER_PROTOCOL_VERSION,
    AdapterRequest,
    AdapterRunResult,
    ExternalAdapter,
    ExternalAdapterError,
    run_external_adapter,
    validate_adapter_output,
)

__all__ = [
    "ADAPTER_PROTOCOL_VERSION",
    "AdapterRequest",
    "AdapterRunResult",
    "ExternalAdapter",
    "ExternalAdapterError",
    "run_external_adapter",
    "validate_adapter_output",
]
