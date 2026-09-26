"""Forensic experts. Importing this package registers every available expert in hearsay.evidence.REGISTRY.

Each module is imported defensively so one broken or dependency-missing expert never takes down
the pipeline; failures are recorded in IMPORT_ERRORS and surfaced in reports.
"""
import importlib

MODULES = ["container", "compression", "spectral", "prosody", "environment", "splice", "speaker", "semantic"]
IMPORT_ERRORS = {}

for _m in MODULES:
    try:
        importlib.import_module(f"{__name__}.{_m}")
    except ModuleNotFoundError as e:
        if e.name != f"{__name__}.{_m}":
            IMPORT_ERRORS[_m] = repr(e)
    except Exception as e:  # pragma: no cover - defensive
        IMPORT_ERRORS[_m] = repr(e)
