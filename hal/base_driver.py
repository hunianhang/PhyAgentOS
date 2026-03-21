"""
hal/base_driver.py

Abstract base class for all robot body drivers.

Every hardware or simulation embodiment MUST subclass `BaseDriver` and
implement the four abstract methods. The HAL Watchdog loads a driver by
name and interacts with it exclusively through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDriver(ABC):
    """Contract that every robot body driver must fulfil."""

    @abstractmethod
    def get_profile_path(self) -> Path:
        """Return the filesystem path to this driver's EMBODIED.md profile."""

    @abstractmethod
    def load_scene(self, scene: dict[str, dict]) -> None:
        """Initialise the world from a scene dict parsed from ENVIRONMENT.md."""

    @abstractmethod
    def execute_action(self, action_type: str, params: dict) -> str:
        """Execute an atomic action and return a human-readable result string."""

    @abstractmethod
    def get_scene(self) -> dict[str, dict]:
        """Return the current world state for ENVIRONMENT.md updates."""

    def connect(self) -> bool:
        """Establish a connection to the embodiment if needed."""
        return True

    def disconnect(self) -> None:
        """Close the current connection if the driver maintains one."""

    def is_connected(self) -> bool:
        """Return whether the driver currently considers itself connected."""
        return True

    def health_check(self) -> bool:
        """Run a lightweight connection health check."""
        return self.is_connected()

    def get_runtime_state(self) -> dict[str, Any]:
        """Return optional runtime state such as nav or connection status."""
        return {}

    def close(self) -> None:
        """Release hardware resources. Override if needed."""

    def __enter__(self) -> "BaseDriver":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
