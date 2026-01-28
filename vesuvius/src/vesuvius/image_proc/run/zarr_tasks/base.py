"""Base class for zarr processing tasks.

Provides an abstract base class that defines the interface for all zarr tasks,
with hooks for argument parsing, preparation, work item generation, processing,
and finalization.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

import zarr
from tqdm import tqdm


@dataclass
class TaskConfig:
    """Base configuration for zarr tasks."""

    input_zarr: str
    output_zarr: Optional[str]
    num_workers: int
    inplace: bool = False
    level: Optional[int] = None


class ZarrTask(ABC):
    """Abstract base class for zarr processing tasks.

    Subclasses must implement:
        - add_arguments(): Add task-specific CLI arguments
        - from_args(): Create task instance from parsed arguments
        - prepare(): Initialize output arrays and task state
        - generate_work_items(): Yield work items for parallel processing
        - process_item(): Process a single work item (must be static/picklable)

    Optional overrides:
        - finalize(): Post-processing after all items are processed
        - run(): Override the entire execution flow if needed
        - use_executor: Set to True to use ProcessPoolExecutor with initializer
    """

    name: str = ""  # Set by @register_task decorator
    use_executor: bool = False  # Override to use ProcessPoolExecutor

    def __init__(self, config: TaskConfig):
        """Initialize the task with configuration.

        Args:
            config: Task configuration dataclass
        """
        self.config = config
        self.input_zarr: Optional[zarr.Array] = None
        self.output_zarr: Optional[zarr.Array] = None
        self._is_ome_zarr: bool = False
        self._levels_to_process: List[str] = []

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add task-specific arguments to the argument parser.

        Args:
            parser: The argument parser to add arguments to
        """
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> "ZarrTask":
        """Create a task instance from parsed command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            Configured task instance
        """
        pass

    @abstractmethod
    def prepare(self) -> None:
        """Prepare the task for execution.

        This method should:
        - Open input zarr arrays
        - Create output zarr arrays/groups
        - Initialize any task-specific state
        """
        pass

    @abstractmethod
    def generate_work_items(self) -> Iterable[Any]:
        """Generate work items for parallel processing.

        Yields:
            Work items to be passed to process_item()
        """
        pass

    @staticmethod
    @abstractmethod
    def process_item(args: Any) -> Any:
        """Process a single work item.

        This method must be static and picklable for multiprocessing.

        Args:
            args: Work item from generate_work_items()

        Returns:
            Result of processing (can be None)
        """
        pass

    def finalize(self) -> None:
        """Finalize the task after all items are processed.

        Override this method to perform post-processing, such as:
        - Building pyramid levels
        - Writing metadata
        - Printing summary statistics
        """
        pass

    def get_worker_initializer(self) -> Optional[Tuple[Callable, Tuple]]:
        """Get initializer function and args for ProcessPoolExecutor.

        Override this method when using ProcessPoolExecutor (use_executor=True)
        to provide a worker initialization function.

        Returns:
            Tuple of (initializer_function, initializer_args) or None
        """
        return None

    def run(self) -> None:
        """Execute the task.

        The default implementation:
        1. Calls prepare()
        2. Generates work items
        3. Processes items in parallel using Pool or ProcessPoolExecutor
        4. Calls finalize()

        Override this method for tasks that need custom execution flow.
        """
        self.prepare()

        work_items = list(self.generate_work_items())
        if not work_items:
            print("No work items to process")
            return

        print(f"Total items to process: {len(work_items)}")

        num_workers = self.config.num_workers

        if self.use_executor:
            # Use ProcessPoolExecutor with optional initializer
            init_info = self.get_worker_initializer()
            if init_info:
                initializer, initargs = init_info
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=initializer,
                    initargs=initargs,
                ) as executor:
                    results = list(
                        tqdm(
                            executor.map(self.process_item, work_items),
                            total=len(work_items),
                            desc=f"{self.name}",
                        )
                    )
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(
                        tqdm(
                            executor.map(self.process_item, work_items),
                            total=len(work_items),
                            desc=f"{self.name}",
                        )
                    )
        else:
            # Use multiprocessing.Pool
            with Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(self.process_item, work_items),
                        total=len(work_items),
                        desc=f"{self.name}",
                    )
                )

        self._results = results
        self.finalize()

    def _detect_ome_zarr(self, path: str) -> Tuple[bool, List[str]]:
        """Detect if path is an OME-Zarr and find resolution levels.

        Args:
            path: Path to zarr

        Returns:
            Tuple of (is_ome_zarr, list_of_levels)
        """
        opened = zarr.open(path, mode="r")

        if isinstance(opened, zarr.Group):
            all_levels = sorted(
                [k for k in opened.keys() if k.isdigit()], key=int
            )
            if all_levels:
                return True, all_levels

        return False, []

    def _get_input_array(self, level: Optional[str] = None) -> zarr.Array:
        """Get input zarr array, handling OME-Zarr levels.

        Args:
            level: Resolution level for OME-Zarr (e.g., "0")

        Returns:
            Zarr array
        """
        opened = zarr.open(self.config.input_zarr, mode="r")

        if isinstance(opened, zarr.Group):
            if level is not None:
                return opened[level]
            return opened["0"]

        return opened

    @classmethod
    def validate_args(cls, args: argparse.Namespace) -> None:
        """Validate parsed arguments.

        Override this method to add task-specific validation.

        Args:
            args: Parsed command-line arguments

        Raises:
            SystemExit: If validation fails
        """
        pass


def get_base_parser() -> argparse.ArgumentParser:
    """Create base argument parser with common options.

    Returns:
        ArgumentParser with common options
    """
    parser = argparse.ArgumentParser(
        description="Operate on zarr arrays with various tasks"
    )
    parser.add_argument(
        "input_zarr", type=str, help="Path to input zarr array"
    )
    parser.add_argument(
        "output_zarr",
        type=str,
        nargs="?",
        default=None,
        help="Path to output zarr array (not required if --inplace)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="threshold",
        help="Task to perform (use --list-tasks to see available tasks)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Perform operation in place (only for supported tasks)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Resolution level for OME-Zarr input (default: process level 0)",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit",
    )
    return parser


def make_task_config(args: argparse.Namespace) -> TaskConfig:
    """Create a TaskConfig from parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        TaskConfig instance
    """
    return TaskConfig(
        input_zarr=args.input_zarr,
        output_zarr=args.output_zarr,
        num_workers=args.num_workers if args.num_workers else cpu_count(),
        inplace=args.inplace,
        level=args.level,
    )
