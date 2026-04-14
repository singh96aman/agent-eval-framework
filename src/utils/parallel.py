"""
Generic parallel execution utilities.

Reusable parallelization with batching, progress tracking, and error handling.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, TypeVar, Tuple
from dataclasses import field
from dataclasses import dataclass

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    workers: int = 4
    batch_size: int = 50
    rate_limit_delay: float = 0.1
    verbose: bool = True
    on_batch_complete: Optional[Callable[[List[Tuple[Any, Any, Optional[Exception]]]], None]] = field(default=None)


def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    config: Optional[ParallelConfig] = None,
    desc: str = "Processing",
) -> List[Tuple[T, R, Optional[Exception]]]:
    """
    Apply func to items in parallel with progress tracking.

    Args:
        func: Function to apply to each item
        items: List of items to process
        config: Parallel execution config
        desc: Description for progress messages

    Returns:
        List of (item, result, error) tuples.
        If successful, error is None. If failed, result is None.
    """
    if config is None:
        config = ParallelConfig()

    results = []
    total = len(items)
    processed = 0

    if config.verbose:
        print(f"\n{desc}: {total} items with {config.workers} workers")

    # Process in batches
    for batch_start in range(0, total, config.batch_size):
        batch = items[batch_start:batch_start + config.batch_size]
        batch_num = batch_start // config.batch_size + 1
        total_batches = (total + config.batch_size - 1) // config.batch_size

        if config.verbose and config.workers > 1:
            print(f"  Batch {batch_num}/{total_batches}...")

        batch_results = []
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            future_to_item = {
                executor.submit(func, item): item
                for item in batch
            }

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    batch_results.append((item, result, None))
                except Exception as e:
                    batch_results.append((item, None, e))

                processed += 1

        results.extend(batch_results)

        # Call batch complete callback for incremental processing
        if config.on_batch_complete and batch_results:
            config.on_batch_complete(batch_results)

        if config.verbose:
            print(f"  Batch {batch_num}/{total_batches} complete ({processed}/{total})")

        # Rate limiting between batches
        if batch_start + config.batch_size < total and config.rate_limit_delay > 0:
            time.sleep(config.rate_limit_delay)

    if config.verbose:
        success = sum(1 for _, _, e in results if e is None)
        failed = total - success
        print(f"  Complete: {success} success, {failed} failed")

    return results


def parallel_map_with_index(
    func: Callable[[int, T], R],
    items: List[T],
    config: Optional[ParallelConfig] = None,
    desc: str = "Processing",
) -> List[Tuple[int, T, R, Optional[Exception]]]:
    """
    Apply func(index, item) to items in parallel.

    Args:
        func: Function taking (index, item) and returning result
        items: List of items to process
        config: Parallel execution config
        desc: Description for progress messages

    Returns:
        List of (index, item, result, error) tuples
    """
    def indexed_func(indexed_item):
        idx, item = indexed_item
        return idx, func(idx, item)

    indexed_items = list(enumerate(items))
    raw_results = parallel_map(indexed_func, indexed_items, config, desc)

    # Unpack results
    results = []
    for (idx, item), result, error in raw_results:
        if error is None and result is not None:
            actual_idx, actual_result = result
            results.append((actual_idx, item, actual_result, None))
        else:
            results.append((idx, item, None, error))

    return results
