"""FIFO processing for the local, single-worker API server.

Duplicate requests for an active session share its result. Finished sessions
can be submitted again. The queue is in memory, so shutdown drains pending jobs.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock


class ProcessingQueue:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan")
        self._lock = Lock()
        self._pending: dict[tuple[str, str], Future] = {}

    def submit(self, key, work, *, on_queued):
        with self._lock:
            existing = self._pending.get(key)
            if existing is not None and not existing.done():
                return existing
            on_queued()
            future = self._executor.submit(work)
            self._pending[key] = future
        # Register outside the lock: an already completed Future calls this
        # synchronously, and cleanup needs the same lock.
        future.add_done_callback(lambda done: self._discard(key, done))
        return future

    def _discard(self, key, future):
        with self._lock:
            if self._pending.get(key) is future:
                del self._pending[key]

    def shutdown(self):
        self._executor.shutdown(wait=True)
