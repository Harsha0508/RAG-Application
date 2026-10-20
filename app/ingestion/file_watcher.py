"""
File Watcher — Monitors the ingestion directory for new PDF files.

Uses watchdog to detect new files dropped into the ingestion folder.
Debounces events (waits for file write to complete) before triggering
the ingestion pipeline.
"""

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from app.core.constants import SUPPORTED_EXTENSIONS
from config import settings

logger = logging.getLogger(__name__)


class PDFIngestionHandler(FileSystemEventHandler):
    """Watchdog handler that triggers ingestion on new PDF files."""

    def __init__(self, vector_store=None, embedder=None) -> None:
        super().__init__()
        self._vector_store = vector_store
        self._embedder = embedder
        self._processing_lock = threading.Lock()
        self._debounce_seconds = settings.WATCHER_DEBOUNCE_SECONDS

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle new file creation events."""
        if event.is_directory:
            return
        self._handle_file(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events (e.g., drag-drop into folder)."""
        if event.is_directory:
            return
        self._handle_file(event.dest_path)

    def _handle_file(self, file_path: str) -> None:
        """Process a detected file if it's a supported PDF."""
        path = Path(file_path)

        # Only process supported file types
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug("Ignoring non-PDF file: %s", path.name)
            return

        # Ignore files in processed/ or failed/ subdirectories
        try:
            relative = path.relative_to(settings.INGESTION_DIR)
            if str(relative).startswith("processed") or str(relative).startswith("failed"):
                return
        except ValueError:
            pass

        # Debounce: wait for file write to complete
        logger.info("Detected new file: %s (waiting %.1fs for write completion)",
                     path.name, self._debounce_seconds)
        time.sleep(self._debounce_seconds)

        # Check file still exists (might have been moved/deleted)
        if not path.exists():
            logger.debug("File no longer exists: %s", path.name)
            return

        # Process in a thread to avoid blocking the watcher
        thread = threading.Thread(
            target=self._run_ingestion,
            args=(path,),
            name=f"ingest-{path.name}",
            daemon=True,
        )
        thread.start()

    def _run_ingestion(self, file_path: Path) -> None:
        """Run the ingestion pipeline for a file (thread-safe)."""
        with self._processing_lock:
            try:
                from app.ingestion.pipeline import ingest_pdf

                result = ingest_pdf(
                    file_path,
                    vector_store=self._vector_store,
                    embedder=self._embedder,
                )

                if result.success:
                    logger.info(
                        "Auto-ingested '%s': %d pages, %d chunks in %.1fs",
                        result.file_name, result.pages_processed,
                        result.chunks_added, result.processing_time_seconds,
                    )
                else:
                    logger.error(
                        "Auto-ingestion failed for '%s': %s",
                        result.file_name, result.error,
                    )

            except Exception as e:
                logger.error("Unexpected error during auto-ingestion of '%s': %s",
                             file_path.name, e, exc_info=True)


class FileWatcher:
    """
    Watches the ingestion directory for new PDF files.

    Usage:
        watcher = FileWatcher(vector_store, embedder)
        watcher.start()  # Non-blocking, runs in background
        ...
        watcher.stop()
    """

    def __init__(self, vector_store=None, embedder=None) -> None:
        self._handler = PDFIngestionHandler(vector_store, embedder)
        self._observer = Observer()
        self._watch_dir = str(settings.INGESTION_DIR)
        self._running = False

    def start(self) -> None:
        """Start watching the ingestion directory."""
        settings.ensure_dirs()

        self._observer.schedule(
            self._handler,
            self._watch_dir,
            recursive=False,
        )
        self._observer.start()
        self._running = True
        logger.info("File watcher started — monitoring '%s'", self._watch_dir)

    def stop(self) -> None:
        """Stop the file watcher."""
        if self._running:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._running = False
            logger.info("File watcher stopped")

    @property
    def is_running(self) -> bool:
        return self._running
