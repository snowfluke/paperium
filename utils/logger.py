"""
Timed Logger Utility
Provides timestamped logging with elapsed time and delta tracking.
"""
import time
from rich.console import Console

console = Console()


class TimedLogger:
    """Logger that adds elapsed time and delta time to messages."""

    def __init__(self):
        self.start_time = time.time()
        self.last_step_time = self.start_time

    def log(self, message, style=""):
        """Print message with [elapsed | +delta] timestamp."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        delta = current_time - self.last_step_time
        self.last_step_time = current_time

        timestamp = f"[dim][{int(elapsed//60):02d}:{int(elapsed%60):02d} | +{delta:.1f}s][/dim]"
        console.print(f"{timestamp} {message}", style=style)

    def reset_step(self):
        """Reset the delta timer for next step."""
        self.last_step_time = time.time()
