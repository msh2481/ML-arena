import subprocess
import sys
import time
from datetime import datetime

from loguru import logger


def run_arena() -> None:
    logger.info("Starting arena.py")
    try:
        result = subprocess.run(
            ["python", "arena.py"], check=True, capture_output=True, text=True
        )
        logger.info("Run completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running arena.py: {e}")
        if e.stdout:
            logger.error(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


def main() -> None:
    logger.info("Arena runner started")
    try:
        while True:
            run_arena()
            logger.info("Waiting 1 second before next run")
            time.sleep(1)  # Small delay between runs
    except KeyboardInterrupt:
        logger.info("Arena runner stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
