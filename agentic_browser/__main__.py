"""
Entry point for running as a module.

Usage: python -m agentic_browser run "GOAL"
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
