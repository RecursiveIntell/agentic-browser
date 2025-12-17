"""
CLI for Agentic Browser.

Provides the command-line interface using argparse.
"""

import argparse
import sys
from typing import Optional

from rich.console import Console

from . import __version__
from .config import AgentConfig, DEFAULTS
from .agent import run_agent


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentic-browser",
        description="A Linux-first agentic browser runner that controls Chromium via Playwright.",
        epilog="""
Examples:
  # Get the title of a webpage
  agentic-browser run "Open example.com and tell me the title"
  
  # Search for something and navigate to results
  agentic-browser run "Search the web for Playwright and open the docs"
  
  # Use a specific LLM endpoint
  agentic-browser run "Check my email" --model-endpoint http://localhost:1234/v1 --model llama2
  
  # Run in headless mode with auto-approval
  agentic-browser run "Scrape product prices from amazon.com" --headless --auto-approve
  
  # Use a fresh browser profile
  agentic-browser run "Log into my account" --fresh-profile
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"agentic-browser {__version__}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the browser agent with a goal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    run_parser.add_argument(
        "goal",
        type=str,
        help="The goal to accomplish in natural language",
    )
    
    run_parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULTS["profile"],
        help=f"Browser profile name (default: {DEFAULTS['profile']})",
    )
    
    run_parser.add_argument(
        "--headless",
        action="store_true",
        default=DEFAULTS["headless"],
        help="Run browser in headless mode",
    )
    
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULTS["max_steps"],
        help=f"Maximum steps to execute (default: {DEFAULTS['max_steps']})",
    )
    
    run_parser.add_argument(
        "--model-endpoint",
        type=str,
        default=None,
        help=f"LLM API endpoint (default: {DEFAULTS['model_endpoint']})",
    )
    
    run_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"LLM model name (default: {DEFAULTS['model']})",
    )
    
    run_parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=DEFAULTS["auto_approve"],
        help="Auto-approve medium-risk actions (high-risk still requires approval)",
    )
    
    run_parser.add_argument(
        "--fresh-profile",
        action="store_true",
        default=False,
        help="Create a fresh browser profile (deletes existing)",
    )
    
    run_parser.add_argument(
        "--no-persist",
        action="store_true",
        default=False,
        help="Use a temporary profile (not persisted)",
    )
    
    run_parser.add_argument(
        "--enable-tracing",
        action="store_true",
        default=False,
        help="Enable Playwright tracing (saves trace.zip)",
    )
    
    run_parser.add_argument(
        "--gui-ipc",
        action="store_true",
        default=False,
        help="Use IPC mode for approval (for GUI subprocess)",
    )
    
    # GUI command
    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the graphical user interface",
    )
    
    return parser


def run_command(args: argparse.Namespace) -> int:
    """Execute the run command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    console = Console()
    
    # Build configuration
    config = AgentConfig.from_cli_args(
        goal=args.goal,
        profile=args.profile,
        headless=args.headless,
        max_steps=args.max_steps,
        model_endpoint=args.model_endpoint,
        model=args.model,
        auto_approve=args.auto_approve,
        fresh_profile=args.fresh_profile,
        no_persist=args.no_persist,
        enable_tracing=args.enable_tracing,
        gui_ipc=args.gui_ipc,
    )
    
    try:
        # Run the agent
        result = run_agent(config)
        
        # Print final status
        console.print()
        if result.success:
            console.print("[bold green]✓ Goal accomplished![/bold green]")
            return 0
        else:
            console.print(f"[bold red]✗ Goal not accomplished[/bold red]")
            if result.error:
                console.print(f"[red]Error: {result.error}[/red]")
            return 1
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        return 1


def gui_command() -> int:
    """Launch the GUI application.
    
    Returns:
        Exit code
    """
    from .gui import run_gui
    return run_gui()


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point.
    
    Args:
        argv: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == "run":
        return run_command(args)
    
    if args.command == "gui":
        return gui_command()
    
    # Unknown command
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
