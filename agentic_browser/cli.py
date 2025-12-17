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
    
    run_parser.add_argument(
        "--use-graph",
        action="store_true",
        default=False,
        help="Use new LangGraph multi-agent architecture",
    )
    
    run_parser.add_argument(
        "--langsmith",
        action="store_true",
        default=False,
        help="Enable LangSmith tracing (requires LANGCHAIN_API_KEY)",
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
        # Check if using new LangGraph architecture
        if getattr(args, 'use_graph', False):
            return run_graph_command(args, config, console)
        
        # Legacy agent mode
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


def run_graph_command(args: argparse.Namespace, config: AgentConfig, console: Console) -> int:
    """Execute using LangGraph multi-agent architecture.
    
    Args:
        args: Parsed arguments
        config: Agent configuration
        console: Rich console for output
        
    Returns:
        Exit code
    """
    from .graph import MultiAgentRunner, configure_tracing
    from .tools import BrowserTools
    from .os_tools import OSTools
    
    # Enable LangSmith tracing if requested
    if getattr(args, 'langsmith', False):
        configure_tracing(enabled=True)
    
    console.print("[bold cyan]🔄 Using LangGraph multi-agent architecture[/bold cyan]")
    console.print(f"[dim]Goal: {config.goal}[/dim]")
    console.print()
    
    # Initialize tools
    os_tools = OSTools(config)
    browser_tools = None  # Will be initialized when browser is needed
    
    # Create runner
    runner = MultiAgentRunner(
        config=config,
        os_tools=os_tools,
        enable_checkpointing=True,
    )
    
    try:
        # Stream execution for real-time output
        console.print("[bold]Starting multi-agent execution...[/bold]")
        console.print()
        
        for event in runner.stream(config.goal, config.max_steps):
            # Display agent activity
            for node_name, state_update in event.items():
                if node_name == "supervisor":
                    domain = state_update.get("current_domain", "")
                    console.print(f"[cyan]Supervisor[/cyan] → routing to [yellow]{domain}[/yellow]")
                else:
                    step = state_update.get("step_count", 0)
                    console.print(f"  [dim]Step {step}:[/dim] [green]{node_name}[/green] agent active")
                
                # Check for completion
                if state_update.get("task_complete"):
                    console.print()
                    console.print("[bold green]✓ Task completed![/bold green]")
                    answer = state_update.get("final_answer", "")
                    if answer:
                        console.print()
                        console.print("[bold]Final Answer:[/bold]")
                        console.print(answer)
                    return 0
                
                # Check for error
                error = state_update.get("error")
                if error:
                    console.print(f"[red]Error: {error}[/red]")
        
        console.print("\n[yellow]Execution ended (no explicit completion)[/yellow]")
        return 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[bold red]Graph execution error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
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
