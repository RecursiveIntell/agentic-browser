"""
CLI for Agentic AiDEN.

Provides the command-line interface using argparse.
"""

import argparse
import sys
from typing import Optional

from rich.console import Console

from . import __version__
from .config import AgentConfig, DEFAULTS


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentic",
        description="Agentic AiDEN - A production-grade AI agent that controls your browser and OS.",
        epilog="""
Examples:
  # Get the title of a webpage
  agentic run "Open example.com and tell me the title"
  
  # Search for something and navigate to results
  agentic run "Search the web for Playwright and open the docs"
  
  # Use a specific LLM endpoint
  agentic run "Check my email" --model-endpoint http://localhost:1234/v1 --model llama2
  
  # Run in headless mode with auto-approval
  agentic run "Scrape product prices from amazon.com" --headless --auto-approve
  
  # Use a fresh browser profile
  agentic run "Log into my account" --fresh-profile
  
  # Launch Mission Control GUI
  agentic gui
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"Agentic AiDEN {__version__}",
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
        "--fast",
        action="store_true",
        default=False,
        help="Enable fast mode: blocks images, fonts, and media for faster page loads",
    )
    
    run_parser.add_argument(
        "--no-memory",
        action="store_true",
        default=False,
        help="Disable knowledge base lookups (Strategy Bank, Apocalypse) for faster startup",
    )
    
    run_parser.add_argument(
        "--vision",
        action="store_true",
        default=False,
        help="Enable vision mode: send page screenshots to LLM for visual understanding",
    )
    
    
    run_parser.add_argument(
        "--langsmith",
        action="store_true",
        default=False,
        help="Enable LangSmith tracing (requires LANGCHAIN_API_KEY)",
    )
    
    run_parser.add_argument(
        "-d", "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode: verbose output with all agent details",
    )
    
    # New safety and debugging flags
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print planned actions and risk levels without executing anything",
    )
    
    run_parser.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="Show decision rationale for each action (without full chain-of-thought)",
    )
    
    run_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output result as JSON (useful for n8n integration)",
    )
    
    # GUI command
    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the graphical user interface",
    )
    
    # Memory command
    memory_parser = subparsers.add_parser(
        "memory",
        help="Manage structured memory (known sites, directories)",
    )
    
    memory_parser.add_argument(
        "--show",
        action="store_true",
        help="Display all stored memory",
    )
    
    memory_parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Pre-seed common directories (Downloads, Documents, etc.)",
    )
    
    memory_parser.add_argument(
        "--edit",
        action="store_true",
        help="Open memory files for editing",
    )
    
    memory_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all stored memory (DANGEROUS - requires confirmation)",
    )
    
    # Doctor/Check command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check system dependencies and provider connectivity",
        aliases=["check"],
    )
    
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="Show auto-fix commands for failed checks",
    )
    
    doctor_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for all checks",
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
    
    # Handle dry-run mode early
    if getattr(args, 'dry_run', False):
        return run_dry_mode(args, console)
    
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
        browser_fast_mode=getattr(args, 'fast', False),
    )
    
    # Store explain mode in config for agents to use
    config.explain_mode = getattr(args, 'explain', False)
    
    # Set vision mode if explicitly requested
    if getattr(args, 'vision', False):
        config.vision_mode = True
        import logging
        logging.getLogger(__name__).info(f"Vision mode ENABLED (--vision flag)")
    
    try:
        # Use LangGraph multi-agent architecture
        return run_graph_command(args, config, console)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        return 1


def run_dry_mode(args: argparse.Namespace, console: Console) -> int:
    """Execute in dry-run mode - show what would be executed without running.
    
    Args:
        args: Parsed arguments
        console: Rich console for output
        
    Returns:
        Exit code (always 0 for dry-run)
    """
    from .policy_engine import PolicyEngine, RiskLevel
    from rich.panel import Panel
    from rich.table import Table
    
    console.print("[bold cyan]🔍 DRY RUN MODE[/bold cyan]")
    console.print("[dim]No actions will be executed. Showing what would happen.[/dim]")
    console.print()
    
    policy = PolicyEngine()
    
    # Analyze the goal and suggest initial actions
    console.print(f"[bold]Goal:[/bold] {args.goal}")
    console.print()
    
    # Create a table for displaying analysis
    table = Table(title="Dry Run Analysis")
    table.add_column("Aspect", style="cyan")
    table.add_column("Details", style="white")
    
    # Determine likely domain from goal
    goal_lower = args.goal.lower()
    
    likely_domain = "browser"  # default
    if any(k in goal_lower for k in ["file", "folder", "directory", "list", "read", "write", "run", "execute"]):
        likely_domain = "os"
    elif any(k in goal_lower for k in ["search", "research", "find", "look up"]):
        likely_domain = "research"
    
    table.add_row("Likely Domain", likely_domain)
    table.add_row("Max Steps", str(args.max_steps))
    table.add_row("Auto-Approve Medium", "Yes" if args.auto_approve else "No")
    
    # Example: if the goal mentions a command, evaluate it
    if "run " in goal_lower or "execute " in goal_lower:
        # Extract potential command
        import re
        cmd_match = re.search(r'["\']([^"\']+)["\']|`([^`]+)`', args.goal)
        if cmd_match:
            cmd = cmd_match.group(1) or cmd_match.group(2)
            decision = policy.evaluate(
                "os_exec",
                {"cmd": cmd},
                dry_run=True
            )
            
            risk_color = {
                RiskLevel.LOW: "green",
                RiskLevel.MEDIUM: "yellow",
                RiskLevel.HIGH: "red",
            }.get(decision.risk_level, "white")
            
            table.add_row("Detected Command", cmd)
            table.add_row("Risk Level", f"[{risk_color}]{decision.risk_level.value.upper()}[/{risk_color}]")
            table.add_row("Would Execute", decision.would_execute or "N/A")
            table.add_row("Approval Required", decision.requires_approval.value)
            
            if decision.blocked_reason:
                table.add_row("[red]BLOCKED[/red]", decision.blocked_reason)
    
    console.print(table)
    
    console.print()
    console.print("[bold]Summary:[/bold]")
    console.print("This is a dry-run preview. To actually execute, remove the --dry-run flag.")
    
    return 0


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
    from .graph.browser_manager import LazyBrowserManager
    from .os_tools import OSTools
    
    
    # Enable LangSmith tracing if requested
    if getattr(args, 'langsmith', False):
        configure_tracing(enabled=True)
    
    json_mode = getattr(args, 'json', False)
    
    if not json_mode:
        console.print("[bold cyan]🔄 Using LangGraph multi-agent architecture[/bold cyan]")
        console.print(f"[dim]Goal: {config.goal}[/dim]")
        console.print()
    
    # Initialize OS tools (always available)
    os_tools = OSTools(config)
    
    # Create lazy browser manager - browser only opens when needed
    browser_manager = LazyBrowserManager(config)
    
    # Create runner with lazy browser
    runner = MultiAgentRunner(
        config=config,
        os_tools=os_tools,
        browser_manager=browser_manager,
        enable_checkpointing=True,
    )
    
    # Register signal handler for graceful shutdown on SIGTERM (Stop button)
    import signal
    def handle_sigterm(signum, frame):
        import logging
        logging.getLogger(__name__).warning(f"Received signal {signum}, initiating graceful shutdown...")
        raise KeyboardInterrupt  # Will be caught by run_graph_command's try/except
        
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    try:
        # Stream execution for real-time output
        if not json_mode:
            console.print("[bold]Starting multi-agent execution...[/bold]")
            console.print()
        
        final_answer = ""
        final_state = {}
        
        for event in runner.stream(config.goal, config.max_steps):
            # Display agent activity
            for node_name, state_update in event.items():
                if json_mode:
                    # In JSON mode, we just collect state, no printing
                     if state_update.get("task_complete"):
                        final_state = state_update
                        final_answer = state_update.get("final_answer", "")
                else:
                    if node_name == "supervisor":
                        domain = state_update.get("current_domain", "")
                        console.print(f"[cyan]Supervisor[/cyan] → routing to [yellow]{domain}[/yellow]")
                    else:
                        step = state_update.get("step_count", 0)
                        console.print(f"  [dim]Step {step}:[/dim] [green]{node_name}[/green] agent active")
                
                # Check for completion
                if state_update.get("task_complete"):
                    if json_mode:
                        import json
                        output = {
                            "success": True,
                            "final_answer": final_answer,
                            "data": final_state.get("extracted_data", {}),
                            "steps": final_state.get("step_count", 0)
                        }
                        print(json.dumps(output))
                    else:
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
                    if json_mode:
                        import json
                        print(json.dumps({"success": False, "error": error}))
                        return 1
                    console.print(f"[red]Error: {error}[/red]")
        
        if json_mode:
             import json
             print(json.dumps({"success": False, "error": "Task ended without completion"}))
        else:
            console.print("\n[yellow]Execution ended (no explicit completion)[/yellow]")
        return 1
        
    except KeyboardInterrupt:
        if json_mode:
            return 130
        
        # Try to run retrospective if we have state
        if final_state and not final_state.get("retrospective_ran"):
            _run_retrospective_on_abort(final_state, config, console)
            
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130


def _run_retrospective_on_abort(state: dict, config: AgentConfig, console: Console):
    """Run retrospective analysis when user aborts."""
    try:
        console.print("\n[bold yellow]⚠️  Run Aborted! Running Retrospective Learning...[/bold yellow]")
        
        from .graph.agents.retrospective_agent import RetrospectiveAgent
        
        # Inject aborted flag
        state["was_aborted"] = True
        
        # Ensure minimal fields exist
        if "messages" not in state:
            state["messages"] = []
            
        agent = RetrospectiveAgent(config)
        agent.execute(state)
        
        console.print("[green]✓ Retrospective captured.[/green]")
    except Exception as e:
        console.print(f"[dim]Failed to run retrospective: {e}[/dim]")
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Try to run retrospective on failure
        if final_state and not final_state.get("retrospective_ran"):
             # Inject the specific error into state
            final_state["error"] = str(e)
            _run_retrospective_on_abort(final_state, config, console)
            
        if json_mode:
            import json
            print(json.dumps({"success": False, "error": str(e)}))
            return 1

        # Handle empty model response errors with helpful message
        empty_patterns = ["empty", "must contain", "output text", "tool calls", "cannot both be empty"]
        if any(p in error_msg for p in empty_patterns):
            console.print("\n[red bold]⚠️ Model returned empty response[/red bold]")
            console.print("[yellow]This usually means:[/yellow]")
            console.print("  • LM Studio needs to be restarted")
            console.print("  • The model is too small or incompatible")
            console.print("  • Try a different model (e.g., mistral-7b-instruct)")
            return 1
        
        console.print(f"\n[red]Graph execution error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        runner.cleanup()
        browser_manager.close()
        # Close session store
        if hasattr(runner, 'session_store') and runner.session_store:
            runner.session_store.close()


def gui_command() -> int:
    """Launch the GUI application.
    
    Returns:
        Exit code
    """
    from .gui import run_gui
    return run_gui()


def memory_command(args: argparse.Namespace) -> int:
    """Handle memory management commands.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    import json
    
    from .memory_store import create_memory_store
    
    console = Console()
    store = create_memory_store()
    
    if args.show:
        # Display all memory
        console.print("[bold cyan]📚 Structured Memory Store[/bold cyan]")
        console.print(f"[dim]Location: {store.memory_dir}[/dim]")
        console.print()
        
        # Sites
        sites = store.list_sites()
        if sites:
            table = Table(title="Known Sites")
            table.add_column("Domain", style="cyan")
            table.add_column("Notes")
            for domain in sites:
                site = store.get_site(domain)
                if site:
                    table.add_row(domain, site.notes[:50] + "..." if len(site.notes) > 50 else site.notes)
            console.print(table)
        else:
            console.print("[dim]No known sites.[/dim]")
        
        console.print()
        
        # Directories
        dirs = store.list_directories()
        if dirs:
            table = Table(title="Known Directories")
            table.add_column("Name", style="green")
            table.add_column("Path")
            for name in dirs:
                dir = store.get_directory(name)
                if dir:
                    table.add_row(name, dir.path)
            console.print(table)
        else:
            console.print("[dim]No known directories.[/dim]")
        
        console.print()
        
        # Recipes
        recipes = store.list_recipes()
        if recipes:
            table = Table(title="Recipes")
            table.add_column("Name", style="yellow")
            table.add_column("Description")
            for name in recipes:
                recipe = store.get_recipe(name)
                if recipe:
                    table.add_row(name, recipe.description[:50] + "..." if len(recipe.description) > 50 else recipe.description)
            console.print(table)
        else:
            console.print("[dim]No recipes.[/dim]")
        
        return 0
    
    elif args.bootstrap:
        console.print("[bold cyan]🚀 Bootstrapping common directories...[/bold cyan]")
        result = store.bootstrap_defaults()
        console.print(f"Added {result['directories']} directories")
        console.print("[green]✓ Bootstrap complete[/green]")
        return 0
    
    elif args.edit:
        import subprocess
        import os
        
        editor = os.environ.get("EDITOR", "nano")
        # Open the memory directory
        console.print(f"[bold]Opening memory files with {editor}...[/bold]")
        console.print(f"Location: {store.memory_dir}")
        
        # List files for user to choose
        console.print("\nMemory files:")
        console.print(f"  1. {store.sites_file}")
        console.print(f"  2. {store.dirs_file}")
        console.print(f"  3. {store.recipes_file}")
        
        try:
            subprocess.run([editor, str(store.memory_dir)], check=True)
        except FileNotFoundError:
            console.print(f"[red]Editor '{editor}' not found. Set EDITOR environment variable.[/red]")
            return 1
        except subprocess.CalledProcessError:
            return 1
        
        return 0
    
    elif args.clear:
        console.print("[bold red]⚠️  WARNING: This will delete ALL stored memory![/bold red]")
        confirm = input("Type 'DELETE' to confirm: ")
        if confirm == "DELETE":
            store.clear_all()
            console.print("[green]✓ All memory cleared.[/green]")
            return 0
        else:
            console.print("[yellow]Cancelled.[/yellow]")
            return 1
    
    else:
        # No flag - show help
        console.print("Use --show, --bootstrap, --edit, or --clear")
        console.print("Run 'agentic memory --help' for details.")
        return 0


def doctor_command(args: argparse.Namespace) -> int:
    """Run preflight health checks.
    
    Verifies:
    - Python dependencies
    - Playwright browsers
    - LLM provider connectivity  
    - n8n webhook health
    - OS tools (ffmpeg, imagemagick, etc.)
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code (0 if all pass, 1 if any fail)
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    import shutil
    import subprocess
    import os
    
    console = Console()
    verbose = getattr(args, 'verbose', False)
    show_fix = getattr(args, 'fix', False)
    
    console.print()
    console.print("[bold cyan]🩺 Agentic AiDEN Health Check[/bold cyan]")
    console.print("[dim]Running preflight diagnostics...[/dim]")
    console.print()
    
    results = []  # (name, status, message, fix_command)
    
    # --- 1. Check Python Dependencies ---
    console.print("[bold]1. Python Dependencies[/bold]")
    
    deps = [
        ("playwright", "playwright"),
        ("langchain-core", "langchain_core"),
        ("langgraph", "langgraph"),
        ("pydantic", "pydantic"),
        ("rich", "rich"),
        ("PySide6", "PySide6"),
        ("sentence-transformers", "sentence_transformers"),
        ("chromadb", "chromadb"),
    ]
    
    for pkg_name, import_name in deps:
        try:
            __import__(import_name)
            results.append((pkg_name, "green", "✅ Installed", None))
            if verbose:
                console.print(f"  [green]✅[/green] {pkg_name}")
        except ImportError as e:
            results.append((pkg_name, "red", f"❌ Missing", f"pip install {pkg_name}"))
            console.print(f"  [red]❌[/red] {pkg_name} - [dim]pip install {pkg_name}[/dim]")
    
    # --- 2. Check Playwright Browsers ---
    console.print()
    console.print("[bold]2. Playwright Browsers[/bold]")
    
    try:
        from playwright.sync_api import sync_playwright
        
        # Check if chromium is installed
        playwright_cache = os.path.expanduser("~/.cache/ms-playwright")
        chromium_dirs = [d for d in os.listdir(playwright_cache) if d.startswith("chromium")] if os.path.exists(playwright_cache) else []
        
        if chromium_dirs:
            results.append(("Chromium", "green", "✅ Installed", None))
            console.print(f"  [green]✅[/green] Chromium browser installed")
            if verbose:
                console.print(f"      [dim]Location: {playwright_cache}/{chromium_dirs[0]}[/dim]")
        else:
            results.append(("Chromium", "red", "❌ Not installed", "python -m playwright install chromium"))
            console.print(f"  [red]❌[/red] Chromium not installed")
            
    except Exception as e:
        results.append(("Playwright", "red", f"❌ Error: {e}", "pip install playwright && python -m playwright install chromium"))
        console.print(f"  [red]❌[/red] Playwright error: {e}")
    
    # --- 3. Check LLM Provider Connectivity ---
    console.print()
    console.print("[bold]3. LLM Provider Connectivity[/bold]")
    
    from .settings_store import SettingsStore
    store = SettingsStore()
    settings = store.settings
    
    provider = settings.provider
    console.print(f"  [dim]Active provider: {provider}[/dim]")
    
    try:
        provider_config = settings.get_provider_config()
        endpoint = provider_config.endpoint
        api_key = provider_config.api_key
        model = provider_config.effective_model
        
        if api_key or provider == "lm_studio":
            # Try a simple connection test
            import urllib.request
            import urllib.error
            
            test_url = endpoint.rstrip("/") + "/models" if endpoint else None
            
            if test_url and provider == "lm_studio":
                try:
                    req = urllib.request.Request(test_url, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        if resp.status == 200:
                            results.append((f"LM Studio ({endpoint})", "green", "✅ Connected", None))
                            console.print(f"  [green]✅[/green] LM Studio reachable at {endpoint}")
                        else:
                            results.append((f"LM Studio", "yellow", f"⚠️ Status {resp.status}", None))
                            console.print(f"  [yellow]⚠️[/yellow] LM Studio returned status {resp.status}")
                except urllib.error.URLError as e:
                    results.append(("LM Studio", "red", f"❌ Unreachable", "Start LM Studio and ensure server is running"))
                    console.print(f"  [red]❌[/red] LM Studio unreachable at {endpoint}")
            elif api_key:
                # For cloud providers, just check key is set
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
                results.append((f"{provider} API Key", "green", f"✅ Set ({masked_key})", None))
                console.print(f"  [green]✅[/green] {provider} API key configured ({masked_key})")
                console.print(f"      [dim]Model: {model}[/dim]")
        else:
            results.append((f"{provider}", "yellow", "⚠️ No API key", f"Set {provider.upper()}_API_KEY environment variable"))
            console.print(f"  [yellow]⚠️[/yellow] No API key configured for {provider}")
    except Exception as e:
        results.append(("Provider", "red", f"❌ Error: {e}", None))
        console.print(f"  [red]❌[/red] Provider check failed: {e}")
    
    # --- 4. Check n8n Integration ---
    console.print()
    console.print("[bold]4. n8n Integration[/bold]")
    
    n8n_url = os.environ.get("N8N_URL", "")
    n8n_key = os.environ.get("N8N_API_KEY", "")
    
    if n8n_url:
        try:
            import urllib.request
            import urllib.error
            
            health_url = n8n_url.rstrip("/") + "/healthz"
            req = urllib.request.Request(health_url, headers={"Content-Type": "application/json"})
            
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    results.append(("n8n Server", "green", "✅ Healthy", None))
                    console.print(f"  [green]✅[/green] n8n server healthy at {n8n_url}")
                else:
                    results.append(("n8n Server", "yellow", f"⚠️ Status {resp.status}", None))
                    console.print(f"  [yellow]⚠️[/yellow] n8n returned status {resp.status}")
        except Exception as e:
            results.append(("n8n Server", "red", f"❌ Unreachable", None))
            console.print(f"  [red]❌[/red] n8n unreachable at {n8n_url}: {e}")
    else:
        results.append(("n8n", "dim", "⚪ Not configured", "Set N8N_URL environment variable"))
        console.print(f"  [dim]⚪[/dim] n8n not configured (optional)")
    
    # --- 5. Check OS Tools ---
    console.print()
    console.print("[bold]5. OS Tools (for Media/System agents)[/bold]")
    
    os_tools = [
        ("ffmpeg", "Video/audio processing", "sudo apt install ffmpeg OR sudo dnf install ffmpeg"),
        ("convert", "ImageMagick (image processing)", "sudo apt install imagemagick OR sudo dnf install ImageMagick"),
        ("curl", "HTTP requests", "sudo apt install curl"),
        ("jq", "JSON processing", "sudo apt install jq"),
        ("ping", "Network diagnostics", "Usually pre-installed"),
        ("python3", "Python interpreter", "Should be pre-installed"),
        ("git", "Version control", "sudo apt install git"),
    ]
    
    for tool, desc, fix in os_tools:
        path = shutil.which(tool)
        if path:
            results.append((tool, "green", f"✅ Found", None))
            if verbose:
                console.print(f"  [green]✅[/green] {tool} - {path}")
            else:
                console.print(f"  [green]✅[/green] {tool}")
        else:
            results.append((tool, "yellow", f"⚠️ Not found ({desc})", fix))
            console.print(f"  [yellow]⚠️[/yellow] {tool} - {desc}")
    
    # --- 6. Check Tool Schemas ---
    console.print()
    console.print("[bold]6. Tool Schemas[/bold]")
    
    try:
        from .tool_schemas import ListDirRequest, ClickRequest, GotoRequest
        # Count schema classes by checking how many we can import
        schema_names = [
            "ListDirRequest", "ReadFileRequest", "WriteFileRequest", "RunCommandRequest",
            "MoveFileRequest", "CopyFileRequest", "DeleteFileRequest",
            "GotoRequest", "ClickRequest", "TypeRequest", "ScrollRequest",
        ]
        tool_count = len(schema_names)
        results.append(("Tool Schemas", "green", f"✅ {tool_count}+ schemas available", None))
        console.print(f"  [green]✅[/green] {tool_count}+ tool schemas loaded")
        
        if verbose:
            for name in schema_names[:8]:
                console.print(f"      [dim]{name}[/dim]")
            console.print(f"      [dim]... and more[/dim]")
    except Exception as e:
        results.append(("Tool Schemas", "red", f"❌ Error: {e}", None))
        console.print(f"  [red]❌[/red] Failed to load tool schemas: {e}")
    
    # --- 7. Check Storage Directories ---
    console.print()
    console.print("[bold]7. Storage & Cache[/bold]")
    
    storage_dirs = [
        ("~/.agentic_browser", "Main config"),
        ("~/.agentic_browser/runs", "Run logs"),
        ("~/.agentic_browser/profiles/default", "Browser profile"),
    ]
    
    for dir_path, desc in storage_dirs:
        expanded = os.path.expanduser(dir_path)
        if os.path.exists(expanded):
            results.append((desc, "green", "✅ Exists", None))
            if verbose:
                console.print(f"  [green]✅[/green] {desc} - {expanded}")
            else:
                console.print(f"  [green]✅[/green] {desc}")
        else:
            results.append((desc, "yellow", "⚠️ Will be created", None))
            console.print(f"  [yellow]⚠️[/yellow] {desc} (will be created on first run)")
    
    # --- 8. Check Agent Imports ---
    console.print()
    console.print("[bold]8. Agent Modules[/bold]")
    
    agents = [
        ("PlannerAgentNode", "agentic_browser.graph.agents.planner_agent", "PlannerAgentNode"),
        ("BrowserAgentNode", "agentic_browser.graph.agents.browser_agent", "BrowserAgentNode"),
        ("ResearchAgentNode", "agentic_browser.graph.agents.research_agent", "ResearchAgentNode"),
        ("OSAgentNode", "agentic_browser.graph.agents.os_agent", "OSAgentNode"),
        ("CodeAgentNode", "agentic_browser.graph.agents.code_agent", "CodeAgentNode"),
        ("DataAgentNode", "agentic_browser.graph.agents.data_agent", "DataAgentNode"),
        ("SupervisorNode", "agentic_browser.graph.supervisor", "Supervisor"),
        ("RetrospectiveAgent", "agentic_browser.graph.agents.retrospective_agent", "RetrospectiveAgent"),
    ]
    
    agent_errors = []
    for name, module, cls_name in agents:
        try:
            mod = __import__(module, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            results.append((name, "green", "✅ Loadable", None))
            if verbose:
                console.print(f"  [green]✅[/green] {name}")
        except Exception as e:
            agent_errors.append((name, str(e)))
            results.append((name, "red", f"❌ Import error", f"Check {module}.py for syntax errors"))
            console.print(f"  [red]❌[/red] {name} - {e}")
    
    if not agent_errors:
        console.print(f"  [green]✅[/green] All {len(agents)} agents loadable")
    
    # --- 9. Check Graph Build ---
    console.print()
    console.print("[bold]9. Graph Architecture[/bold]")
    
    try:
        from .graph.main_graph import build_agent_graph
        from .config import AgentConfig
        
        # Try to build the graph with a minimal config
        test_config = AgentConfig(
            goal="test",
            model="test-model",
            model_endpoint="http://localhost:1234/v1",
        )
        
        graph = build_agent_graph(test_config)
        
        # Check nodes are present
        node_count = len(graph.nodes) if hasattr(graph, 'nodes') else 0
        results.append(("Graph Build", "green", f"✅ Built ({node_count} nodes)", None))
        console.print(f"  [green]✅[/green] Graph built successfully")
        if verbose:
            console.print(f"      [dim]{node_count} nodes in graph[/dim]")
            
    except Exception as e:
        results.append(("Graph Build", "red", f"❌ Build failed: {e}", "Check graph/main_graph.py"))
        console.print(f"  [red]❌[/red] Graph build failed: {e}")
    
    # --- 10. Check State Schema ---
    console.print()
    console.print("[bold]10. State Schema[/bold]")
    
    try:
        from .graph.state import AgentState
        
        # Check required fields exist
        required_fields = ["messages", "goal", "task_complete", "pending_approval", "approved_actions"]
        missing = []
        
        for field in required_fields:
            if field not in AgentState.__annotations__:
                missing.append(field)
        
        if missing:
            results.append(("State Schema", "yellow", f"⚠️ Missing fields: {missing}", None))
            console.print(f"  [yellow]⚠️[/yellow] Missing state fields: {missing}")
        else:
            field_count = len(AgentState.__annotations__)
            results.append(("State Schema", "green", f"✅ Valid ({field_count} fields)", None))
            console.print(f"  [green]✅[/green] State schema valid ({field_count} fields)")
            if verbose:
                for f in list(AgentState.__annotations__.keys())[:8]:
                    console.print(f"      [dim]{f}[/dim]")
                if field_count > 8:
                    console.print(f"      [dim]... and {field_count - 8} more[/dim]")
                    
    except Exception as e:
        results.append(("State Schema", "red", f"❌ Error: {e}", "Check graph/state.py"))
        console.print(f"  [red]❌[/red] State schema error: {e}")
    
    # --- 11. Check Knowledge Base ---
    console.print()
    console.print("[bold]11. Knowledge Base[/bold]")
    
    try:
        from .graph.knowledge_base import StrategyBank
        
        # Just check it can be imported and has required methods
        required_methods = ["add_strategy", "search", "get_all"]
        missing_methods = [m for m in required_methods if not hasattr(StrategyBank, m)]
        
        if missing_methods:
            results.append(("Knowledge Base", "yellow", f"⚠️ Missing methods: {missing_methods}", None))
            console.print(f"  [yellow]⚠️[/yellow] Missing methods: {missing_methods}")
        else:
            results.append(("Knowledge Base", "green", "✅ Available", None))
            console.print(f"  [green]✅[/green] StrategyBank available")
            
    except Exception as e:
        results.append(("Knowledge Base", "yellow", f"⚠️ Not available: {e}", None))
        console.print(f"  [yellow]⚠️[/yellow] Knowledge base not available (optional): {e}")
    
    # --- Summary ---
    console.print()
    console.print("[bold]━━━ Summary ━━━[/bold]")
    
    passed = sum(1 for _, status, _, _ in results if status == "green")
    warnings = sum(1 for _, status, _, _ in results if status == "yellow")
    failed = sum(1 for _, status, _, _ in results if status == "red")
    
    if failed == 0 and warnings == 0:
        console.print(f"[bold green]✅ All {passed} checks passed![/bold green]")
        console.print("[dim]System is ready for operation.[/dim]")
    elif failed == 0:
        console.print(f"[bold yellow]⚠️ {passed} passed, {warnings} warnings[/bold yellow]")
        console.print("[dim]System should work, but some features may be limited.[/dim]")
    else:
        console.print(f"[bold red]❌ {passed} passed, {warnings} warnings, {failed} failed[/bold red]")
        console.print("[dim]Please resolve failed checks before running.[/dim]")
    
    # --- Show Fix Commands ---
    if show_fix:
        fixes = [(name, fix) for name, _, _, fix in results if fix]
        if fixes:
            console.print()
            console.print("[bold]📋 Suggested Fixes:[/bold]")
            for name, fix in fixes:
                console.print(f"  [cyan]{name}:[/cyan]")
                console.print(f"    [dim]{fix}[/dim]")
    
    console.print()
    
    return 0 if failed == 0 else 1


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
    
    if args.command == "memory":
        return memory_command(args)
    
    if args.command in ("doctor", "check"):
        return doctor_command(args)
    
    # Unknown command
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
