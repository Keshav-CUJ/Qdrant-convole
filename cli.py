import typer
import uuid
import os
import time
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from src.graph.workflow import build_graph

# --- SETUP ---
app = typer.Typer()
console = Console()

state = {
    "user_id": "guest", 
    "thread_id": None, 
    "graph": None
}

def get_new_thread_id():
    return str(uuid.uuid4())

# --- 1. CLEAN MODE (For Standard Users) ---
def run_clean_mode(input_payload, config):
    """
    Hides the raw logs behind a spinner for a clean experience.
    """
    response_text = ""
    # We use a spinner to HIDE the raw prints and show "Thinking..."
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Agent is analyzing...[/bold green]"),
        transient=True,
    ) as progress:
        progress.add_task("", total=None)
        
        # We redirect stdout to devnull temporarily so the raw prints don't break the spinner
        # (Or we just ignore them if they print above the spinner)
        try:
            for event in state["graph"].stream(input_payload, config=config):
                for k, v in event.items():
                    if k == "write_answer":
                        response_text = v['messages'][-1].content
        except Exception as e:
            console.print(f"[bold red]❌ Error:[/bold red] {e}")
            return None
    return response_text

# --- 2. RAW LOGS MODE (For Judges/Debug) ---
def run_raw_mode(input_payload, config):
    """
    Runs the graph WITHOUT any UI wrappers. 
    This allows the raw 'print()' statements from the nodes to appear directly.
    """
    print("\n" + "="*40)
    print(f"🚀 STARTING PIPELINE for Thread: {config['configurable']['thread_id']}")
    print("="*40 + "\n")
    
    response_text = ""
    try:
        # Just run the stream! The nodes will print their own logs.
        for event in state["graph"].stream(input_payload, config=config):
            for k, v in event.items():
                if k == "write_answer":
                    response_text = v['messages'][-1].content
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return None
        
    print("\n" + "="*40)
    print("✅ PIPELINE FINISHED")
    print("="*40 + "\n")
    return response_text

# --- MAIN COMMAND ---
@app.command()
def chat(
    user: Optional[str] = typer.Option(None, "--user", "-u", help="Initial user ID. If empty, asks interactively."),
    logs: bool = typer.Option(False, "--logs", "-l", help="Show RAW execution logs (No UI wrappers)"),
):
    """
    Starts the Election Agent CLI.
    """
    console.print(Panel.fit("[bold blue]🗳️  MISINFORMATION DETECTION SYSTEM[/bold blue]", subtitle="v4.0 (Raw Logs)"))

    # 1. HANDLE LOGIN
    if user is None:
        console.print("[yellow]Authentication Required[/yellow]")
        user = Prompt.ask("👤 Enter Username")
    
    # 2. INITIALIZE
    console.print(f"[dim]Loading AI Models for {user}...[/dim]")
    state["graph"] = build_graph()
    state["user_id"] = user
    state["thread_id"] = get_new_thread_id()
    
    console.print(f"✅ Logged in as: [bold cyan]{state['user_id']}[/bold cyan]")
    if logs:
        console.print("[yellow]⚠️  RAW LOGS ENABLED: You will see internal node outputs.[/yellow]")
    
    console.print("[dim]Type '/help' for commands. Type 'quit' to exit.[/dim]\n")

    # 3. CHAT LOOP
    while True:
        user_input = Prompt.ask(f"[bold green]{state['user_id']}[/bold green]")
        
        if not user_input.strip(): continue

        # --- COMMANDS ---
        if user_input.lower() == "/help":
            console.print(Panel("""
            [bold]Available Commands:[/bold]
            /new           - Start a fresh conversation
            /login <name>  - Switch user
            /image <path>  - Attach image
            quit / exit    - Close app
            """, title="Help Menu", border_style="green"))
            continue

        if user_input.lower() in ["quit", "exit"]:
            console.print("[bold red]👋 Exiting...[/bold red]")
            break
            
        elif user_input.lower() == "/new":
            state["thread_id"] = get_new_thread_id()
            console.print(f"[yellow]✨ New Session Started (ID: {state['thread_id']})[/yellow]")
            continue

        elif user_input.lower().startswith("/login"):
            parts = user_input.split(" ")
            if len(parts) < 2:
                console.print("[red]⚠️ Usage: /login <username>[/red]")
                continue
            new_user = parts[1]
            state["user_id"] = new_user
            state["thread_id"] = get_new_thread_id()
            console.print(f"[cyan]👤 Switched to user: {new_user}[/cyan]")
            continue

        # IMAGE HANDLING
        image_path = None
        clean_input = user_input

        if user_input.lower().startswith("/image"):
            parts = user_input.split(" ", 2)
            if len(parts) < 2:
                console.print("[red]⚠️ Usage: /image <path> [question][/red]")
                continue
            path = parts[1]
            if not os.path.exists(path):
                console.print(f"[red]❌ Image not found: {path}[/red]")
                continue
            image_path = path
            console.print(f"[blue]📎 Image Attached: {path}[/blue]")
            if len(parts) > 2:
                clean_input = parts[2]
            else:
                clean_input = Prompt.ask("[bold green]Question about this image[/bold green]")

        # --- EXECUTION ---
        config = {
            "configurable": {
                "user_id": state["user_id"],
                "thread_id": state["thread_id"]
            }
        }
        
        payload = {
            "messages": [("user", clean_input)], 
            "current_image_path": image_path 
        }

        # EXECUTE BASED ON MODE
        if logs:
            # RAW MODE: Just run it. The nodes will print to the console.
            response = run_raw_mode(payload, config)
        else:
            # CLEAN MODE: Hide logs behind spinner
            response = run_clean_mode(payload, config)
        
        if response:
            console.print("\n[bold purple]🤖 Agent Verdict:[/bold purple]")
            console.print(Panel(Markdown(response), border_style="purple"))
            console.print("\n")

if __name__ == "__main__":
    app()