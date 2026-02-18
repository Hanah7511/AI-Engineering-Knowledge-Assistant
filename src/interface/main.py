"""
 AI ENGINEERING KNOWLEDGE ASSISTANT - PRODUCTION CLI
==================================================================================
This is the main entry point for your Elite Production RAG System.
It initializes the EnhancedProductionRAG class and provides an interactive CLI.

Features:
- ✅ Path resolution for src/ modules
- ✅ Interactive command-line interface
- ✅ Mode switching (similarity/mmr)
- ✅ Cache management
- ✅ ROI analytics
- ✅ Graceful error handling
==================================================================================
"""

import sys
import os

# ================= CHECK FOR RICH DISPLAY =================
try:
    from rich.console import Console
    from rich import print
    USE_RICH = True
except ImportError:
    USE_RICH = False


# ================= PATH FIX (VERY IMPORTANT) =================
# This ensures Python can find src/ modules when running from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

# ================= IMPORT YOUR RAG BRAIN =================
try:
    from src.rag.rag_pipeline import EnhancedProductionRAG
except ImportError as e:
    print(f" Import Error: {e}")
    print("Please ensure you're running from the correct directory and rag_pipeline.py exists.")
    print("Expected structure:")
    print("  project_root/")
    print("  ├── src/")
    print("  │   └── rag/")
    print("  │       └── rag_pipeline.py")
    print("  └── main.py (this file)")
    sys.exit(1)


def print_banner():
    """Print beautiful startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                      AI ENGINEERING KNOWLEDGE ASSISTANT                      ║
║                      Production-Grade  RAG System                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Architecture:                                                                ║
║   • Local Embeddings (HuggingFace) - FREE                                    ║
║   • Chroma Vector Database - LOCAL                                           ║
║   • GPT-4o-mini - PAY-PER-USE                                                ║
║   • Hybrid Retriever - SIMILARITY | MMR | ENSEMBLE (Hybrid)  (future)        ║
║   • Retry Logic - EXPONENTIAL BACKOFF                                        ║
║   • Fallback - ZERO COST                                                     ║
║   • Caching - PERFORMANCE BOOST                                              ║
║   • Metrics - BUSINESS VALUE                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Commands:                                                                    ║
║   • Ask any technical question                                               ║
║   • 'exit' or 'quit' - Exit the system                                       ║
║   • 'mode' - Switch retriever (similarity/mmr)                               ║
║   • 'clear' - Clear query cache                                              ║
║   • 'roi' - Show ROI analytics                                               ║
║   • 'report' - Show system performance report                                ║
║   • 'stats' - Show detailed statistics                                       ║
║   • 'export' - Export query history                                          ║
║   • 'compare' - A/B test retrieval strategies (if enabled)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def initialize_system():
    """Initialize the Hybrid RAG System (loads vector DB + LLM + retrievers)"""
    print("🔧 Initializing Production Hybrid RAG System...\n")

    try:
        # Configuration - LAPTOP SAFE defaults
        config = {
            "retriever_k": 4,
            "llm_model": "gpt-4o-mini",
            "temperature": 0.1,
            "enable_caching": True,
            "enable_source_filtering": True,
            "verbose": True,
            "enable_ab_testing": False,  # KEEP FALSE for laptop safety
        }
        
        # Initialize with config
        rag = EnhancedProductionRAG(config=config)
        
        print("\n System Initialized Successfully!")
        print(" Ready to answer your engineering questions!\n")
        return rag
        
    except Exception as e:
        print(f"\n Failed to initialize system: {str(e)}")
        print("\n🔍 Troubleshooting Checklist:")
        print("   1. Vector database exists? Run ingestion pipeline first")
        print("   2. .env file has OPENAI_API_KEY?")
        print("   3. All dependencies installed? pip install -r requirements.txt")
        print("   4. Running from correct directory?")
        raise


def handle_special_commands(command, rag):
    """Handle system commands like mode, clear, roi, etc."""
    cmd = command.lower().strip()

    # Exit commands
    if cmd in ["exit", "quit", "q"]:
        return "exit"

    # Clear cache
    if cmd == "clear":
        rag.clear_cache()
        return True

    # Switch retriever mode
    if cmd == "mode":
        print("\n Current Retriever Mode:", rag.active_retriever_mode.upper())
        print("\n Available Retriever Modes:")
        print("   • similarity - Fast semantic search (default)")
        print("   • mmr - Maximum Marginal Relevance (diverse results)")
        print("   • hybrid - FUTURE: Ensemble retriever (similarity + BM25)")
        print("\n Tip: similarity is fastest, mmr gives diverse results")
        
        new_mode = input("\nEnter retriever mode: ").strip().lower()
        
        if new_mode in rag.retrievers:
            rag.switch_retriever_mode(new_mode)
        else:
            print(f" Invalid mode. Available: {list(rag.retrievers.keys())}")
        return True

    # Show ROI analytics
    if cmd == "roi":
        roi_data = rag.calculate_roi()
        
        if "message" in roi_data:
            print(f"\n {roi_data['message']}")
        else:
            print("\n" + "=" * 60)
            print(" BUSINESS ROI ANALYSIS")
            print("=" * 60)
            for key, value in roi_data.items():
                if isinstance(value, float):
                    if 'cost' in key or 'saved' in key or 'value' in key:
                        print(f"   {key:25}: ${value:,.2f}")
                    elif 'percentage' in key:
                        print(f"   {key:25}: {value:,.1f}%")
                    else:
                        print(f"   {key:25}: {value:,.2f}")
                else:
                    print(f"   {key:25}: {value}")
        return True

    # Show system report
    if cmd == "report":
        rag.system_report(detailed=False)
        return True

    # Show detailed stats
    if cmd == "stats":
        rag.system_report(detailed=True)
        return True

    # Export data
    if cmd == "export":
        print("\n Export Options:")
        print("   1. JSON only")
        print("   2. CSV only")
        print("   3. Both (default)")
        
        choice = input("Choose format (1/2/3): ").strip()
        
        if choice == "1":
            rag.export_data("json")
        elif choice == "2":
            rag.export_data("csv")
        else:
            rag.export_data("all")
        return True

    # A/B test comparison (only if enabled)
    if cmd == "compare":
        if rag.config["enable_ab_testing"]:
            test_question = input("Enter question for A/B test: ").strip()
            if test_question:
                rag.compare_retriever_strategies(test_question)
        else:
            print("\n A/B Testing is disabled in config (laptop safe mode)")
            print("   To enable, set 'enable_ab_testing': True in config")
        return True

    # Help menu
    if cmd in ["help", "h", "?"]:
        print_banner()
        return True

    # Not a command
    return False


def main():
    """Main entry point with error handling"""
    
    # Print banner
    print_banner()

    try:
        # Initialize system (LOADS YOUR BRAIN)
        rag = initialize_system()
    except Exception:
        print("\n Fatal: Could not initialize system. Please fix errors and try again.")
        sys.exit(1)

    # Show current mode
    print(f" Current Retriever Mode: [bold cyan]{rag.active_retriever_mode.upper()}[/bold cyan]" if USE_RICH else f"🔄 Current Retriever Mode: {rag.active_retriever_mode.upper()}")
    print(" Type 'help' for commands, 'exit' to quit\n")

    # Interactive loop
    while True:
        try:
            # Get user input
            user_query = input("\n??? Your Question: ").strip()

            # Handle special commands
            cmd_result = handle_special_commands(user_query, rag)
            
            if cmd_result == "exit":
                break
            elif cmd_result is True:
                continue
            elif user_query == "":
                print("⚠️ Please enter a valid question or type 'help' for commands.")
                continue

            print("\n🔎 Processing query with Hybrid RAG pipeline...\n")

            # ================= CORE EXECUTION =================
            # This line runs your FULL architecture:
            # Dynamic Retrieval → LCEL Chain → GPT-4o-mini → Metrics
            result = rag.ask(user_query)

            # Optional: structured access if needed for further processing
            # answer = result["answer"]
            # sources = result["sources"]
            # metrics = result["metrics"]

        except KeyboardInterrupt:
            print("\n\n⛔ Interrupted by user. Exiting safely...")
            break

        except Exception as e:
            print(f"\n🚨 Unexpected Error: {str(e)}")
            print("System will continue running...")
            if rag.config["verbose"]:
                import traceback
                traceback.print_exc()

    # Exit gracefully
    print("\n" + "=" * 60)
    print("📊 SESSION SUMMARY")
    print("=" * 60)
    if hasattr(rag, 'query_history') and rag.query_history:
        print(f"📈 Total Queries: {len(rag.query_history)}")
        print(f"💰 Total Cost: ${sum(q.cost_usd for q in rag.query_history):.6f}")
        print(f"🎯 Avg Confidence: {sum(q.confidence_score for q in rag.query_history)/len(rag.query_history)*100:.1f}%")
    else:
        print("No queries in this session")
    
    print("\n👋 Thanks for using AI Engineering Knowledge Assistant!")
    print("📁 Check 'exports/' directory for data exports")
    print("=" * 60)


# ================= ENTRY POINT =================
if __name__ == "__main__":
    main()