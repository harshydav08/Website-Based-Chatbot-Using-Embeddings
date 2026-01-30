#!/usr/bin/env python3
"""
Easy run script for Website Chatbot.
This script provides various commands to run and test the chatbot system.
"""

import sys
import subprocess
import argparse
import logging

def run_streamlit_app():
    """Run the Streamlit application."""
    print("🚀 Starting Website Chatbot Streamlit Application...")
    print("📍 The app will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit app: {e}")
        sys.exit(1)

def run_tests():
    """Run the system validation tests."""
    print("🧪 Running Website Chatbot System Tests...")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "test_system.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed with exit code: {e.returncode}")
        sys.exit(1)

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def show_system_info():
    """Show system information and status."""
    print("ℹ️  Website Chatbot System Information")
    print("=" * 60)
    
    try:
        from core import WebsiteChatbot
        from config import config
        
        chatbot = WebsiteChatbot()
        status = chatbot.get_system_status()
        
        print("📊 System Status:")
        print(f"   Status: {status['status']}")
        
        if 'services' in status:
            services = status['services']
            print(f"   Embedding Service: {services.get('embedding_service', {}).get('loaded', 'Unknown')}")
            print(f"   Vector Database: {services.get('vector_database', {}).get('status', 'Unknown')}")
            print(f"   Total Chunks: {services.get('vector_database', {}).get('total_chunks', 0)}")
        
        if 'config' in status:
            config_info = status['config']
            print("\n⚙️  Configuration:")
            print(f"   Embedding Model: {config_info.get('embedding_model', 'Unknown')}")
            print(f"   Chunk Size: {config_info.get('chunk_size', 'Unknown')}")
            print(f"   Top K Results: {config_info.get('top_k_results', 'Unknown')}")
        
        print("\n✅ System information retrieved successfully")
        
    except Exception as e:
        print(f"❌ Error getting system info: {e}")
        print("   Try running 'python run.py install' first")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Website Chatbot - Easy run script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                 # Run the Streamlit app (default)
  python run.py app             # Run the Streamlit app
  python run.py test            # Run system tests
  python run.py install         # Install dependencies
  python run.py info            # Show system information
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="app",
        choices=["app", "test", "install", "info"],
        help="Command to run (default: app)"
    )
    
    args = parser.parse_args()
    
    print("🤖 Website Chatbot - Production Ready AI Assistant")
    print("=" * 60)
    
    if args.command == "app":
        run_streamlit_app()
    elif args.command == "test":
        run_tests()
    elif args.command == "install":
        install_dependencies()
    elif args.command == "info":
        show_system_info()

if __name__ == "__main__":
    main()
