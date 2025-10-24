#!/usr/bin/env python3
"""
Main script to run all DAISY pipeline figure creation scripts.
This script orchestrates the creation of all visualizations and tables.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import setup_figure_environment, should_recreate_plots

def run_script(script_name, description, timeout=300):
    """Run a figure creation script with timeout."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        return False
    
    try:
        # Use input to provide "y" to the subprocess
        result = subprocess.run([sys.executable, script_path], 
                              input="y\n", capture_output=True, text=True, check=True, timeout=timeout)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return True
    except subprocess.TimeoutExpired as e:
        print(f"Timeout: {script_name} took longer than {timeout} seconds")
        print("This might be due to font loading or processing many languages")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error running {script_name}: {e}")
        return False

def main():
    """Main function to run all figure creation scripts."""
    print("DAISY Pipeline - Complete Figure Creation")
    print("=" * 60)
    
    # Set up environment
    setup_figure_environment()
    
    # Skip user prompt when running all scripts - assume we want to recreate
    print("Running in automated mode - will recreate existing plots")
    recreate_plots = True
    
    if not recreate_plots:
        print("Skipping all figure creation.")
        return
    
    # Define scripts to run
    scripts = [
        ("create_table.py", "Data Summary Table Creation"),
        ("create_flower.py", "Flower Plot Creation"),
        ("create_figures.py", "Main Figures Creation (LLM Sources, Filtered Data, Speaker Embeddings)"),
    ]
    
    # Run each script
    success_count = 0
    total_scripts = len(scripts)
    
    for script_name, description in scripts:
        if run_script(script_name, description):
            success_count += 1
        else:
            print(f"Failed to run {script_name}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully completed: {success_count}/{total_scripts} scripts")
    
    if success_count == total_scripts:
        print("All figure creation scripts completed successfully!")
    else:
        print(f"Some scripts failed. Check the output above for details.")
    
    print(f"\nFigures saved to: figures/")
    print("Data summary saved to: data_summary.md")

if __name__ == "__main__":
    main()
