"""
Common utilities for figure creation scripts.
Handles environment setup, data paths, and user prompts.
"""

import os
import sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Set up data paths
DAISY_ROOT = os.getenv("DAISY_ROOT", "../daisy_dataset")
# Use absolute path for figures directory
FIGURES_DIR = os.path.abspath("../../figures")

def setup_figure_environment():
    """Set up the environment for figure creation."""
    # Create figures directory
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    print(f"Using DAISY_ROOT: {DAISY_ROOT}")
    print(f"Output directory: {FIGURES_DIR}")

def should_recreate_plots():
    """Ask user if they want to recreate existing plots."""
    existing_plots = []
    for file in os.listdir(FIGURES_DIR):
        if file.endswith('.png'):
            existing_plots.append(file)
    
    if not existing_plots:
        return True
    
    print(f"\nFound {len(existing_plots)} existing plots:")
    for plot in sorted(existing_plots)[:10]:  # Show first 10
        print(f"  - {plot}")
    if len(existing_plots) > 10:
        print(f"  ... and {len(existing_plots) - 10} more")
    
    while True:
        response = input("\nDo you want to recreate existing plots? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please answer 'y' or 'n'")

def get_language_data_path(language):
    """Get the data path for a specific language."""
    return os.path.join(DAISY_ROOT, language)

def get_utterances_path(language):
    """Get the utterances path for a specific language."""
    return os.path.join(DAISY_ROOT, language, "utterances")

def save_figure(filename, dpi=300, bbox_inches='tight'):
    """Save figure with consistent settings."""
    filepath = os.path.join(FIGURES_DIR, filename)
    
    try:
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {filepath}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    finally:
        plt.close()
