# DAISY Pipeline Figure Creation Scripts

This directory contains scripts for creating visualizations and tables for the DAISY pipeline.

## Scripts

### Main Scripts

- **`run_all.py`** - Main script that runs all figure creation scripts in sequence
- **`utils.py`** - Common utilities for all figure scripts (environment setup, data paths, user prompts)

### Individual Scripts

- **`create_figures.py`** - Creates main visualizations:
  - LLM sources bar charts (grouped by language and type)
  - Link results charts (actual downloaded content)
  - Filtered dataset analysis (flag distributions, category-specific charts)
  - Speaker embeddings PCA visualization (for specific languages)

- **`create_flower.py`** - Creates flower plot visualization showing language distribution and country diversity

- **`create_table.py`** - Creates data summary table in markdown format with language statistics

- **`analyze_speaker_similarity.py`** - Analyzes speaker consistency by computing cosine similarities between utterances from the same speaker

## Usage

### Quick Start

Run all figure creation scripts:
```bash
cd scripts/figures
python run_all.py
```

### Individual Scripts

Run individual scripts:
```bash
cd scripts/figures

# Create data summary table
python create_table.py

# Create flower plot
python create_flower.py

# Create main figures
python create_figures.py

# Analyze speaker similarities
python analyze_speaker_similarity.py
```

## Environment Setup

The scripts use the `DAISY_ROOT` environment variable to locate the dataset. Set this in your `.env` file:

```bash
DAISY_ROOT=/path/to/daisy_dataset
```

If not set, it defaults to `../daisy_dataset`.

## Output

- **Figures**: Saved to `figures/` directory (created automatically)
- **Data Summary**: Saved as `data_summary.md` in the current directory

## Interactive Features

- **Plot Recreation**: Scripts will ask if you want to recreate existing plots to speed up computation
- **Progress Reporting**: Each script provides detailed progress information
- **Error Handling**: Graceful handling of missing data files and other errors

## Dependencies

The scripts require the following Python packages:
- matplotlib
- seaborn
- pandas
- numpy
- scikit-learn
- torch
- torchaudio
- nemo-toolkit
- countryflag
- python-dotenv

## Notes

- The speaker embeddings PCA visualization requires the NeMo speaker verification model
- Font configuration is handled automatically for Unicode support
- All scripts include comprehensive error handling and progress reporting
