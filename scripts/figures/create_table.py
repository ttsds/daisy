#!/usr/bin/env python3
"""
Create data summary table for DAISY pipeline.
Generates a markdown table with language statistics including sources, countries, and date ranges.
"""

import os
import sys
import pandas as pd

# Import utils first to set up the path
from utils import get_language_data_path
from daisy.core import LANGUAGES

# Try to import countryflag, make it optional
try:
    import countryflag
    COUNTRYFLAG_AVAILABLE = True
except ImportError:
    COUNTRYFLAG_AVAILABLE = False
    print("Warning: countryflag not available. Country flags will not be included in the table.")

def map_country(country):
    """Map country names to standardized versions."""
    country_map = {
        "UK": "United Kingdom",
        "Korea": "South Korea",
        "Macedonia": "North Macedonia",
        "España": "Spain",
        "UAE": "United Arab Emirates",
        "Palestinian Territories": "Palestine",
        "USA": "United States",
        "US": "United States",
    }
    
    if country in country_map:
        return country_map[country]
    return country

def create_data_summary_table():
    """Create a comprehensive data summary table."""
    print("Creating Data Summary Table")
    print("=" * 40)
    
    data_dict = {
        "language_iso2": [],
        "language": [],
        "language_en": [],
        "sources": [],
        "countries": [],
        "date range": [],
    }

    additional_emojis = {
        "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
        "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
        "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    }

    for language in LANGUAGES:
        try:
            lang_path = get_language_data_path(language)
            df = pd.read_csv(os.path.join(lang_path, "sampled_items.csv"))
            df["date"] = pd.to_datetime(df["date"])
            
            data_dict["language_iso2"].append(LANGUAGES[language].iso2)
            data_dict["language"].append(LANGUAGES[language].native_name)
            data_dict["language_en"].append(LANGUAGES[language].english_name)
            data_dict["sources"].append(", ".join(df["name"].value_counts().index[:3]) + "...")
            
            # Map countries based on country_map for the whole dataframe
            df["country"] = df["country"].map(map_country)
            country_counts = []
            
            for country in df["country"].value_counts().index:
                _country = country
                emoji = None
                
                if COUNTRYFLAG_AVAILABLE:
                    try:
                        emoji = countryflag.getflag(country)
                    except:
                        pass
                
                if emoji is None and country in additional_emojis:
                    emoji = additional_emojis[country]
                        
                if emoji is not None:
                    country_counts.append(
                        f"{_country} {emoji} ({df['country'].value_counts()[_country]})"
                    )
                else:
                    country_counts.append(
                        f"{_country} ({df['country'].value_counts()[_country]})"
                    )
                    
            data_dict["countries"].append(", ".join(country_counts))
            
            start_date_formatted = df["date"].min().strftime("%Y-%m-%d")
            end_date_formatted = df["date"].max().strftime("%Y-%m-%d")
            data_dict["date range"].append(f"{start_date_formatted} - {end_date_formatted}")
            
        except FileNotFoundError:
            print(f"Warning: No data found for language {language}")
            continue
        except Exception as e:
            print(f"Error processing language {language}: {e}")
            continue

    df = pd.DataFrame(data_dict)
    
    # Rename columns for markdown
    df.columns = [
        "Language ISO 2",
        "Language",
        "Language (English)",
        "Sources",
        "Countries",
        "Date Range",
    ]
    
    # Save to markdown file
    output_path = "data_summary.md"
    df.to_markdown(output_path, index=False)
    
    print(f"Data summary table saved to: {output_path}")
    print(f"Processed {len(df)} languages")

def main():
    """Main function to create data summary table."""
    print("DAISY Pipeline Data Summary Table Creation")
    print("=" * 50)
    
    # Create data summary table
    create_data_summary_table()
    
    print("\nData summary table creation complete!")

if __name__ == "__main__":
    main()
