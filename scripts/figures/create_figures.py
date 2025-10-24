#!/usr/bin/env python3
"""
Main figure creation script for DAISY pipeline.
Creates various visualizations including LLM sources, filtered data analysis, and speaker embeddings PCA.
"""

import json
import os
import sys
from glob import glob
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torchaudio
from sklearn.decomposition import PCA
from nemo.collections.asr.models import EncDecSpeakerLabelModel

# Import utils first to set up the path
from utils import setup_figure_environment, should_recreate_plots, get_language_data_path, get_utterances_path, save_figure
from daisy.core import LANGUAGES

def create_llm_sources_charts():
    """Create LLM sources bar charts grouped by language and type."""
    print("\nCreating LLM Sources Charts")
    print("=" * 40)
    
    languages = []
    types = []
    counts = []

    for language in LANGUAGES:
        lang_path = get_language_data_path(language)
        
        try:
            with open(os.path.join(lang_path, "podcasts.json"), "r", encoding="utf-8") as f:
                podcasts = json.load(f)
            with open(os.path.join(lang_path, "broadcast_news.json"), "r", encoding="utf-8") as f:
                broadcast_news = json.load(f)
            with open(os.path.join(lang_path, "content_creators.json"), "r", encoding="utf-8") as f:
                content_creators = json.load(f)
                
            languages.append(language)
            types.append("Podcasts")
            counts.append(len(podcasts))

            languages.append(language)
            types.append("Broadcast News")
            counts.append(len(broadcast_news))

            languages.append(language)
            types.append("Content Creators")
            counts.append(len(content_creators))
        except FileNotFoundError as e:
            print(f"Warning: Missing data files for {language}: {e}")
            continue

    df = pd.DataFrame({"language": languages, "type": types, "count": counts})

    # Combined chart
    plt.figure(figsize=(20, 10))
    sns.barplot(x="language", y="count", hue="type", data=df)
    plt.xticks(rotation=90)
    plt.title("LLM Sources by Language and Type", fontsize=16, fontweight="bold")
    save_figure("llm_sources.png")

    # Individual charts
    for chart_type in ["Podcasts", "Broadcast News", "Content Creators"]:
        df_subset = df[df["type"] == chart_type].sort_values(by="count", ascending=False)
        plt.figure(figsize=(20, 10))
        sns.barplot(x="language", y="count", data=df_subset)
        plt.xticks(rotation=90)
        plt.title(f"{chart_type} Sources by Language", fontsize=16, fontweight="bold")
        save_figure(f"{chart_type.lower().replace(' ', '_')}.png")

def create_link_results_charts():
    """Create link results charts showing actual downloaded content."""
    print("\nCreating Link Results Charts")
    print("=" * 40)
    
    languages = []
    types = []
    counts = []

    for language in LANGUAGES:
        lang_path = get_language_data_path(language)
        
        try:
            podcasts = glob(os.path.join(lang_path, "podcasts", "*.json"))
            broadcast_news = glob(os.path.join(lang_path, "broadcast_news", "*.json"))
            content_creators = glob(os.path.join(lang_path, "content_creators", "*.json"))

            podcasts_count = 0
            for podcast in podcasts:
                with open(podcast, "r", encoding="utf-8") as f:
                    podcast_data = json.load(f)
                podcasts_count += len(podcast_data)

            broadcast_news_count = 0
            for news in broadcast_news:
                with open(news, "r", encoding="utf-8") as f:
                    news_data = json.load(f)
                broadcast_news_count += len(news_data)

            content_creators_count = 0
            for creator in content_creators:
                with open(creator, "r", encoding="utf-8") as f:
                    creator_data = json.load(f)
                content_creators_count += len(creator_data)

            languages.append(language)
            types.append("Podcasts")
            counts.append(podcasts_count)

            languages.append(language)
            types.append("Broadcast News")
            counts.append(broadcast_news_count)

            languages.append(language)
            types.append("Content Creators")
            counts.append(content_creators_count)
        except Exception as e:
            print(f"Warning: Error processing {language}: {e}")
            continue

    df = pd.DataFrame({"language": languages, "type": types, "count": counts})

    # Combined chart
    plt.figure(figsize=(20, 10))
    sns.barplot(x="language", y="count", hue="type", data=df)
    plt.xticks(rotation=90)
    plt.title("Link Results by Language and Type", fontsize=16, fontweight="bold")
    save_figure("link_results.png")

    # Individual charts
    for chart_type in ["Podcasts", "Broadcast News", "Content Creators"]:
        df_subset = df[df["type"] == chart_type].sort_values(by="count", ascending=False)
        plt.figure(figsize=(20, 10))
        sns.barplot(x="language", y="count", data=df_subset)
        plt.xticks(rotation=90)
        plt.title(f"{chart_type} Link Results by Language", fontsize=16, fontweight="bold")
        save_figure(f"{chart_type.lower().replace(' ', '_')}_link_results.png")

def load_filtered_data():
    """Load all filtered data and return structured statistics."""
    stats = {
        "total_items": 0,
        "is_from_creator_true": 0,
        "is_right_language_true": 0,
        "mostly_spoken_content_true": 0,
        "all_three_true": 0,
        "by_language": defaultdict(
            lambda: {
                "total_items": 0,
                "is_from_creator_true": 0,
                "is_right_language_true": 0,
                "mostly_spoken_content_true": 0,
                "all_three_true": 0,
                "by_category": defaultdict(
                    lambda: {
                        "total_items": 0,
                        "is_from_creator_true": 0,
                        "is_right_language_true": 0,
                        "mostly_spoken_content_true": 0,
                        "all_three_true": 0,
                    }
                ),
            }
        ),
    }

    categories = ["podcasts", "broadcast_news", "content_creators"]

    for language in LANGUAGES:
        lang_path = get_language_data_path(language)
        for category in categories:
            filtered_files = glob(os.path.join(lang_path, f"{category}-filtered", "*.json"))

            for file_path in filtered_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        items = json.load(f)

                    for item in items:
                        # Global stats
                        stats["total_items"] += 1
                        if item.get("is_from_creator", False):
                            stats["is_from_creator_true"] += 1
                        if item.get("is_right_language", False):
                            stats["is_right_language_true"] += 1
                        if item.get("mostly_spoken_content", False):
                            stats["mostly_spoken_content_true"] += 1
                        if (
                            item.get("is_from_creator", False)
                            and item.get("is_right_language", False)
                            and item.get("mostly_spoken_content", False)
                        ):
                            stats["all_three_true"] += 1

                        # Language stats
                        lang_stats = stats["by_language"][language]
                        lang_stats["total_items"] += 1
                        if item.get("is_from_creator", False):
                            lang_stats["is_from_creator_true"] += 1
                        if item.get("is_right_language", False):
                            lang_stats["is_right_language_true"] += 1
                        if item.get("mostly_spoken_content", False):
                            lang_stats["mostly_spoken_content_true"] += 1
                        if (
                            item.get("is_from_creator", False)
                            and item.get("is_right_language", False)
                            and item.get("mostly_spoken_content", False)
                        ):
                            lang_stats["all_three_true"] += 1

                        # Category stats
                        cat_stats = lang_stats["by_category"][category]
                        cat_stats["total_items"] += 1
                        if item.get("is_from_creator", False):
                            cat_stats["is_from_creator_true"] += 1
                        if item.get("is_right_language", False):
                            cat_stats["is_right_language_true"] += 1
                        if item.get("mostly_spoken_content", False):
                            cat_stats["mostly_spoken_content_true"] += 1
                        if (
                            item.get("is_from_creator", False)
                            and item.get("is_right_language", False)
                            and item.get("mostly_spoken_content", False)
                        ):
                            cat_stats["all_three_true"] += 1

                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not process {file_path}: {e}")
                    continue

    return stats

def create_overall_flag_chart(stats):
    """Create bar chart showing percentage of individual flags and all three true."""
    if stats["total_items"] == 0:
        print("No data found for overall flag chart")
        return

    # Calculate percentages
    total = stats["total_items"]
    percentages = {
        "Is from Creator": (stats["is_from_creator_true"] / total) * 100,
        "Is Right Language": (stats["is_right_language_true"] / total) * 100,
        "Mostly Spoken Content": (stats["mostly_spoken_content_true"] / total) * 100,
        "All Three True": (stats["all_three_true"] / total) * 100,
    }

    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        percentages.keys(),
        percentages.values(),
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )

    # Add value labels on bars
    for bar, value in zip(bars, percentages.values()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            rotation=90,
        )

    plt.title(
        "Filtered Dataset: Flag Distribution\n(Overall Statistics)",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.ylim(0, max(percentages.values()) * 1.2)
    plt.xticks(rotation=90)
    plt.grid(axis="y", alpha=0.3)

    # Add total count annotation
    plt.text(
        0.8,
        0.98,
        f"Total Items: {total:,}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    plt.tight_layout()
    save_figure("filtered_flags_overview.png")

    print(f"Created overall flag chart: {total:,} total items")

def create_category_language_charts(stats):
    """Create charts showing percentage of all three flags true across languages for each category."""
    categories = ["broadcast_news", "podcasts", "content_creators"]
    category_names = ["Broadcast News", "Podcasts", "Content Creators"]

    for category, category_name in zip(categories, category_names):
        # Prepare data for this category
        languages = []
        percentages = []
        total_counts = []

        for language in LANGUAGES:
            lang_stats = stats["by_language"][language]
            cat_stats = lang_stats["by_category"][category]

            if cat_stats["total_items"] > 0:
                percentage = (
                    cat_stats["all_three_true"] / cat_stats["total_items"]
                ) * 100
                languages.append(language)
                percentages.append(percentage)
                total_counts.append(cat_stats["total_items"])

        if not languages:
            print(f"No data found for {category_name}")
            continue

        # Create the plot
        plt.figure(figsize=(20, 10))

        # Sort by percentage for better visualization
        sorted_data = sorted(
            zip(languages, percentages, total_counts), key=lambda x: x[1], reverse=True
        )
        sorted_languages, sorted_percentages, sorted_counts = zip(*sorted_data)

        bars = plt.bar(sorted_languages, sorted_percentages, color="#2ca02c", alpha=0.7)

        # Add value labels on bars
        for bar, value, count in zip(bars, sorted_percentages, sorted_counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}%\n(n={count})",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
                rotation=90,
            )

        plt.title(
            f"{category_name}: Percentage with All Three Flags True\n"
            f"(Is from Creator AND Is Right Language AND Mostly Spoken Content)",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xlabel("Language", fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis="y", alpha=0.3)

        # Add summary statistics
        avg_percentage = np.mean(sorted_percentages)
        total_items = sum(sorted_counts)
        plt.text(
            0.8,
            0.98,
            f"Average: {avg_percentage:.1f}%\nTotal Items: {total_items:,}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen"),
        )

        plt.tight_layout()
        save_figure(f"filtered_{category}_all_three_flags.png")

        print(
            f"Created {category_name} chart: {total_items:,} total items, "
            f"{avg_percentage:.1f}% average"
        )

def create_filtered_dataset_visualizations():
    """Create filtered dataset visualizations."""
    print("\nCreating Filtered Dataset Visualizations")
    print("=" * 50)

    # Load the data
    print("Loading filtered data...")
    stats = load_filtered_data()

    if stats["total_items"] > 0:
        print(f"Loaded {stats['total_items']:,} filtered items")

        # Create visualizations
        print("\nCreating overall flag distribution chart...")
        create_overall_flag_chart(stats)

        print("\nCreating category-specific charts...")
        create_category_language_charts(stats)

        print("\nAll filtered dataset visualizations created successfully!")
    else:
        print("No filtered data found. Please run the filtering stage first.")

# Speaker embeddings PCA visualization functions
def load_speaker_model():
    """Load the speaker verification model."""
    print("Loading speaker verification model...")
    model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model.eval()
    return model

def extract_embeddings_from_wav(wav_path, speaker_model):
    """Extract speaker embedding from a single wav file."""
    try:
        # Load audio
        audio, sr = torchaudio.load(wav_path)
        audio = audio.squeeze(0).numpy()  # Convert to mono numpy array
        
        # Extract embedding
        with torch.no_grad():
            embedding, _ = speaker_model.infer_segment(audio)
            embedding = embedding.squeeze(0).cpu().numpy()
        
        return embedding
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def create_speaker_embeddings_pca(language="af"):
    """Create PCA visualization of speaker embeddings for a specific language."""
    print(f"\nCreating Speaker Embeddings PCA Visualization for {language.upper()}")
    print("=" * 60)
    
    # Set up paths
    utterances_dir = get_utterances_path(language)
    
    # Check if utterances directory exists
    if not os.path.exists(utterances_dir):
        print(f"Error: Utterances directory not found: {utterances_dir}")
        return
    
    print(f"Collecting embeddings from {utterances_dir}...")
    
    # Load speaker model
    speaker_model = load_speaker_model()
    
    # Find all wav files
    wav_files = glob(os.path.join(utterances_dir, "*", "*", "*.wav"))
    print(f"Found {len(wav_files)} wav files")
    
    # Extract embeddings for each wav file individually
    embeddings_list = []
    speakers_list = []
    wav_files_list = []
    
    for wav_file in wav_files:
        # Extract speaker name from path: language/utterances/video_dir/speaker_dir/file.wav
        path_parts = wav_file.split(os.sep)
        speaker_dir = path_parts[-2]  # The speaker directory name
        
        # Extract embedding
        embedding = extract_embeddings_from_wav(wav_file, speaker_model)
        if embedding is not None:
            embeddings_list.append(embedding)
            speakers_list.append(speaker_dir)
            wav_files_list.append(wav_file)
    
    print(f"Extracted embeddings for {len(embeddings_list)} wav files from {len(set(speakers_list))} speakers")
    
    if len(embeddings_list) == 0:
        print("No speaker embeddings found!")
        return
    
    # Prepare data for PCA
    embeddings_matrix = np.array(embeddings_list)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_matrix)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Speaker': speakers_list,
        'WavFile': [os.path.basename(f) for f in wav_files_list]
    })
    
    # Create the plot
    plt.figure(figsize=(15, 15))
    
    # Create scatter plot with hue by speaker
    scatter = sns.scatterplot(
        data=df,
        x='PC1',
        y='PC2',
        hue='Speaker',
        alpha=0.7,
        palette='tab20'  # Use a palette that handles many colors well
    )
    
    # Customize the plot
    unique_speakers = len(set(speakers_list))
    plt.title(f'Speaker Embeddings PCA Visualization - {language.upper()} Language\n'
              f'({unique_speakers} speakers, {len(embeddings_list)} total utterances)',
              fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    save_figure(f"speaker_embeddings_pca_{language}.png")
    
    # Print some statistics
    print(f"\nPCA Statistics:")
    print(f"Total variance explained by PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"Total variance explained by PC2: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total variance explained by both PCs: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Print speaker statistics
    print(f"\nSpeaker Statistics:")
    speaker_counts = Counter(speakers_list)
    for speaker, count in speaker_counts.most_common(10):  # Top 10 speakers
        print(f"  {speaker}: {count} utterances")
    
    print(f"\nSpeaker embeddings PCA visualization for {language.upper()} complete!")

def main():
    """Main function to create all visualizations."""
    print("DAISY Pipeline Figure Creation")
    print("=" * 50)
    
    # Set up environment
    setup_figure_environment()
    
    # Ask user about recreating plots
    recreate_plots = should_recreate_plots()
    
    if not recreate_plots:
        print("Skipping figure creation.")
        return
    
    # Create all visualizations
    try:
        # LLM Sources charts
        create_llm_sources_charts()
        
        # Link results charts
        create_link_results_charts()
        
        # Filtered dataset visualizations
        create_filtered_dataset_visualizations()
        
        # Speaker embeddings PCA visualization
        create_speaker_embeddings_pca("af")
        
        print("\n" + "=" * 50)
        print("All visualizations created successfully!")
        
    except Exception as e:
        print(f"Error during figure creation: {e}")
        raise

if __name__ == "__main__":
    main()
