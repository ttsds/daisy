import json
import os
from glob import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from daisy.abstract import LANGUAGES


os.makedirs("figures", exist_ok=True)
# LLM sources bar chart. Grouped by language, three bars for podcasts, broadcast news, and content creators.

REPLACE_CHARTS = True

if REPLACE_CHARTS or not os.path.exists("data/figures/llm_sources.png"):
    languages = []
    types = []
    counts = []

    for language in LANGUAGES:
        with open(f"data/{language}/podcasts.json", "r", encoding="utf-8") as f:
            podcasts = json.load(f)
        with open(f"data/{language}/broadcast_news.json", "r", encoding="utf-8") as f:
            broadcast_news = json.load(f)
        with open(f"data/{language}/content_creators.json", "r", encoding="utf-8") as f:
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

    df = pd.DataFrame({"language": languages, "type": types, "count": counts})

    plt.figure(figsize=(20, 10))
    sns.barplot(x="language", y="count", hue="type", data=df)
    plt.xticks(rotation=90)
    plt.savefig("figures/llm_sources.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df_podcasts = df[df["type"] == "Podcasts"]
    df_podcasts = df_podcasts.sort_values(by="count", ascending=False)
    sns.barplot(x="language", y="count", data=df_podcasts)
    plt.xticks(rotation=90)
    plt.savefig("figures/podcasts.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df_broadcast_news = df[df["type"] == "Broadcast News"]
    df_broadcast_news = df_broadcast_news.sort_values(by="count", ascending=False)
    sns.barplot(x="language", y="count", data=df_broadcast_news)
    plt.xticks(rotation=90)
    plt.savefig("figures/broadcast_news.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df_content_creators = df[df["type"] == "Content Creators"]
    df_content_creators = df_content_creators.sort_values(by="count", ascending=False)
    sns.barplot(x="language", y="count", data=df_content_creators)
    plt.xticks(rotation=90)
    plt.savefig("figures/content_creators.png")
    plt.close()

if REPLACE_CHARTS or not os.path.exists("data/figures/link_results.png"):
    languages = []
    types = []
    counts = []

    for language in LANGUAGES:
        podcasts = glob(f"data/{language}/podcasts/*.json")
        broadcast_news = glob(f"data/{language}/broadcast_news/*.json")
        content_creators = glob(f"data/{language}/content_creators/*.json")

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

    df = pd.DataFrame({"language": languages, "type": types, "count": counts})

    plt.figure(figsize=(20, 10))
    sns.barplot(x="language", y="count", hue="type", data=df)
    plt.xticks(rotation=90)
    plt.savefig("figures/link_results.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df_podcasts = df[df["type"] == "Podcasts"]
    df_podcasts = df_podcasts.sort_values(by="count", ascending=False)
    sns.barplot(x="language", y="count", data=df_podcasts)
    plt.xticks(rotation=90)
    plt.savefig("figures/podcasts_link_results.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df_broadcast_news = df[df["type"] == "Broadcast News"]
    df_broadcast_news = df_broadcast_news.sort_values(by="count", ascending=False)
    sns.barplot(x="language", y="count", data=df_broadcast_news)
    plt.xticks(rotation=90)
    plt.savefig("figures/broadcast_news_link_results.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df_content_creators = df[df["type"] == "Content Creators"]
    df_content_creators = df_content_creators.sort_values(by="count", ascending=False)
    sns.barplot(x="language", y="count", data=df_content_creators)
    plt.xticks(rotation=90)
    plt.savefig("figures/content_creators_link_results.png")
    plt.close()


# Filtered dataset analysis functions
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
        for category in categories:
            filtered_files = glob(f"data/{language}/{category}-filtered/*.json")

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
    plt.xticks(rotation=90)  # Rotate labels 90 degrees
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
    plt.savefig("figures/filtered_flags_overview.png", dpi=300, bbox_inches="tight")
    plt.close()

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
        plt.xticks(rotation=90)  # Rotate labels 90 degrees
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
        plt.savefig(
            f"figures/filtered_{category}_all_three_flags.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f"Created {category_name} chart: {total_items:,} total items, "
            f"{avg_percentage:.1f}% average"
        )


# Create filtered dataset visualizations
if REPLACE_CHARTS or not os.path.exists("figures/filtered_flags_overview.png"):
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


# Count-based filtered dataset analysis functions
def create_overall_flag_count_chart(stats):
    """Create bar chart showing total counts of individual flags and all three true."""
    if stats["total_items"] == 0:
        print("No data found for overall flag count chart")
        return

    # Calculate counts
    counts = {
        "Is from Creator": stats["is_from_creator_true"],
        "Is Right Language": stats["is_right_language_true"],
        "Mostly Spoken Content": stats["mostly_spoken_content_true"],
        "All Three True": stats["all_three_true"],
    }

    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        counts.keys(),
        counts.values(),
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )

    # Add value labels on bars
    for bar, value in zip(bars, counts.values()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values()) * 0.01,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
            rotation=90,
        )

    plt.title(
        "Filtered Dataset: Flag Counts\n(Overall Statistics)",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("Count", fontsize=12)
    plt.ylim(0, max(counts.values()) * 1.2)
    plt.xticks(rotation=90)  # Rotate labels 90 degrees
    plt.grid(axis="y", alpha=0.3)

    # Add total count annotation
    plt.text(
        0.8,
        0.98,
        f'Total Items: {stats["total_items"]:,}',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    plt.tight_layout()
    plt.savefig(
        "figures/filtered_flags_count_overview.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Created overall flag count chart: {stats['total_items']:,} total items")


def create_category_language_count_charts(stats):
    """Create charts showing total counts of all three flags true across languages for each category."""
    categories = ["broadcast_news", "podcasts", "content_creators"]
    category_names = ["Broadcast News", "Podcasts", "Content Creators"]

    for category, category_name in zip(categories, category_names):
        # Prepare data for this category
        languages = []
        counts = []
        total_counts = []

        for language in LANGUAGES:
            lang_stats = stats["by_language"][language]
            cat_stats = lang_stats["by_category"][category]

            if cat_stats["total_items"] > 0:
                count = cat_stats["all_three_true"]
                languages.append(language)
                counts.append(count)
                total_counts.append(cat_stats["total_items"])

        if not languages:
            print(f"No data found for {category_name}")
            continue

        # Create the plot
        plt.figure(figsize=(20, 10))

        # Sort by count for better visualization
        sorted_data = sorted(
            zip(languages, counts, total_counts), key=lambda x: x[1], reverse=True
        )
        sorted_languages, sorted_counts, sorted_total_counts = zip(*sorted_data)

        bars = plt.bar(sorted_languages, sorted_counts, color="#2ca02c", alpha=0.7)

        # Add value labels on bars
        for bar, value, total in zip(bars, sorted_counts, sorted_total_counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sorted_counts) * 0.01,
                f"{value:,}\n(of {total:,})",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
                rotation=90,
            )

        plt.title(
            f"{category_name}: Count of Items with All Three Flags True\n"
            f"(Is from Creator AND Is Right Language AND Mostly Spoken Content)",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel("Count", fontsize=12)
        plt.xlabel("Language", fontsize=12)
        plt.xticks(rotation=90)  # Rotate labels 90 degrees
        plt.grid(axis="y", alpha=0.3)

        # Add summary statistics
        total_count = sum(sorted_counts)
        total_items = sum(sorted_total_counts)
        plt.text(
            0.8,
            0.98,
            f"Total Count: {total_count:,}\nTotal Items: {total_items:,}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen"),
        )

        plt.tight_layout()
        plt.savefig(
            f"figures/filtered_{category}_all_three_flags_count.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f"Created {category_name} count chart: {total_count:,} items with all flags true out of {total_items:,} total"
        )


def create_stacked_language_chart(stats):
    """Create stacked bar chart showing total counts per language after filtering."""
    languages = []
    category_counts = {"Broadcast News": [], "Podcasts": [], "Content Creators": []}

    for language in LANGUAGES:
        lang_stats = stats["by_language"][language]

        # Only include languages that have data
        total_lang_items = (
            lang_stats["by_category"]["broadcast_news"]["total_items"]
            + lang_stats["by_category"]["podcasts"]["total_items"]
            + lang_stats["by_category"]["content_creators"]["total_items"]
        )

        if total_lang_items > 0:
            languages.append(language)
            category_counts["Broadcast News"].append(
                lang_stats["by_category"]["broadcast_news"]["total_items"]
            )
            category_counts["Podcasts"].append(
                lang_stats["by_category"]["podcasts"]["total_items"]
            )
            category_counts["Content Creators"].append(
                lang_stats["by_category"]["content_creators"]["total_items"]
            )

    if not languages:
        print("No data found for stacked language chart")
        return

    # Sort by total count (highest to lowest)
    total_counts = [
        sum(category_counts[cat][i] for cat in category_counts)
        for i in range(len(languages))
    ]
    sorted_data = sorted(
        zip(
            languages,
            category_counts["Broadcast News"],
            category_counts["Podcasts"],
            category_counts["Content Creators"],
            total_counts,
        ),
        key=lambda x: x[4],
        reverse=True,
    )

    (
        sorted_languages,
        sorted_broadcast,
        sorted_podcasts,
        sorted_content,
        sorted_totals,
    ) = zip(*sorted_data)

    # Update category_counts with sorted data
    category_counts = {
        "Broadcast News": list(sorted_broadcast),
        "Podcasts": list(sorted_podcasts),
        "Content Creators": list(sorted_content),
    }
    languages = list(sorted_languages)

    # Create the stacked bar chart
    plt.figure(figsize=(20, 10))

    # Calculate bottom positions for stacking
    bottom_podcasts = category_counts["Broadcast News"]
    bottom_content = [
        bottom_podcasts[i] + category_counts["Podcasts"][i]
        for i in range(len(languages))
    ]

    # Create bars
    plt.bar(
        languages,
        category_counts["Broadcast News"],
        label="Broadcast News",
        color="#1f77b4",
        alpha=0.8,
    )
    plt.bar(
        languages,
        category_counts["Podcasts"],
        bottom=bottom_podcasts,
        label="Podcasts",
        color="#ff7f0e",
        alpha=0.8,
    )
    plt.bar(
        languages,
        category_counts["Content Creators"],
        bottom=bottom_content,
        label="Content Creators",
        color="#2ca02c",
        alpha=0.8,
    )

    # Add value labels on top of each stack
    for i, language in enumerate(languages):
        total = (
            category_counts["Broadcast News"][i]
            + category_counts["Podcasts"][i]
            + category_counts["Content Creators"][i]
        )
        plt.text(
            i,
            total
            + max(
                [
                    sum(category_counts[cat][i] for cat in category_counts)
                    for i in range(len(languages))
                ]
            )
            * 0.01,
            f"{total:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
            rotation=90,
        )

    plt.title(
        "Filtered Dataset: Total Items per Language\n(Stacked by Category)",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylabel("Total Count", fontsize=12)
    plt.xlabel("Language", fontsize=12)
    plt.xticks(rotation=90)  # Rotate labels 90 degrees
    plt.legend(loc="upper right")  # Put legend in top right
    plt.grid(axis="y", alpha=0.3)

    # Add summary statistics
    total_all_items = sum(
        [
            sum(category_counts[cat][i] for cat in category_counts)
            for i in range(len(languages))
        ]
    )
    plt.text(
        0.8,
        0.98,
        f"Total Items: {total_all_items:,}\nLanguages: {len(languages)}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue"),
    )

    plt.tight_layout()
    plt.savefig(
        "figures/filtered_stacked_language_totals.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"Created stacked language chart: {total_all_items:,} total items across {len(languages)} languages"
    )


# Create count-based filtered dataset visualizations
if REPLACE_CHARTS or not os.path.exists("figures/filtered_flags_count_overview.png"):
    print("\nCreating Count-Based Filtered Dataset Visualizations")
    print("=" * 60)

    # Load the data
    print("Loading filtered data...")
    stats = load_filtered_data()

    if stats["total_items"] > 0:
        print(f"Loaded {stats['total_items']:,} filtered items")

        # Create count-based visualizations
        print("\nCreating overall flag count distribution chart...")
        create_overall_flag_count_chart(stats)

        print("\nCreating category-specific count charts...")
        create_category_language_count_charts(stats)

        print("\nCreating stacked language chart...")
        create_stacked_language_chart(stats)

        print("\nAll count-based filtered dataset visualizations created successfully!")
    else:
        print("No filtered data found. Please run the filtering stage first.")
