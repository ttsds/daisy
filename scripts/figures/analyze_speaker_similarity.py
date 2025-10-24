#!/usr/bin/env python3
"""
Measure cosine similarity between utterances belonging to the same speaker.
This script analyzes speaker consistency by computing pairwise cosine similarities
between all utterances from the same speaker for a given language.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import torchaudio
from nemo.collections.asr.models import EncDecSpeakerLabelModel

# Import utils first to set up the path
from utils import setup_figure_environment, get_utterances_path, save_figure
from daisy.core import LANGUAGES

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

def collect_speaker_embeddings(language="af"):
    """Collect embeddings for all utterances, grouped by speaker."""
    print(f"Collecting speaker embeddings for {language.upper()}...")
    
    utterances_dir = get_utterances_path(language)
    
    if not os.path.exists(utterances_dir):
        print(f"Error: Utterances directory not found: {utterances_dir}")
        return {}
    
    # Load speaker model
    speaker_model = load_speaker_model()
    
    # Find all wav files
    wav_files = []
    for root, dirs, files in os.walk(utterances_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    print(f"Found {len(wav_files)} wav files")
    
    # Group by speaker and collect embeddings
    speaker_embeddings = defaultdict(list)
    speaker_files = defaultdict(list)
    
    for wav_file in wav_files:
        # Extract speaker name from path: language/utterances/video_dir/speaker_dir/file.wav
        path_parts = wav_file.split(os.sep)
        speaker_dir = path_parts[-2]  # The speaker directory name
        
        # Extract embedding
        embedding = extract_embeddings_from_wav(wav_file, speaker_model)
        if embedding is not None:
            speaker_embeddings[speaker_dir].append(embedding)
            speaker_files[speaker_dir].append(os.path.basename(wav_file))
    
    # Convert to numpy arrays
    speaker_data = {}
    for speaker, embeddings_list in speaker_embeddings.items():
        if len(embeddings_list) > 1:  # Only include speakers with multiple utterances
            speaker_data[speaker] = {
                'embeddings': np.array(embeddings_list),
                'files': speaker_files[speaker],
                'count': len(embeddings_list)
            }
    
    print(f"Found {len(speaker_data)} speakers with multiple utterances")
    return speaker_data

def compute_speaker_similarities(speaker_data):
    """Compute pairwise cosine similarities for each speaker."""
    print("Computing pairwise cosine similarities...")
    
    similarities_data = []
    
    for speaker, data in speaker_data.items():
        embeddings = data['embeddings']
        files = data['files']
        count = data['count']
        
        # Compute pairwise cosine similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        # Extract upper triangle (excluding diagonal) for pairwise similarities
        for i in range(count):
            for j in range(i + 1, count):
                similarity = similarity_matrix[i, j]
                similarities_data.append({
                    'speaker': speaker,
                    'file1': files[i],
                    'file2': files[j],
                    'similarity': similarity,
                    'speaker_count': count
                })
    
    return pd.DataFrame(similarities_data)

def create_similarity_visualizations(df, language="af"):
    """Create visualizations of speaker similarities."""
    print("Creating similarity visualizations...")
    
    # Overall similarity distribution
    plt.figure(figsize=(12, 8))
    plt.hist(df['similarity'], bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of Cosine Similarities Between Same-Speaker Utterances\n{language.upper()} Language', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_sim = df['similarity'].mean()
    std_sim = df['similarity'].std()
    plt.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.3f}')
    plt.legend()
    
    plt.tight_layout()
    save_figure(f"speaker_similarity_distribution_{language}.png")
    
    # Similarity by speaker count
    plt.figure(figsize=(12, 8))
    speaker_stats = df.groupby('speaker_count')['similarity'].agg(['mean', 'std', 'count']).reset_index()
    
    plt.errorbar(speaker_stats['speaker_count'], speaker_stats['mean'], 
                yerr=speaker_stats['std'], fmt='o-', capsize=5, capthick=2)
    plt.title(f'Average Similarity vs Number of Utterances per Speaker\n{language.upper()} Language', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Number of Utterances per Speaker', fontsize=12)
    plt.ylabel('Average Cosine Similarity', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(f"speaker_similarity_vs_count_{language}.png")
    
    # Box plot by speaker count ranges
    plt.figure(figsize=(12, 8))
    
    # Create ranges for better visualization
    df['count_range'] = pd.cut(df['speaker_count'], 
                              bins=[0, 2, 3, 4, 5, 10, 20, 100], 
                              labels=['2', '3', '4', '5', '6-10', '11-20', '20+'])
    
    sns.boxplot(data=df, x='count_range', y='similarity')
    plt.title(f'Similarity Distribution by Speaker Utterance Count\n{language.upper()} Language', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Number of Utterances per Speaker', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(f"speaker_similarity_boxplot_{language}.png")
    
    # Individual speaker similarities (top 20 speakers by utterance count)
    top_speakers = df.groupby('speaker')['similarity'].agg(['mean', 'count']).reset_index()
    top_speakers = top_speakers.sort_values('count', ascending=False).head(20)
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(top_speakers)), top_speakers['mean'], 
                   alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, top_speakers['count'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    plt.title(f'Average Similarity for Top 20 Speakers by Utterance Count\n{language.upper()} Language', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Speaker (ranked by utterance count)', fontsize=12)
    plt.ylabel('Average Cosine Similarity', fontsize=12)
    plt.xticks(range(len(top_speakers)), [f"S{i+1}" for i in range(len(top_speakers))], rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(f"top_speakers_similarity_{language}.png")

def print_statistics(df, language="af"):
    """Print detailed statistics about speaker similarities."""
    print(f"\n{'='*60}")
    print(f"SPEAKER SIMILARITY STATISTICS - {language.upper()} LANGUAGE")
    print(f"{'='*60}")
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total pairwise comparisons: {len(df):,}")
    print(f"  Number of speakers: {df['speaker'].nunique()}")
    print(f"  Mean similarity: {df['similarity'].mean():.4f}")
    print(f"  Median similarity: {df['similarity'].median():.4f}")
    print(f"  Standard deviation: {df['similarity'].std():.4f}")
    print(f"  Min similarity: {df['similarity'].min():.4f}")
    print(f"  Max similarity: {df['similarity'].max():.4f}")
    
    # High similarity threshold analysis
    high_similarity_threshold = 0.8
    high_sim_count = len(df[df['similarity'] >= high_similarity_threshold])
    high_sim_percentage = (high_sim_count / len(df)) * 100
    
    print(f"\nHigh Similarity Analysis (≥{high_similarity_threshold}):")
    print(f"  Number of pairs with similarity ≥{high_similarity_threshold}: {high_sim_count:,}")
    print(f"  Percentage of pairs with similarity ≥{high_similarity_threshold}: {high_sim_percentage:.1f}%")
    
    # Speaker-level analysis for high similarity
    speaker_means = df.groupby('speaker')['similarity'].agg(['mean', 'count']).reset_index()
    high_sim_speakers = speaker_means[speaker_means['mean'] >= high_similarity_threshold]
    high_sim_speaker_percentage = (len(high_sim_speakers) / len(speaker_means)) * 100
    
    print(f"\nSpeaker-Level High Similarity Analysis (≥{high_similarity_threshold}):")
    print(f"  Number of speakers with average similarity ≥{high_similarity_threshold}: {len(high_sim_speakers):,}")
    print(f"  Percentage of speakers with average similarity ≥{high_similarity_threshold}: {high_sim_speaker_percentage:.1f}%")
    
    # Percentiles
    percentiles = [25, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(df['similarity'], p)
        print(f"  {p}th percentile: {value:.4f}")
    
    # Statistics by speaker count
    print(f"\nStatistics by Speaker Utterance Count:")
    speaker_stats = df.groupby('speaker_count')['similarity'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    speaker_stats = speaker_stats.sort_values('speaker_count')
    
    for _, row in speaker_stats.iterrows():
        print(f"  {row['speaker_count']} utterances: {row['count']} pairs, "
              f"mean={row['mean']:.4f}, std={row['std']:.4f}")
    
    # Top and bottom speakers
    speaker_means = df.groupby('speaker')['similarity'].agg(['mean', 'count']).reset_index()
    speaker_means = speaker_means.sort_values('mean', ascending=False)
    
    print(f"\nTop 5 Most Consistent Speakers (highest average similarity):")
    for i, (_, row) in enumerate(speaker_means.head(5).iterrows()):
        print(f"  {i+1}. {row['speaker']}: {row['mean']:.4f} (n={row['count']})")
    
    print(f"\nTop 5 Least Consistent Speakers (lowest average similarity):")
    for i, (_, row) in enumerate(speaker_means.tail(5).iterrows()):
        print(f"  {i+1}. {row['speaker']}: {row['mean']:.4f} (n={row['count']})")

def main():
    """Main function to analyze speaker similarities."""
    print("DAISY Pipeline - Speaker Similarity Analysis")
    print("=" * 60)
    
    # Set up environment
    setup_figure_environment()
    
    # Analyze speaker similarities for Afrikans
    language = "af"
    
    try:
        # Collect speaker embeddings
        speaker_data = collect_speaker_embeddings(language)
        
        if not speaker_data:
            print("No speaker data found!")
            return
        
        # Compute similarities
        similarities_df = compute_speaker_similarities(speaker_data)
        
        if similarities_df.empty:
            print("No similarity data computed!")
            return
        
        # Create visualizations
        create_similarity_visualizations(similarities_df, language)
        
        # Print statistics
        print_statistics(similarities_df, language)
        
        print(f"\n{'='*60}")
        print("Speaker similarity analysis complete!")
        print(f"Visualizations saved to: figures/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
