import os
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import glob

def select_folder():
    """Select parent folder containing result subfolders"""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(
        title="Select Parent Folder Containing Results Subfolders",
        initialdir=os.getcwd()
    )
    root.destroy()
    return folder_path

def select_subfolders(parent_folder):
    """Let user choose which subfolders to include"""
    all_subfolders = [f for f in os.listdir(parent_folder)
                      if os.path.isdir(os.path.join(parent_folder, f))]
    
    print("\nAvailable subfolders:")
    for idx, folder in enumerate(all_subfolders, 1):
        print(f"{idx}: {folder}")
    
    chosen = input("\nEnter the numbers of the subfolders to include "
                   "(comma-separated, e.g. 1,3,5): ")
    chosen_indices = [int(x.strip()) - 1 for x in chosen.split(",") if x.strip().isdigit()]
    
    selected_folders = [os.path.join(parent_folder, all_subfolders[i]) 
                        for i in chosen_indices if 0 <= i < len(all_subfolders)]
    
    return selected_folders

def find_smape_csv_files(selected_folders):
    """Find smape_results.csv files in the selected subfolders"""
    csv_files = []
    for folder in selected_folders:
        candidate = os.path.join(folder, "smape_results.csv")
        if os.path.exists(candidate):
            csv_files.append(candidate)
    return csv_files

def load_and_label_data(csv_files):
    """Load CSV data and create labels from folder names"""
    datasets, labels = [], []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            folder_name = os.path.basename(os.path.dirname(csv_file))
            if 'Image Number' in df.columns and 'SMAPE' in df.columns:
                df = df.sort_values('Image Number')
                datasets.append(df)
                labels.append(folder_name)
                print(f"Loaded {len(df)} points from {folder_name}")
            else:
                print(f"Warning: Invalid CSV format in {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    return datasets, labels

def plot_comparison(datasets, labels, output_file, title):
    """Plot datasets for comparison"""
    plt.figure(figsize=(9, 8))
    colors = plt.cm.tab10(range(len(datasets)))
    for i, (df, label) in enumerate(zip(datasets, labels)):
        plt.loglog(df['Image Number'], df['SMAPE'], 
                   color=colors[i], linewidth=4, label=label, alpha=0.8)
    plt.xlabel('Image Number (log scale)', fontsize=26)
    plt.ylabel('SMAPE (log scale)', fontsize=26)
    plt.title(title, fontsize=30)
    plt.grid(True, alpha=0.3, which='both')
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.legend(loc='upper right', fontsize=24)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")
    plt.show()

def print_summary_statistics(datasets, labels):
    """Print stats for datasets"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for df, label in zip(datasets, labels):
        smape_values = df['SMAPE']
        print(f"\n{label}:")
        print(f"  Images: {len(df)}")
        print(f"  Mean SMAPE: {smape_values.mean():.6f}")
        print(f"  Min SMAPE:  {smape_values.min():.6f}")
        print(f"  Max SMAPE:  {smape_values.max():.6f}")
        print(f"  Std SMAPE:  {smape_values.std():.6f}")

def save_combined_csv(datasets, labels, output_file):
    """Save combined results"""
    combined_data = []
    for df, label in zip(datasets, labels):
        df_copy = df.copy()
        df_copy['Dataset'] = label
        combined_data.append(df_copy)
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined results saved to {output_file}")

if __name__ == "__main__":
    print("SMAPE Results Comparison Tool")
    print("-" * 40)
    
    parent_folder = select_folder()
    if not parent_folder:
        print("No folder selected. Exiting.")
        exit()

    selected_folders = select_subfolders(parent_folder)
    if not selected_folders:
        print("No subfolders selected. Exiting.")
        exit()
    
    csv_files = find_smape_csv_files(selected_folders)
    if not csv_files:
        print("No smape_results.csv found in selected subfolders.")
        exit()
    
    print("\nLoading data...")
    datasets, labels = load_and_label_data(csv_files)
    if not datasets:
        print("No valid datasets loaded. Exiting.")
        exit()
    
    plot_title = input("\nEnter the title for the plot: ").strip()
    if not plot_title:
        plot_title = "SMAPE Comparison Across Selected Datasets"

    output_name = input("\nEnter base name for output files (without extension): ").strip()
    if not output_name:
        output_name = "smape_comparison"
    
    plot_file = os.path.join(parent_folder, output_name + ".png")
    combined_csv = os.path.join(parent_folder, output_name + ".csv")
    
    plot_comparison(datasets, labels, plot_file, plot_title)
    save_combined_csv(datasets, labels, combined_csv)
    print_summary_statistics(datasets, labels)
    
    print("\nDone!")
