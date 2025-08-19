import os
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import glob


def select_folder():
    """Select folder containing subfolders with SMAPE results"""
    root = tk.Tk()
    root.withdraw()
    
    folder_path = filedialog.askdirectory(
        title="Select Parent Folder Containing Results Subfolders",
        initialdir=os.getcwd()
    )
    
    root.destroy()
    return folder_path


def find_smape_csv_files(parent_folder):
    """Find all smape_results.csv files in subfolders"""
    pattern = os.path.join(parent_folder, "*", "smape_results.csv")
    csv_files = glob.glob(pattern)
    return csv_files


def load_and_label_data(csv_files):
    """Load CSV data and create labels from folder names"""
    datasets = []
    labels = []
    
    for csv_file in csv_files:
        try:
            # Load the CSV
            df = pd.read_csv(csv_file)
            
            # Extract folder name as label
            folder_name = os.path.basename(os.path.dirname(csv_file))
            
            # Ensure required columns exist
            if 'Image Number' in df.columns and 'SMAPE' in df.columns:
                # Sort by image number
                df = df.sort_values('Image Number')
                datasets.append(df)
                labels.append(folder_name)
                print(f"Loaded {len(df)} data points from {folder_name}")
            else:
                print(f"Warning: Invalid CSV format in {csv_file}")
                
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return datasets, labels


def plot_comparison(datasets, labels, output_folder):
    """Plot all datasets for comparison"""
    plt.figure(figsize=(14, 10))
    
    # Color palette for different datasets
    colors = plt.cm.tab10(range(len(datasets)))
    
    for i, (df, label) in enumerate(zip(datasets, labels)):
        plt.loglog(df['Image Number'], df['SMAPE'], 
                  color=colors[i], linewidth=2, label=label, alpha=0.8)
    
    plt.xlabel('Image Number (log scale)')
    plt.ylabel('SMAPE (log scale)')
    plt.title('SMAPE Comparison Across Different Datasets')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot
    plot_file = os.path.join(output_folder, 'smape_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_file}")
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    return plot_file


def print_summary_statistics(datasets, labels):
    """Print summary statistics for all datasets"""
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


def save_combined_csv(datasets, labels, output_folder):
    """Save combined data to a single CSV for further analysis"""
    combined_data = []
    
    for df, label in zip(datasets, labels):
        df_copy = df.copy()
        df_copy['Dataset'] = label
        combined_data.append(df_copy)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        csv_file = os.path.join(output_folder, 'combined_smape_results.csv')
        combined_df.to_csv(csv_file, index=False)
        print(f"Combined results saved to {csv_file}")
        return csv_file
    
    return None


if __name__ == "__main__":
    print("SMAPE Results Comparison Tool")
    print("-" * 40)
    
    # Select parent folder
    parent_folder = select_folder()
    if not parent_folder:
        print("No folder selected. Exiting.")
        exit()
    
    print(f"Selected folder: {parent_folder}")
    
    # Find all SMAPE CSV files
    csv_files = find_smape_csv_files(parent_folder)
    if not csv_files:
        print("No smape_results.csv files found in subfolders.")
        print("Make sure the folder contains subfolders with smape_results.csv files.")
        exit()
    
    print(f"Found {len(csv_files)} SMAPE result files:")
    for csv_file in csv_files:
        subfolder = os.path.basename(os.path.dirname(csv_file))
        print(f"  - {subfolder}")
    
    # Load data
    print("\nLoading data...")
    datasets, labels = load_and_label_data(csv_files)
    
    if not datasets:
        print("No valid datasets loaded. Exiting.")
        exit()
    
    # Plot comparison
    print("\nCreating comparison plot...")
    plot_file = plot_comparison(datasets, labels, parent_folder)
    
    # Save combined CSV
    print("\nSaving combined results...")
    combined_csv = save_combined_csv(datasets, labels, parent_folder)
    
    # Print statistics
    print_summary_statistics(datasets, labels)
    
    print(f"\nComparison complete!")
    print(f"Files saved in: {parent_folder}")
    