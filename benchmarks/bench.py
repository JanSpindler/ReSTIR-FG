import OpenEXR
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import Imath
import numpy as np
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def select_exr_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an Reference OpenEXR file",
        filetypes=[("OpenEXR files", "*.exr")],
        initialdir=os.getcwd()
    )

    root.destroy()
    return file_path


def select_folder():
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(
        title="Select Rendered Image Folder",
        initialdir=os.getcwd()
    )

    root.destroy()
    return folder_path


def load_exr_image(file_path):
    try:
        exr_file = OpenEXR.InputFile(file_path)
        header = exr_file.header()

        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        (r, g, b) = exr_file.channels("RGB", FLOAT)

        r = np.frombuffer(r, dtype=np.float32).reshape((height, width))
        g = np.frombuffer(g, dtype=np.float32).reshape((height, width))
        b = np.frombuffer(b, dtype=np.float32).reshape((height, width))

        rgb_image = np.stack([r, g, b], axis=2)
        exr_file.close()
        return rgb_image
    except Exception as e:
        print(f"Error loading EXR image: {e}")
        exit()


def calc_mse(reference, test):
    return np.mean((reference - test) ** 2)


def calc_mape(reference, test, epsilon=1e-8):
    reference_safe = np.where(np.abs(reference) < epsilon, epsilon, reference)
    ape = np.abs((reference - test) / reference_safe)
    return np.mean(ape)


def calc_smape(reference, test, epsilon=1e-8):
    numerator = np.abs(reference - test)
    denominator = (np.abs(reference) + np.abs(test)) / 2.0
    denominator = np.maximum(denominator, epsilon)
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = numerator / denominator
    smape = smape[np.isfinite(smape)]
    return np.mean(smape)


def extract_number_from_filename(filename):
    match = re.search(r'img\.(\d+)\.exr', filename)
    if match:
        return int(match.group(1))
    return None


if __name__ == "__main__":
    # Select reference
    exr_file = select_exr_file()
    if not exr_file:
        messagebox.showerror("Error", "No EXR file selected.")
        exit()
    print(f"Selected EXR file: {exr_file}")
    reference_image = load_exr_image(exr_file)
    if reference_image is None:
        messagebox.showerror("Error", "Failed to load reference image.")
        exit()

    # Select rendered images folder
    rendered_folder = select_folder()
    if rendered_folder is None:
        messagebox.showerror("Error", "No folder selected.")
        exit()
    print(f"Selected rendered images folder: {rendered_folder}")

    # Load rendered files
    pattern = os.path.join(rendered_folder, "img.*.exr")
    rendered_files = glob.glob(pattern)
    rendered_files.sort()
    if rendered_files is None:
        messagebox.showerror("Error", "No rendered EXR files found in the selected folder.")
        exit()
    print(f"Found {len(rendered_files)} rendered EXR files.")

    # Compare
    results = []
    for rendered_file in tqdm(rendered_files):
        filename = os.path.basename(rendered_file)
        number = extract_number_from_filename(filename)
        if number is None:
            print(f"Could not extract number from filename: {filename}")
            continue

        rendered_image = load_exr_image(rendered_file)
        if rendered_image is None:
            print(f"Failed to load rendered image: {rendered_file}")
            continue

        if rendered_image.shape != reference_image.shape:
            print(f"Image shape mismatch for {filename}: {rendered_image.shape} vs {reference_image.shape}")
            continue

        mape = calc_smape(reference_image, rendered_image)
        results.append((number, mape))

    # Store CSV
    if results:
        df = pd.DataFrame(results, columns=['Image Number', 'SMAPE'])
        csv_file = os.path.join(rendered_folder, 'smape_results.csv')
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")

        # Plot the results
        plt.figure(figsize=(12, 8))
        plt.loglog(df['Image Number'], df['SMAPE'], 'b-', linewidth=2, markersize=4)
        plt.xlabel('Image Number')
        plt.ylabel('SMAPE')
        plt.title('SMAPE Error vs Image Number')
        plt.grid(True, alpha=0.3, which='both')
                
        # Save plot
        plot_file = os.path.join(rendered_folder, 'smape_plot.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
        
        # Show plot
        plt.show()
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Number of images processed: {len(df)}")
        print(f"Mean SMAPE: {df['SMAPE'].mean():.6f}")
        print(f"Min SMAPE: {df['SMAPE'].min():.6f}")
        print(f"Max SMAPE: {df['SMAPE'].max():.6f}")
