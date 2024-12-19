import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk


def load_file():
    """
    Open a file dialog to select an Excel or CSV file, then load it into a DataFrame.
    Returns the DataFrame.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df


def prepare_and_plot(df, y_column='Total Chirps'):
    """
    Prepare data and generate a violin plot.
    """
    plt.figure(figsize=(12, 6))
    # Filter data for plotting
    plot_data = df[['Experiment', 'Subject', 'File', y_column]]

    # Create a violin plot
    sns.violinplot(data=plot_data, x='Experiment', y=y_column, hue='Subject', split=True, scale='count',
                   inner='quartile')
    plt.title(f'Violin Plot of {y_column} by Experiment for each Subject')
    plt.xlabel("Experiment Type (LL vs LD)")
    plt.ylabel(y_column)
    plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()



df = load_file()
prepare_and_plot(df)
