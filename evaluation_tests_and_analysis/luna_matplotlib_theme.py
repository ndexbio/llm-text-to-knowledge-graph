import matplotlib.pyplot as plt
import numpy as np

# Define global visual style
custom_theme = {
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 9,
    'legend.edgecolor': 'black',
    'legend.frameon': False,
    'lines.linewidth': 0.5,
    'font.family': ['Arial', 'sans-serif'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 'truetype',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
}


# Apply the theme to matplotlib globally
def apply_luna_theme():
    for key, value in custom_theme.items():
        plt.rcParams[key] = value
