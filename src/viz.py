import matplotlib.pyplot as plt

ACCENT = "#5A8DEE"       
SERIES_WARM = "#E8A44C"  
REFERENCE = "#C7D0E0"    
MUTED = "#6B7688"        
SAND = "#E9C46A"         
POSITIVE = "#3FB950"     
NEGATIVE = "#F85149"     

_TEXT = "#C3CAD6"        
_GRID = "#8A94A8"        


def apply_theme() -> None:
    plt.rcParams.update({
        # Transparent figure + axes so PNGs blend into the dark cards behind them.
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,

        # Light, low-contrast text and axis furniture.
        "text.color": _TEXT,
        "axes.labelcolor": _TEXT,
        "axes.titlecolor": _TEXT,
        "xtick.color": _TEXT,
        "ytick.color": _TEXT,
        "axes.edgecolor": _GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Subtle gridlines behind the data.
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": _GRID,
        "grid.alpha": 0.14,
        "grid.linewidth": 0.7,

        # Legends.
        "legend.frameon": False,
        "legend.labelcolor": _TEXT,

        # Typography / sizes.
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "figure.dpi": 130,
    })
