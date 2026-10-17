import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple


# Set style globally
sns.set_style("whitegrid")
sns.set_palette("husl")


def bar_chart(
    df: pd.DataFrame,
    cat_col: str,
    ax: Optional[plt.Axes] = None,
    top_n: Optional[int] = None,
    orient: str = "v",
    title: Optional[str] = None,
    color: str = "steelblue",
) -> plt.Axes:
    """
    Create a bar chart for categorical data.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    cat_col : str
        Name of categorical column
    ax : plt.Axes, optional
        Matplotlib axis to plot on
    top_n : int, optional
        Show only top N categories by frequency
    orient : str
        'v' for vertical (default), 'h' for horizontal
    title : str, optional
        Plot title
    color : str
        Bar color

    Returns:
    --------
    plt.Axes
        Matplotlib axis with plot
    """
    if ax is None:
        ax = plt.gca()

    # Validate column
    if cat_col not in df.columns:
        raise ValueError(f"Column '{cat_col}' not found in dataframe")

    # Get value counts
    counts = df[cat_col].value_counts()

    # Filter to top_n if specified
    if top_n is not None and len(counts) > top_n:
        counts = counts.head(top_n)

    # Plot
    if orient == "h":
        sns.barplot(y=counts.index, x=counts.values, ax=ax, color=color)
    else:
        sns.barplot(x=counts.index, y=counts.values, ax=ax, color=color)

    # Styling
    ax.set_xlabel(cat_col, fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title(
        title or f"Distribution of {cat_col}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    # Rotate x labels if vertical
    if orient == "v" and len(counts) > 5:
        ax.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=2)

    return ax


def histogram(
    df: pd.DataFrame,
    num_col: str,
    ax: Optional[plt.Axes] = None,
    bins: int = 30,
    title: Optional[str] = None,
    color: str = "skyblue",
    edgecolor: str = "black",
    kde: bool = True,
) -> plt.Axes:
    """
    Create a histogram for numerical data.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    num_col : str
        Name of numerical column
    ax : plt.Axes, optional
        Matplotlib axis to plot on
    bins : int
        Number of bins (default: 30)
    title : str, optional
        Plot title
    color : str
        Histogram color
    edgecolor : str
        Edge color of bars
    kde : bool
        Whether to overlay KDE curve (default: True)

    Returns:
    --------
    plt.Axes
        Matplotlib axis with plot
    """
    if ax is None:
        ax = plt.gca()

    # Validate column
    if num_col not in df.columns:
        raise ValueError(f"Column '{num_col}' not found in dataframe")

    # Remove NaN values
    data = df[num_col].dropna()

    # Plot histogram
    ax.hist(
        data,
        bins=bins,
        color=color,
        edgecolor=edgecolor,
        alpha=0.7,
        density=kde,
    )

    # Overlay KDE if requested
    if kde:
        data.plot.kde(ax=ax, color="red", linewidth=2, label="KDE")
        ax.legend()

    # Styling
    ax.set_xlabel(num_col, fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title(
        title or f"Distribution of {num_col}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    # Add statistics as text
    stats_text = f"μ={data.mean():.2f}\nσ={data.std():.2f}\nn={len(data)}"
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return ax


def box_plot(
    df: pd.DataFrame,
    num_col: str,
    cat_col: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    orient: str = "v",
    palette: str = "Set2",
) -> plt.Axes:
    """
    Create a box plot for numerical data, optionally grouped by categorical.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    num_col : str
        Name of numerical column
    cat_col : str, optional
        Name of categorical column for grouping
    ax : plt.Axes, optional
        Matplotlib axis to plot on
    title : str, optional
        Plot title
    orient : str
        'v' for vertical (default), 'h' for horizontal
    palette : str
        Seaborn color palette

    Returns:
    --------
    plt.Axes
        Matplotlib axis with plot
    """
    if ax is None:
        ax = plt.gca()

    # Validate columns
    if num_col not in df.columns:
        raise ValueError(f"Column '{num_col}' not found in dataframe")

    if cat_col and cat_col not in df.columns:
        raise ValueError(f"Column '{cat_col}' not found in dataframe")

    # Plot
    if cat_col:
        sns.boxplot(
            data=df,
            x=cat_col if orient == "v" else num_col,
            y=num_col if orient == "v" else cat_col,
            ax=ax,
            palette=palette,
        )
    else:
        sns.boxplot(data=df[[num_col]], ax=ax, palette=palette)

    # Styling
    y_label = num_col if orient == "v" else (cat_col or "")
    x_label = (cat_col or "") if orient == "v" else num_col

    if y_label:
        ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
    if x_label:
        ax.set_xlabel(x_label, fontsize=11, fontweight="bold")

    ax.set_title(
        title
        or (f"{num_col} by {cat_col}" if cat_col else f"Distribution of {num_col}"),
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    if orient == "v" and cat_col and len(df[cat_col].unique()) > 5:
        ax.tick_params(axis="x", rotation=45)

    return ax


def line_plot(
    df: pd.DataFrame,
    num_col: str,
    time_col: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = "steelblue",
    marker: str = "o",
    linewidth: float = 2,
) -> plt.Axes:
    """
    Create a line plot for time series or trend data.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (should be sorted by time_col)
    num_col : str
        Name of numerical column to plot
    time_col : str
        Name of time/x-axis column
    ax : plt.Axes, optional
        Matplotlib axis to plot on
    title : str, optional
        Plot title
    color : str
        Line color
    marker : str
        Marker style
    linewidth : float
        Width of line

    Returns:
    --------
    plt.Axes
        Matplotlib axis with plot
    """
    if ax is None:
        ax = plt.gca()

    # Validate columns
    if num_col not in df.columns:
        raise ValueError(f"Column '{num_col}' not found in dataframe")
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in dataframe")

    # Plot
    ax.plot(
        df[time_col],
        df[num_col],
        color=color,
        marker=marker,
        linewidth=linewidth,
        markersize=5,
        alpha=0.8,
    )

    # Styling
    ax.set_xlabel(time_col, fontsize=11, fontweight="bold")
    ax.set_ylabel(num_col, fontsize=11, fontweight="bold")
    ax.set_title(
        title or f"{num_col} over {time_col}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3)

    # Rotate x labels if many time points
    if len(df) > 10:
        ax.tick_params(axis="x", rotation=45)

    return ax


# ==================== ADDITIONAL HELPFUL FUNCTIONS ====================


def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: Optional[plt.Axes] = None,
    hue: Optional[str] = None,
    size: Optional[str] = None,
    title: Optional[str] = None,
    alpha: float = 0.6,
) -> plt.Axes:
    """
    Create a scatter plot for relationship exploration.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    x_col : str
        X-axis column
    y_col : str
        Y-axis column
    ax : plt.Axes, optional
        Matplotlib axis
    hue : str, optional
        Column for color coding
    size : str, optional
        Column for point size
    title : str, optional
        Plot title
    alpha : float
        Transparency level

    Returns:
    --------
    plt.Axes
    """
    if ax is None:
        ax = plt.gca()

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue,
        size=size,
        ax=ax,
        alpha=alpha,
    )

    ax.set_title(
        title or f"{y_col} vs {x_col}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel(x_col, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=11, fontweight="bold")

    return ax


def violin_plot(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    palette: str = "Set2",
) -> plt.Axes:
    """
    Create a violin plot (distribution + box plot).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    num_col : str
        Numerical column
    cat_col : str
        Categorical column for grouping
    ax : plt.Axes, optional
        Matplotlib axis
    title : str, optional
        Plot title
    palette : str
        Color palette

    Returns:
    --------
    plt.Axes
    """
    if ax is None:
        ax = plt.gca()

    sns.violinplot(
        data=df,
        x=cat_col,
        y=num_col,
        ax=ax,
        palette=palette,
    )

    ax.set_title(
        title or f"Distribution of {num_col} by {cat_col}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel(cat_col, fontsize=11, fontweight="bold")
    ax.set_ylabel(num_col, fontsize=11, fontweight="bold")

    if len(df[cat_col].unique()) > 5:
        ax.tick_params(axis="x", rotation=45)

    return ax


def correlation_heatmap(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Correlation Matrix",
    cmap: str = "coolwarm",
    annot: bool = True,
) -> plt.Axes:
    """
    Create a correlation heatmap for numerical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (will auto-select numerical columns)
    ax : plt.Axes, optional
        Matplotlib axis
    title : str
        Plot title
    cmap : str
        Colormap name
    annot : bool
        Whether to annotate correlation values

    Returns:
    --------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numerical columns found in dataframe")

    # Compute correlation
    corr = numeric_df.corr()

    # Plot heatmap
    sns.heatmap(
        corr,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        center=0,
        ax=ax,
        cbar_kws={"label": "Correlation"},
        vmin=-1,
        vmax=1,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    return ax


def missing_data_plot(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Missing Data",
) -> plt.Axes:
    """
    Visualize missing data across columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    ax : plt.Axes, optional
        Matplotlib axis
    title : str
        Plot title

    Returns:
    --------
    plt.Axes
    """
    if ax is None:
        ax = plt.gca()

    # Calculate missing percentage
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)

    if missing_pct.empty:
        ax.text(0.5, 0.5, "No missing values!", ha="center", va="center")
        return ax

    # Plot
    missing_pct.plot(kind="barh", ax=ax, color="coral", edgecolor="black")
    ax.set_xlabel("Missing %", fontsize=11, fontweight="bold")
    ax.set_ylabel("Column", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    # Add percentage labels
    for i, v in enumerate(missing_pct):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    return ax


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_df = pd.DataFrame({
        "age": np.random.normal(35, 15, 200),
        "salary": np.random.normal(60000, 20000, 200),
        "department": np.random.choice(["Sales", "Engineering", "HR", "Marketing"], 200),
        "experience": np.random.randint(1, 20, 200),
        "date": pd.date_range("2023-01-01", periods=200, freq="D"),
    })

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("EDA Dashboard Example", fontsize=16, fontweight="bold", y=1.00)

    # Plot examples
    bar_chart(sample_df, "department", ax=axes[0, 0])
    histogram(sample_df, "salary", ax=axes[0, 1])
    box_plot(sample_df, "salary", "department", ax=axes[0, 2])
    scatter_plot(sample_df, "experience", "salary", ax=axes[1, 0])
    violin_plot(sample_df, "salary", "department", ax=axes[1, 1])
    missing_data_plot(sample_df, ax=axes[1, 2])

    plt.tight_layout()
    plt.show()
