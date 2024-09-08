import matplotlib.pyplot as plt
import numpy as np


def scatterplot_compare_series(x, y, x_label='', y_label='', title='', size_label=None, legend=False, data_label='', **kwargs_scatter):
    """
    Creates a scatter plot comparing two series with a 45-degree line.
    
    Parameters:
    x (array-like): Data for the x-axis.
    y (array-like): Data for the y-axis.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    title (str): Title of the plot.
    legend (bool): Display legend.
    data_label (str): Label for data points in legend.
    """
    # Create scatter plot
    plt.scatter(x, y, label=data_label, **kwargs_scatter)

    
    # Add a 45-degree line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='45-degree line')

    # Labels and legend
    if size_label is not None:
        plt.xlabel(x_label, fontsize=size_label)
        plt.ylabel(y_label, fontsize=size_label)
    else:
        plt.xlabel(x_label)
        plt.ylabel(y_label)     
    plt.title(title)
    if legend:
        plt.legend()

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Example usage
    x = np.random.rand(100)
    y = x + np.random.normal(0, 0.1, 100)  # y is similar to x with some noise

    scatterplot_compare_series(x, y, x_label='Series 1', y_label='Series 2', title='Scatter Plot of Two Series')
