import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


# Class to handle zoom and synchronization
class MultiZoomer:
    def __init__(self, fig, axes):
        self.fig = fig
        self.axes = np.atleast_2d(axes)
        self.cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        #print(self.axes[0, 0].get_xlim())
        self.x_range = self.axes[0, 0].get_xlim()
        spans = [365, 30, 2]
        self.spans = (self.x_range[1] - self.x_range[0]) * np.array([365, 30, 1]) / 365.0
        #print("Range:", self.x_range)
        self.x_center = (self.x_range[0] + self.x_range[1]) / 2

        self.update()

    def update(self):
        # Update each plot to center around the clicked time point
        for ax_row in self.axes:
            for ax, span in zip(ax_row, self.spans):
                # Adjust limits while respecting the data range
                new_xmin = max(self.x_range[0], self.x_center - span/2)
                new_xmax = min(self.x_range[1], self.x_center + span/2)
                #print(new_xmin, new_xmax, span)
                ax.set_xlim(new_xmin, new_xmax)
            # Custom tick locators and formatters
            # axes[0]: Month minor ticks for month starts, month shortcut labels, major tick for year start
            ax_row[0].xaxis.set_major_locator(mdates.YearLocator())  # Major ticks at year start
            ax_row[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Major tick format: Year
            ax_row[0].xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks at month start
            ax_row[0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))  # Minor tick format: Month shortcut
            ax_row[0].tick_params(axis='x', which='minor', rotation=45, labelsize=8, pad=15)  # Rotate minor tick labels
    
            # ax_row[1]: Major tick for month start, minor tick for days, labeled by day in month
            ax_row[1].xaxis.set_major_locator(mdates.MonthLocator())  # Major ticks at month start
            ax_row[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Major tick format: Month and year
            ax_row[1].xaxis.set_minor_locator(mdates.DayLocator())  # Minor ticks at day start
            ax_row[1].xaxis.set_minor_formatter(mdates.DateFormatter('%d'))  # Minor tick format: Day in month
            ax_row[1].tick_params(axis='x', which='minor', labelsize=7)  # Rotate minor tick labels

            # ax_row[2]: Major tick for day, minor tick for hours
            ax_row[2].xaxis.set_major_locator(mdates.DayLocator())  # Major ticks at day start
            ax_row[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))  # Major tick format: Day and month
            ax_row[2].xaxis.set_minor_locator(mdates.HourLocator())  # Minor ticks every hour
            ax_row[2].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))  # Minor tick format: Hour
            ax_row[2].tick_params(axis='x', which='minor', labelsize=7)  # Rotate minor tick labels

        # Redraw the canvas
        self.fig.canvas.draw()

    def onclick(self, event):
        #print(event.inaxes)
        #if event.inaxes not in self.axes:
        #    return
        # Determine the zoom factor based on the button clicked
        if event.button == 1 and event.xdata is not None:  # Left click to zoom in
            self.x_center = event.xdata
            self.update()
        else:
            return


def zoom_plot_df(df):
    df.index = pd.to_datetime(df.index)
    cols = df.columns
    # Set up the figure and grid of axes (3 columns)
    fig, axes = plt.subplots(len(cols), 3, figsize=(30, 6), constrained_layout=True, sharey='row', sharex='col')
    #fig.subplots_adjust(wspace=0.3)

    # Plot data with different spans
    for ax_row, col in zip(axes, cols):
        for ax in ax_row:
            ax.plot(mdates.date2num(df.index), df[col])
            range = df[col].min(), df[col].max()
            assert range[0] < range[1], f"Col={col}, Invalid range: {range}"
            ax.set_ylim(*range)
            ax.grid()
            #print("X range", ax.get_xlim())
        ax_row[0].set_ylabel(col)

    labels = ["Year", "Month", "Day"]
    for i, ax in enumerate(axes[0]):
        ax.set_title(labels[i])

    # Instantiate the multi-zoom handler
    zoom_handler = MultiZoomer(fig, axes)
    plt.show()

#################################
if __name__ == '__main__':
    # Generate some early data
    dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i, hours=j) for i in range(365) for j in range(24)]
    x  = np.linspace(0.0, 365.0, len(dates))
    #data = np.sin(2 * np.pi * x/365) + np.sin(2 * np.pi * (x % 365))
    data =  x / 365 + np.sin(2 * np.pi * (x % 365))
    df = pd.DataFrame({'time':dates, 'val':data, 'v1':2*data})
    df = df.set_index('time')

    zoom_plot_df(df)


