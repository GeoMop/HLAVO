import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import contextily as ctx
import pyproj


class InteractiveMapPlotter:
    def __init__(self, df, data_selector):
        """
        Parameters:
        -----------
        df : polars.DataFrame
            Initial DataFrame with columns 'time', 'lon', 'lat', and physical quantities
            (e.g., 'temp', 'perc', 'humidity', 'wind_speed', 'insol').
        data_selector : dict
            A dictionary with keys 'lon', 'lat', 'time' used to hold the currently selected point.
        """
        self.df = df
        self.data_selector = data_selector

        # List of physical quantities that will be shown as stripes.
        self.quantities = ['temp', 'perc', 'humidity', 'wind_speed', 'insol']

        # Predefine colormaps for each physical quantity.
        self.colormaps = {
            'temp': plt.cm.coolwarm,  # Temperature: cool-to-warm colors
            'perc': plt.cm.Blues,  # Precipitation: blues
            'humidity': plt.cm.Greens,  # Humidity: greens
            'wind_speed': plt.cm.Oranges,  # Wind speed: oranges
            'insol': plt.cm.YlOrBr  # Insolation: yellow-orange-brown
        }

        # Create figure and axis.
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # List of squares; each is a dict with keys: index, x, y, row, rect_patch.
        self.squares = []
        # NumPy array storing rectangle bounds: each row is [left, right, bottom, top].
        self.rect_bounds = None

        # Persistent hover elements.
        self.hover_patch = patches.Rectangle((0, 0), 0, 0, linewidth=2, edgecolor='yellow',
                                             facecolor='none', visible=False)
        self.ax.add_patch(self.hover_patch)
        self.hover_annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 20), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
            visible=False)

        # Keep track of selected and highlighted squares.
        self.selected_sq = None
        self.highlighted_sq = None

        # Connect events.
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)

        # Initial plot.
        self._update_plot()

    def set_rect_patch_style(self, sq, selected=False, highlighted=False):
        """
        Update the style of the square's rectangle patch.

        Parameters:
        -----------
        sq : dict
            The square dictionary containing 'rect_patch'.
        selected : bool
            If True, use the "selected" style.
        highlighted : bool
            If True, use the "highlighted" style.

        Priority: selected > highlighted. If selected is True, the square will be red.
        """
        if selected:
            sq['rect_patch'].set_edgecolor('red')
            sq['rect_patch'].set_linewidth(2)
        elif highlighted:
            sq['rect_patch'].set_edgecolor('yellow')
            sq['rect_patch'].set_linewidth(2)
        else:
            sq['rect_patch'].set_edgecolor('black')
            sq['rect_patch'].set_linewidth(1)

    def _update_plot(self):
        """
        Internal method to update the plot based on self.df and self.data_selector.
        Called when new data is provided.
        """
        # Convert Polars DataFrame to pandas.
        df_pd = self.df.to_pandas()

        # If time is not set, default to the first available time.
        if self.data_selector.get('time') is None:
            self.data_selector['time'] = df_pd['time'].iloc[0]

        selected_time = self.data_selector['time']
        self.df_selected = df_pd[df_pd['time'] == selected_time]
        if self.df_selected.empty:
            raise ValueError(f"No data available for time = {selected_time}")

        # Normalize each quantity based on the full dataset.
        self.norms = {}
        for q in self.quantities:
            vmin = df_pd[q].min()
            vmax = df_pd[q].max()
            self.norms[q] = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Transform coordinates from EPSG:4326 (lon, lat) to EPSG:3857.
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.df_selected['x'], self.df_selected['y'] = zip(*self.df_selected.apply(
            lambda row: transformer.transform(row['lon'], row['lat']), axis=1))

        # Compute bounding box with a 10% margin.
        margin = 0.1
        min_x, max_x = self.df_selected['x'].min(), self.df_selected['x'].max()
        min_y, max_y = self.df_selected['y'].min(), self.df_selected['y'].max()
        dx = max_x - min_x
        dy = max_y - min_y
        self.bbox = [min_x - margin * dx, max_x + margin * dx, min_y - margin * dy, max_y + margin * dy]

        # Clear the axis and redraw the basemap.
        self.ax.cla()
        self.ax.set_xlim(self.bbox[0], self.bbox[1])
        self.ax.set_ylim(self.bbox[2], self.bbox[3])
        ctx.add_basemap(self.ax, crs="EPSG:3857")

        # Re-add hover elements (they are removed on cla()).
        self.ax.add_patch(self.hover_patch)
        self.ax.add_artist(self.hover_annotation)

        # Define square size as a fraction of the bounding box width.
        self.square_size = (self.bbox[1] - self.bbox[0]) / 50

        # Draw squares and build rectangle bounds.
        self.squares = []
        rect_bounds_list = []
        for idx, row in self.df_selected.iterrows():
            x, y = row['x'], row['y']
            # Draw square and obtain its outer rectangle patch.
            rect_patch = self.draw_square_with_stripes(x, y, row)
            left = x - self.square_size / 2
            right = x + self.square_size / 2
            bottom = y - self.square_size / 2
            top = y + self.square_size / 2
            rect_bounds_list.append([left, right, bottom, top])
            sq = {'index': idx, 'x': x, 'y': y, 'row': row, 'rect_patch': rect_patch}
            # Set initial style.
            self.set_rect_patch_style(sq, selected=False, highlighted=False)
            self.squares.append(sq)
        self.rect_bounds = np.array(rect_bounds_list)

        # Reset selected and highlighted squares.
        self.selected_sq = None
        self.highlighted_sq = None

        self.fig.canvas.draw()

    def draw_square_with_stripes(self, x, y, row):
        """
        Draws a square at (x, y) subdivided into vertical stripes.
        Returns the outer rectangle patch.
        """
        left = x - self.square_size / 2
        bottom = y - self.square_size / 2

        # Create and add the outer rectangle patch (default style will be updated later).
        rect_patch = patches.Rectangle((left, bottom), self.square_size, self.square_size,
                                       linewidth=1, edgecolor='black', facecolor='none')
        self.ax.add_patch(rect_patch)

        # Draw vertical stripes for each physical quantity.
        n = len(self.quantities)
        stripe_width = self.square_size / n
        for i, q in enumerate(self.quantities):
            val = row[q]
            norm_val = self.norms[q](val)
            color = self.colormaps[q](norm_val)
            stripe_left = left + i * stripe_width
            stripe = patches.Rectangle((stripe_left, bottom), stripe_width, self.square_size,
                                       facecolor=color, edgecolor=None)
            self.ax.add_patch(stripe)

        return rect_patch

    def on_click(self, event):
        """
        Event handler for mouse clicks.
        Uses vectorized hit testing and then updates only the changed squares'
        border styles using set_rect_patch_style.
        """
        if event.inaxes != self.ax:
            return

        click_x, click_y = event.xdata, event.ydata
        bounds = self.rect_bounds  # shape (N,4)
        lefts = bounds[:, 0]
        rights = bounds[:, 1]
        bottoms = bounds[:, 2]
        tops = bounds[:, 3]
        mask = (click_x >= lefts) & (click_x <= rights) & (click_y >= bottoms) & (click_y <= tops)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            idx = indices[0]
            new_selected = self.squares[idx]
            row = new_selected['row']
            self.data_selector['lon'] = row['lon']
            self.data_selector['lat'] = row['lat']
            self.data_selector['time'] = row['time']
            print(f"Selected point: lon={row['lon']}, lat={row['lat']}, time={row['time']}")

            # Update only the affected squares: old selected, old highlighted, and new selected.
            if self.selected_sq is not None and self.selected_sq != new_selected:
                # Revert old selected to normal (or highlighted if it's still hovered).
                self.set_rect_patch_style(self.selected_sq,
                                          selected=False,
                                          highlighted=(self.selected_sq == self.highlighted_sq))
            self.selected_sq = new_selected
            self.set_rect_patch_style(new_selected, selected=True)

            self.fig.canvas.draw_idle()

    def on_hover(self, event):
        """
        Event handler for mouse motion.
        Uses vectorized hit testing to determine the newly hovered square and updates
        border styles only for the affected squares.
        """
        if event.inaxes != self.ax:
            # If cursor left axes, revert any highlighted square.
            if self.highlighted_sq is not None:
                # If the highlighted square is selected, keep red; otherwise, revert to normal.
                self.set_rect_patch_style(self.highlighted_sq,
                                          selected=(self.highlighted_sq == self.selected_sq),
                                          highlighted=False)
                self.highlighted_sq = None
                self.hover_patch.set_visible(False)
                self.hover_annotation.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        hover_x, hover_y = event.xdata, event.ydata
        bounds = self.rect_bounds
        lefts = bounds[:, 0]
        rights = bounds[:, 1]
        bottoms = bounds[:, 2]
        tops = bounds[:, 3]
        mask = (hover_x >= lefts) & (hover_x <= rights) & (hover_y >= bottoms) & (hover_y <= tops)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            new_highlight = self.squares[indices[0]]
            if new_highlight != self.highlighted_sq:
                # Revert previous highlighted square.
                if self.highlighted_sq is not None:
                    self.set_rect_patch_style(self.highlighted_sq,
                                              selected=(self.highlighted_sq == self.selected_sq),
                                              highlighted=False)
                self.highlighted_sq = new_highlight
                # Only update style if not selected.
                if new_highlight != self.selected_sq:
                    self.set_rect_patch_style(new_highlight, selected=False, highlighted=True)

                # Update hover patch and annotation.
                x, y = new_highlight['x'], new_highlight['y']
                left = x - self.square_size / 2
                bottom = y - self.square_size / 2
                self.hover_patch.set_xy((left, bottom))
                self.hover_patch.set_width(self.square_size)
                self.hover_patch.set_height(self.square_size)
                self.hover_patch.set_visible(True)

                # Prepare annotation text.
                row = new_highlight['row']
                text = "\n".join([f"{q}: {row[q]:.2f}" for q in self.quantities])
                ann_x = left + self.square_size / 2
                ann_y = bottom + self.square_size
                self.hover_annotation.xy = (ann_x, ann_y)
                self.hover_annotation.set_text(text)
                self.hover_annotation.set_visible(True)
        else:
            # If no square is hovered, revert previously highlighted square.
            if self.highlighted_sq is not None:
                self.set_rect_patch_style(self.highlighted_sq,
                                          selected=(self.highlighted_sq == self.selected_sq),
                                          highlighted=False)
                self.highlighted_sq = None
            self.hover_patch.set_visible(False)
            self.hover_annotation.set_visible(False)
        self.fig.canvas.draw_idle()

    def update_data(self, new_df):
        """
        Update the plot with a new Polars DataFrame.

        Parameters:
        -----------
        new_df : polars.DataFrame
            New DataFrame to use for updating the plot.
        """
        self.df = new_df
        self._update_plot()