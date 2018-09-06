import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def savefig(title):
    fname = title.lower().replace(' ', '_')
    fpath = f'../images/{fname}'
    plt.savefig(fpath, bbox_inches='tight')


def plot_ifs(xy, title):
    fig, ax = plt.subplots()
    ax.scatter(*xy.T, c='green', s=4)
    ax.set_title(title)
    ax.set_axis_off()
    savefig(title)


def get_limits(xy):
    min_vals = np.min(xy, axis=0)
    max_vals = np.max(xy, axis=0)
    limits = np.vstack((min_vals, max_vals))
    return limits.T


def configure_axes(ax, title, limits):
    ax.set_title(title)
    ax.set_xlim(limits[0, :])
    ax.set_ylim(limits[1, :])
    ax.set_axis_off()


class Plotter:
    '''Maintains state for FuncAnimation'''
    
    def __init__(self, xy, title, incr):
        self.xy = xy
        self.incr = incr
        self.title = title
        self.fig, self.ax = plt.subplots()

    def init_ani(self):
        limits = get_limits(self.xy)
        configure_axes(self.ax, self.title, limits)
        self.history = self.ax.scatter([], [], s=4, c='green')
        return self.history,
    
    def step(self, frame):
        stop = self.incr * frame
        self.history.set_offsets(self.xy[:stop])
        return self.history,
    
    def animate(self, savepath):
        num_points = len(self.xy)
        frames = num_points // self.incr

        kwargs1 = dict(fig=self.fig, func=self.step, init_func=self.init_ani)
        kwargs2 = dict(frames=frames, interval=20, repeat=False, blit=True)

        ani = FuncAnimation(**kwargs1, **kwargs2)
        ani.save(savepath, dpi=80, writer='imagemagick')
