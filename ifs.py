import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_random_transform(transforms):
    choice = np.random.choice(transforms.shape[0], p=transforms[:, -1])
    transform = transforms[choice, :-1]
    abcd = transform[:4].reshape((2, 2))
    ef = transform[4:6]
    return abcd, ef


def ifs(x0, y0, transforms, num_iters=1000):
    xy = np.hstack((x0, y0))
    XY = [xy]
   
    for i in range(num_iters):
        abcd, ef = get_random_transform(transforms)
        xy = np.matmul(abcd, xy) + ef
        XY.append(xy)
    return np.array(XY)


def plot_ifs(xy, title):
    fig, ax = plt.subplots()
    ax.scatter(*xy.T, c='green', s=10)
    ax.set_title(title)
    ax.set_axis_off()
    fname = title.lower().replace(' ', '_')
    plt.savefig('images/{}'.format(fname), bbox_inches='tight')


def load_transforms():
    barnsley = np.loadtxt('transforms/barnsley.csv')
    von_koch = np.loadtxt('transforms/von_koch.csv')
    crystal = np.loadtxt('transforms/crystal.csv')

    transforms = [barnsley, von_koch, crystal]
    return transforms


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
        self.history = self.ax.scatter([], [], s=10, c='green')
        return self.history,
    
    def step(self, frame):
        stop = self.incr * frame
        self.history.set_offsets(self.xy[:stop])
        return self.history,
    
    def animate(self, savepath):
        num_points = len(self.xy)
        frames = num_points // self.incr

        kwargs1 = dict(fig=self.fig, func=self.step, init_func=self.init_ani)
        kwargs2 = dict(frames=frames, interval=50, repeat=False, blit=True)

        ani = FuncAnimation(**kwargs1, **kwargs2)
        ani.save(savepath, dpi=80, writer='imagemagick')


def main():
    transforms = load_transforms()
    titles = ['Barnsley', 'von Koch', 'Crystal']

    for transform, title in zip(transforms, titles):
        xy = ifs(0, 0, transform)
        fname = title.lower().replace(' ', '_')
        plotter = Plotter(xy, title, incr=10)
        plotter.animate(f'./images/{fname}.gif')


if __name__ == '__main__':
    main()
