import numpy as np
from plotter import Plotter, plot_ifs


def get_random_transform(transforms):
    choice = np.random.choice(transforms.shape[0], p=transforms[:, -1])
    transform = transforms[choice, :-1]
    abcd = transform[:4].reshape((2, 2))
    ef = transform[4:6]
    return abcd, ef


def ifs(x0, y0, transforms, num_iters):
    xy = np.hstack((x0, y0))
    XY = [xy]
   
    for i in range(num_iters):
        abcd, ef = get_random_transform(transforms)
        xy = np.matmul(abcd, xy) + ef
        XY.append(xy)
    return np.array(XY)


def load_transforms():
    barnsley = np.loadtxt('../transforms/barnsley.csv')
    von_koch = np.loadtxt('../transforms/von_koch.csv')
    crystal = np.loadtxt('../transforms/crystal.csv')

    transforms = [barnsley, von_koch, crystal]
    return transforms


def get_savepath(title):
    fname = title.lower().replace(' ', '_')
    savepath = f'../images/{fname}.gif'
    return savepath


def main():
    transforms = load_transforms()
    titles = ['Barnsley', 'von Koch', 'Crystal']
    N = [3000, 2000, 2000]

    for transform, title, n in zip(transforms, titles, N):
        xy = ifs(0, 0, transform, n)
        plotter = Plotter(xy, title, incr=20)
        savepath = get_savepath(title)
        plotter.animate(savepath)


if __name__ == '__main__':
    main()
