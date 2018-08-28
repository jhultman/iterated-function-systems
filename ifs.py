import numpy as np
import matplotlib.pyplot as plt


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


def plot_ifs(xy, transform, title):
    fig, ax = plt.subplots()
    ax.scatter(*xy.T, c='green', s=10)
    ax.set_title(title)
    ax.set_axis_off()
    fname = title.lower().replace(' ', '_')
    plt.savefig('images/{}'.format(fname), bbox_inches='tight')


def main():
    barnsley = np.loadtxt('transforms/barnsley.csv')
    von_koch = np.loadtxt('transforms/von_koch.csv')
    crystal = np.loadtxt('transforms/crystal.csv')

    transforms = [barnsley, von_koch, crystal]
    titles = ['Barnsley', 'von Koch', 'Crystal']

    for transform, title in zip(transforms, titles):
        xy = ifs(0, 0, transform)
        plot_ifs(xy, transform, title)


if __name__ == '__main__':
    main()
