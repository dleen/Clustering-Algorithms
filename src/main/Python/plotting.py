import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg
from matplotlib.patches import Ellipse


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def pie_radius_points(theta1, theta2):
    cx = [0] + np.cos(np.linspace(2.0 * math.pi * theta1,
        2.0 * math.pi * theta2, 10)).tolist()
    cy = [0] + np.sin(np.linspace(2.0 * math.pi * theta1,
        2.0 * math.pi * theta2, 10)).tolist()
    c = list(zip(cx, cy))
    return c


def plot_figure(X, G, r):
    size = 40
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('The conditional probabilities of points belonging to ' +\
        'one of three clusters\n' + 'After 30 iterations:')
    ax.set_xlabel('X dimension')
    ax.set_ylabel('Y dimension')

    for i in range(3):
        plot_cov_ellipse(G.covars[i], G.means[i], nstd=2, ax=ax, alpha=0.2)

    for i, p in enumerate(X):
        ang = np.cumsum(r[i, :])
        ax.scatter(p[0], p[1], marker=(pie_radius_points(0, ang[0]), 0),
            s=size, facecolor='blue')
        ax.scatter(p[0], p[1], marker=(pie_radius_points(ang[0], ang[1]), 0),
            s=size, facecolor='green')
        ax.scatter(p[0], p[1], marker=(pie_radius_points(ang[1], ang[2]), 0),
            s=size, facecolor='red')


    plt.show()
