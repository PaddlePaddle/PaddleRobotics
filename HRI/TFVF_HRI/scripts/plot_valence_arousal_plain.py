import os
import codecs
import numpy as np
import matplotlib.pyplot as plt

Has_Header = True
CSV = 'data/valence_arousal_exp.csv'


def calculate_mean_variance(data):
    theta = np.arctan(data[:, 0] / data[:, 1])
    m_x = np.mean(np.cos(theta))
    m_y = np.mean(np.sin(theta))
    mu = np.arctan(m_y / m_x)

    R = np.sqrt(m_x ** 2 + m_y ** 2)
    sigma = np.sqrt(-2 * np.log(R))
    return mu, sigma


def filled_arc(center, radius, theta1, theta2, color):
    # Ref: https://stackoverflow.com/a/30642704
    phi = np.linspace(theta1, theta2, 100)
    x = center[0] + radius * np.cos(phi)
    y = center[1] + radius * np.sin(phi)

    # Equation of the chord
    m = (y[-1] - y[0]) / (x[-1] - x[0])
    c = y[0] - m * x[0]
    y2 = m * x + c

    # Plot the filled arc
    plt.fill_between(x, y, y2, facecolor=color, edgecolor='none', alpha=0.5)


def filled_sector(center, radius, theta1, theta2, color):
    filled_arc(center, radius, theta1, theta2, color)

    # Fill triangle
    x_0, y_0 = center
    x_1 = center[0] + radius * np.cos(theta1)
    y_1 = center[1] + radius * np.sin(theta1)

    x_2 = center[0] + radius * np.cos(theta2)
    y_2 = center[1] + radius * np.sin(theta2)

    plt.fill([x_0, x_1, x_2, x_0], [y_0, y_1, y_2, y_0], facecolor=color,
             edgecolor='none', alpha=0.5)


def plot(name_lst, group_lst, mu_lst, sigma_lst):
    cx, cy = 5.0, 5.0
    colors = ['red', 'blue']
    markers = ['x', '+']
    linestyles = ['r-', 'b--']

    bg_img = plt.imread('data/28-affect-words.png')
    # plt.imshow(bg_img, extent=[-0.5, 10.5, -0.5, 10.5])
    plt.imshow(bg_img, extent=[-0.2, 10.2, 0.1, 9.9])

    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 4.8
    x = radius * np.cos(theta) + cx
    y = radius * np.sin(theta) + cy
    plt.plot(x, y, color='black')

    for name, group, mu, sigma, color, marker, linestyle in \
        zip(name_lst, group_lst, mu_lst, sigma_lst, colors, markers, linestyles):
        plt.plot(group[:, 0], group[:, 1], marker, label=name, color=color)

        ex = cx + radius * np.cos(mu)
        ey = cy + radius * np.sin(mu)
        plt.plot([cx, ex], [cy, ey], linestyle)

        for d_mu in [-sigma, sigma]:
            ex = cx + radius * np.cos(mu + d_mu)
            ey = cy + radius * np.sin(mu + d_mu)
            plt.plot([cx, ex], [cy, ey], linestyle='-', color='black')

        filled_sector([cx, cy], radius, mu - sigma, mu + sigma, color)

    plt.axis('equal')
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend(loc='lower left', bbox_to_anchor=(0.65, 0.0))
    plt.savefig('valence_arousal_plain.pdf', bbox_inches='tight')
    plt.show()


group_1, group_2 = [], []
with codecs.open(CSV, 'r', 'utf-8') as f:
    for line in f.readlines():
        if Has_Header:
            Has_Header = False
            continue

        eps = np.random.random(2) * 0.1
        data = line.strip().split(',')
        if int(data[0]) == 1:
            group_1.append((int(data[2]) + eps[0], int(data[3]) + eps[1]))
        elif int(data[0]) == 2:
            group_2.append((int(data[2]) + eps[0], int(data[3]) + eps[1]))

group_1 = np.array(group_1)
group_2 = np.array(group_2)

mu_1, sigma_1 = calculate_mean_variance(group_1)
mu_2, sigma_2 = calculate_mean_variance(group_2)

plot(['Reactive HRI', 'TFVT-HRI'], [group_2, group_1], [mu_2, mu_1], [sigma_2, sigma_1])
