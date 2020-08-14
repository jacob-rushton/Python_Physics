# Okay, so this code is SUPPOSED to demonstrate that the MB distribution is an
# inevitable and inescapable occurence; much like Pumpkin Spice lattes at
# Starbucks in the fall. I'm demonstrating this with the "particle in a box"
# model with N=400 particles

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import combinations
from scipy.optimize import curve_fit

npoint = 400  # amount of particles in my box
nframe = 1000  # number of frames for my beautiful movie
xmin, xmax, ymin, ymax = 0, 1, 0, 1  # length and width of my box
fig, ax = plt.subplots()
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
Dt = 0.00002  # s; time step
r = 0.00001  # m; radius of each particle
m = 2.672E-26  # kg
kB = 1.38064852E-23  # m**2 kg s**-2 K**-1


def update_point(num):
    global x, y, vx, vy
    print(num)
    indx = np.where((x < xmin) | (x > xmax))
    indy = np.where((y < ymin) | (y > ymax))
    vx[indx] = -vx[indx]  # sets v=0 outside of box, keeping particles trapped
    vy[indy] = -vy[indy]
    xx = np.asarray(list(combinations(x, 2)))
    yy = np.asarray(list(combinations(y, 2)))
    dd = (xx[:, 0]-xx[:, 1])**2+(yy[:, 0]-yy[:, 1])**2
    a = np.arange(400)  # array of indices useful for collision calculation
    aa = np.asarray(list(combinations(a, 2)))  # index => same order as xx & yy
    for i in range(len(dd)-1):
        v_1 = np.array([vx[aa[i, 0]], vy[aa[i, 0]]])
        v_2 = np.array([vx[aa[i, 1]], vy[aa[i, 1]]])
        p_1 = np.array([xx[i, 0], yy[i, 0]])
        p_2 = np.array([xx[i, 1], yy[i, 1]])
        if dd[i] <= 2*r:  # would mean particles are within each other, so a
            # collision must have taken place
            v_1 = v_1-np.dot((v_1-v_2), (p_1-p_2))*(p_1-p_2)/np.dot((p_1-p_2),
                                                                    (p_1-p_2))
            v_2 = v_2-np.dot((v_2-v_1), (p_2-p_1))*(p_2-p_1)/np.dot((p_2-p_1),
                                                                    (p_2-p_1))
            vx[aa[i, 0]] = v_1[0]  # calculating new velocities post-collision
            vy[aa[i, 0]] = v_1[1]
            vx[aa[i, 1]] = v_2[0]
            vy[aa[i, 1]] = v_2[1]

    dx = Dt*vx  # simple, velocity times time interval = change in distance
    dy = Dt*vy
    x = x+dx
    y = y+dy
    data = np.stack((x, y), axis=-1)
    im.set_offsets(data)
    return im,  # returns the updated data


x = np.random.random(npoint)  # initial positions for each particle
y = np.random.random(npoint)
vx = -500.*np.ones(npoint)  # initial velocities for each particle
vy = np.zeros(npoint)
color = np.where(x < 0.5, 'b', 'r')  # sets one side blue and other side red
vx[np.where(x <= 0.5)] = -vx[np.where(x <= 0.5)]
s = np.array([10])  # size of particles; it'd be cooler if they were bigger tho
im = ax.scatter(x, y, color=color)  # initial plot to be updated & animated
im.set_sizes(s)
animation = animation.FuncAnimation(fig, update_point, nframe, interval=10,
                                    repeat=False)
animation.save('collisions.mp4')
plt.cla()  # let's keep the dots and lines separate, shall we?

v_mag = np.sqrt(vx**2 + vy**2)  # m/s magnitude of each velocity vector
E_mag = 0.5*m*v_mag**2  # J; kinetic energy calculated with each magnitude of v


def f(v, T):  # velocity disstribution function
    return ((m*v)/(kB*T))*np.exp(-(0.5*m*v**2)/(kB*T))


def g(E, T):  # energy distribution function
    return (1/(kB*T))*np.exp(-E/(kB*T))


plt.subplot(311)  # 3 rows so the two plots don't overlap each other
v_n, v_bins, v_patches = plt.hist(v_mag, bins='auto', density=True)
v_bin_centres = (v_bins[:-1]+v_bins[1:])/2  # so #bins=#n
v_popt, v_pcov = curve_fit(f, v_bin_centres, v_n, p0=200)
# I have to plot both graphs with the same domain to get both on the same fig
plt.plot(v_bin_centres, f(v_bin_centres, *v_popt), label='Curve Fit')
plt.legend(loc='best')  # This puts the legend where it blocks the least data
plt.title('MB Distribution of velocity')
plt.xlabel('Magnitude of Velocity (m/s)')
plt.ylabel('Frequency')  # it's just the no. of times each value shows up

plt.subplot(313)
E_n, E_bins, E_patches = plt.hist(E_mag, bins='auto', density=True)
E_bin_centres = (E_bins[:-1]+E_bins[1:])/2
E_popt, E_cov = curve_fit(g, E_bin_centres, E_n, p0=200)
plt.plot(E_bin_centres, f(E_bin_centres, *E_popt), label='Curve Fit')
plt.legend(loc='best')  # gives my legend the best location
plt.title("Energy Distribution")  # attractive labels for axes and graph
plt.xlabel("Energy (J)")
plt.ylabel('Frequency')
plt.savefig('distributions.pdf')
plt.show()

w = open('collisions.txt', 'w')  # open a new file
w.write('The Temperature is {} Kelvin'.format(v_popt))
# don't forget the units!
w.close()  # if the file isn't closed it isn't saved
