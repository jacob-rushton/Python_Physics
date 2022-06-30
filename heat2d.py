# This code was difficult, but the resulting movies (one of them, at least)
# speak for themselves.
# I optimized dt and divt to run as fast as possible while still
# successfully converging/diverging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.cla()
# constants first
L = 0.01  # this is the "length" of the box
D = 4.25e-6  # units: m2 s-1
N = 100  # This is the number of points along each side adding to 10000 points
dx = L/N  # m;  change in x
dy = L/N  # m; change in y; a bit redundant but clear and also had errors when
# I tried to set it to just dp and simplify
dt = 5e-4  # s; change in time
divt = 7.5e-4   # s; change in time for divergent animation
t_end = 10+dt
Thi, Tmid, Tlow = 400, 250, 200  # in Kelvin

T_conv = np.empty((N+1, N+1), float)
T_conv[:, :] = Tmid
T_conv[0, :], T_conv[N, :] = Thi, Thi
T_conv[:, 0], T_conv[:, N] = Tlow, Tlow
Tp_conv = np.empty((N+1, N+1), float)
Tp_conv[0, :], Tp_conv[N, :] = Thi, Thi
Tp_conv[:, 0], Tp_conv[:, N] = Tlow, Tlow

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Converged Solution")
ax.set_xlabel("Position (x)")
ax.set_ylabel("Position (y)")
im = ax.imshow(T_conv, origin='lower') 
cb = fig.colorbar(im, label="Temperature (K)")

t = 0.0  # s
c = dt*D
convergence = []
while t < t_end:
    print(t)
# np.roll makes getting f(x+-1,y) and f(x,y+-1) really easy
    lx = np.roll(T_conv, 1, axis=1)
    rx = np.roll(T_conv, -1, axis=1)
    ty = np.roll(T_conv, 1, axis=0)
    by = np.roll(T_conv, -1, axis=0)
    Tp_conv = T_conv + c*(((lx+rx-2*T_conv)/dx**2)+((ty+by-2*T_conv)/dy**2))
    T_conv = np.copy(Tp_conv)
    convergence.append(T_conv)
    t += dt  # s

framedlist = convergence[0:1000:10]  # optimized slice list for convergence



def update_im(i, list):  # animating function
    print(i)
    plt.imshow(list[i])


Conv_movie = animation.FuncAnimation(fig, update_im, repeat=False,
                                 fargs=(framedlist, ))
Conv_movie.save('heat2d_converged.mp4')
plt.clf()
# The following are the original grids for divergence
T_div = np.empty((N+1, N+1), float)
T_div[:, :] = Tmid
T_div[0, :], T_div[N, :] = Thi, Thi
T_div[:, 0], T_div[:, N] = Tlow, Tlow
Tp_div = np.empty((N+1, N+1), float)
Tp_div[0, :], Tp_div[N, :] = Thi, Thi
Tp_div[:, 0], Tp_div[:, N] = Tlow, Tlow

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Diverged Solution")
ax.set_xlabel("Position (x)")
ax.set_ylabel("Position (y)")
im = ax.imshow(T_div, origin='lower') 
cb = fig.colorbar(im, label="Temperature (K)")

t_d = 0.0  # s
f = divt*D
divergence = []
while t_d < t_end:
    print(t_d)
# The same MO for divergence as for convergence; only difference is dt
    lx_d = np.roll(T_div, 1, axis=1)
    rx_d = np.roll(T_div, -1, axis=1)
    ty_d = np.roll(T_div, 1, axis=0)
    by_d = np.roll(T_div, -1, axis=0)
    Tp_div = T_div + f*(((lx_d+rx_d-2*T_div)/dx**2)
                        + ((ty_d+by_d-2*T_div)/dy**2))
    T_div = np.copy(Tp_div)
    divergence.append(T_div)
    t_d += divt  # s

d_framedlist = divergence[0:1000:10]  # optimized list for divergence


Div_movie = animation.FuncAnimation(fig, update_im, repeat=False, frames=100,
                                fargs=(d_framedlist, ))
Div_movie.save('heat2d_diverged.mp4')
plt.clf()
# enjoy the videos :)
