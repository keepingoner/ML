# -*- encoding:utf-8 -*-
"""
Points[
    {x:0.0,y:0.0},
    {x:2.0,y:0.0},
    {x:2.0,y:2.0},
    {x:0.0,y:2.0}]

Doors[
{start_x:0.0,
start_y :0.5,
end_x:0.0,
end_y:1.3},]

Windows[{start_x:0.0,
start_y :0.5,
end_x:0.0,
end_y:1.3},]

"""

import matplotlib.pyplot as plt

x_home = [0, 0, 2, 2, 2]
y_home = [0, 2, 2, 0, 0]

x_door = [0, 0, ]
y_door = [0.5, 1.5]


x_window = [2, 2]
y_window = [0.5, 1.3]


plt.figure()
plt.plot(x_home, y_home, 'm--')
plt.plot(x_door, y_door, 'g-')
plt.plot(x_window, y_window, 'b-')

plt.show()

