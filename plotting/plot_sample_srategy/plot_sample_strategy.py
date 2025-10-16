from tools.agents.extortion_maps import make_monotone_equispaced_map

import matplotlib.pyplot as plt
import numpy as np



if __name__=="__main__":

    params = [0.1,0.1,0,0.1,0.1]

    map_fn, map_dict = make_monotone_equispaced_map(params)

    plt.rcParams.update({'font.size': 18})
    x_pos = map_dict["x_pos"]
    y_pos = map_dict["y_pos"]
    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, y_pos, marker='o')
    plt.xlabel('Input (normalized)')
    plt.ylabel('Mapped output (normalized)')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("monotone_equispaced_map.png")
    plt.close()