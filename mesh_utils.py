import numpy as np
import math
import matplotlib.pyplot as plt

def read_gmsh(filename):
    """Read GMSH mesh file and return nodes and elements"""
    if not filename.endswith('.msh'):
        raise RuntimeError('Unexpected mesh file extension')
    nodes = []
    elems = []
    parse_nodes = False
    parse_elems = False
    
    with open(filename) as msh:
        for line in msh:
            sp_line = line.split()
            if '$Nodes' in line:
                parse_nodes = True
            elif '$Elements' in line:
                parse_nodes = False
                parse_elems = True
            elif parse_nodes and len(sp_line) > 1:
                coords = np.array(sp_line[1:], dtype=np.float64)
                nodes.append(coords)
            elif parse_elems and len(sp_line) > 1:
                eltype = int(sp_line[1])
                inodes = np.array(sp_line[3 + int(sp_line[2]):], dtype=np.int32) - 1
                elems.append(inodes)
    return np.array(nodes), np.array(elems)

def plot_meshes(filenames):
    ncol = 1 + (len(filenames) > 1)
    nrow = math.ceil(len(filenames)/ncol)
    fig = plt.figure(figsize=(6*ncol,6*nrow))
    for idx, filename in enumerate(filenames):
        nodes, elems = read_gmsh(filename)
        ax = fig.add_subplot(nrow, ncol, idx+1)
        ax.triplot(nodes[:,0], nodes[:,1], elems, 'k-', lw=0.2)
        ax.set_title(filename[:-4])
        plt.axis('equal')
        plt.axis('off')

