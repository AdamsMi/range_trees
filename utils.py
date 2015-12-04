
def special_comp(x, y):
    a = x[1] if type(x) is tuple else x
    b = y[1] if type(y) is tuple else y

    return 1 if a > b else -1 if a < b else 0


def special_comp2(x, y):
    a = x[0] if type(x) is tuple else x
    b = y[0] if type(y) is tuple else y

    return 1 if a > b else -1 if a < b else 0

def hierarchy_pos(G, root, width=3., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                  pos=None, parent=None, dim=0):
    width = width
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    if dim==0:
        neighbors = sorted(G.neighbors(root), cmp=special_comp)
    else:
        neighbors = sorted(G.neighbors(root), cmp=special_comp2)
    if parent is not None:
        neighbors.remove(parent)

    if len(neighbors) != 0:
        dx = width/len(neighbors)
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos,
                                parent=root, dim=dim)
    return pos


def median(points):
    l = len(points)
    if l % 2 == 0:
        return l/2 - 1

    return l/2
