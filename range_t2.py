import networkx as nx
import matplotlib.pyplot as plt
import random

from range_t1           import Tree
from utils              import hierarchy_pos, median
from multiprocessing    import Process


class Node2D(object):
    def __init__(self, left, right, val, aStruct):
        self.left = left
        self.right = right
        self.val = val
        self.aStruct = aStruct

    def is_leaf(self):
        return (self.left is None) and (self.right is None)


class Tree2D(object):

    def __init__(self, points):
        self.root = self.build(points)

    @staticmethod
    def build(pts):
        if len(pts) == 1:
            return Node2D(None, None, pts[0], Tree(pts))
        else:
            pts = sorted(pts, key= lambda x:x[0])

            m_idx = median(pts)
            xSmallerOrEqual = pts[:m_idx+1]
            xLarger = pts[m_idx+1:]
            v_left = Tree2D.build(xSmallerOrEqual)
            v_right = Tree2D.build(xLarger)
            associatedStr = Tree(pts)
            v_val = pts[m_idx]
            return Node2D(v_left, v_right, v_val, associatedStr)

    def get_in_range(self, x_min, x_max, y_min, y_max):
        results = []

        v_div = self.get_pivot(x_min, x_max)

        if v_div is None:
            return []
        if v_div.is_leaf() and (y_min <= v_div.val[1] <= y_max) and (x_min <= v_div.val[0] <= x_max):
            return [v_div]

        above = self.get_above(v_div.left, x_min, y_min, y_max)
        below = self.get_below(v_div.right, x_max, y_min, y_max)
        return above + below, v_div

    def get_above(self, root, x_min, y_min, y_max):

        if root is None:
            return []

        if root.is_leaf() and root.val[0] >= x_min and (y_min <= root.val[1] <= y_max):
            return [root.val]

        elif root.is_leaf():
            return []

        if root.val[0] < x_min:
            return self.get_above(root.right, x_min, y_min, y_max)

        return self.get_above(root.left, x_min, y_min, y_max) + root.right.aStruct.get_in_range(y_min, y_max)


    def get_below(self, root, x_max, y_min, y_max):
        if root is None:
            return []

        if root.is_leaf() and root.val[0] <= x_max and (y_min <= root.val[1] <= y_max):
            return [root.val]

        if root.val[0] <= x_max:
            bel = self.get_below(root.right, x_max, y_min, y_max)
            lft = root.left.aStruct.get_in_range(y_min, y_max) if root.left else []
            return bel + lft
        return self.get_below(root.left, x_max, y_min, y_max)

    def get_pivot(self, x_min, x_max):
        return self.get_pivot_calculator(self.root, x_min, x_max)

    def get_pivot_calculator(self, root, x_min, x_max):
        if root is None:
            return None

        if x_min > root.val[0]:
            return self.get_pivot_calculator(root.right, x_min, x_max)
        if x_max < root.val[0]:
            return self.get_pivot_calculator(root.left, x_min, x_max)

        return root

    def get_node_by_values(self, x, y):
        return self.get_node_by_values_helper(self.root, x, y)

    def get_node_by_values_helper(self, root, x, y):
        if root is None:
            raise ValueError('Wrong value of x and y')

        if root.val[0]==x:
            return root

        if x>root.val[0]:
            return self.get_node_by_values_helper(root.right, x, y)

        else:
            return self.get_node_by_values_helper(root.left, x, y)

    def get_leaves(self, root):
        lvs = []
        if root is None:
            return []

        if not (root.left is None):
            lvs = lvs + self.get_leaves(root.left)

        if root.is_leaf():
            lvs.append(root.val)

        if not (root.right is None):
            lvs = lvs + self.get_leaves(root.right)

        return lvs


    def print_graph(self, pts, leaves, v_div, special_nodes=None):
        G = nx.Graph()
        self.get_graph(self.root, G)
        poss = hierarchy_pos(G, self.root.val[0], dim=1)

        from pylab import rcParams
        rcParams['figure.figsize'] = 20, 10

        if special_nodes:
            nx.draw_networkx_nodes(G, pos=poss, nodelist=special_nodes, node_color='b',node_size=900)


        nx.draw_networkx(G, pos=poss, default=True, node_color='y', with_labels=False)

        if v_div.is_leaf():
            nx.draw_networkx_nodes(G, poss, nodelist=[v_div.val], size = 500, color='r2')
        else:
            nx.draw_networkx_nodes(G, poss, nodelist=[v_div.val[0]], size = 500, color='r2')


        internal = []
        for x in pts:
            if x in leaves:
                a, b = poss[x]
                plt.text(a,b-0.08,s=str(x[0]) + ', ' + str(x[1]), bbox=dict(facecolor='red', alpha=0.5),horizontalalignment='center')
            else:
                internal.append(x)
        labs = {}



        for k, v in poss.items():
            if type(k) == int:
                labs[k] = poss[k]
            else:
                labs[k] = (200,200)
        nx.draw_networkx_labels(G, pos=labs)
        plt.show()


    def get_graph(self, root, G):
        if root.left is not None:
            n_name = root.left.val if root.left.is_leaf() else root.left.val[0]
            G.add_edge(root.val[0], n_name)
            self.get_graph(root.left, G)

        if root.right is not None:
            n_name = root.right.val if root.right.is_leaf() else root.right.val[0]
            G.add_edge(root.val[0], n_name)
            self.get_graph(root.right, G)

wayPointsAreGenerated = int(raw_input('Points read from: \n(1) File \n(2) Generated'))

pts = []
if wayPointsAreGenerated == 1:
    with open(raw_input('name of the file:')) as input:
        lines = input.readlines()
        for line in lines:
            pts.append(tuple(map(int, line.split(','))))

else:
    SIZE = int(raw_input('Amount of points'))
    a = range(SIZE)
    random.shuffle(a)
    b = range(SIZE)
    random.shuffle(b)
    pts = [(a[i],b[i]) for i in xrange(SIZE)]

tree = Tree2D(pts)
lvs = tree.get_leaves(tree.root)

xmin, xmax, ymin, ymax = map(int, raw_input('xmin xmax ymin ymax:').split())

a, v_div = tree.get_in_range(xmin,xmax,ymin,ymax)


res1 = sorted(a, key = lambda x: x[0])
print 'results: ', res1

p = Process(target=tree.print_graph, args=(pts,lvs, v_div, res1))
p.start()
x,y  = map(int, raw_input('x and y of point: ').split())

nod = tree.get_node_by_values(x,y)
nod.aStruct.print_graph(None)
print nod.val
p.join()