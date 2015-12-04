import networkx as nx
import matplotlib.pyplot as plt

from utils import median, hierarchy_pos


class Node(object):
    def __init__(self, left, right, val):
        self.left = left
        self.right = right
        self.val = val

    def is_leaf(self):
        return (self.left is None) and (self.right is None)


class Tree(object):
    def __init__(self, points):
        self.root = Tree.build(points)

    @staticmethod
    def build(points):
        if len(points) == 1:
            return Node(None, None, points[0])
        else:
            points = sorted(points, key=lambda x: x[1])
            m_idx = median(points)
            v_left = Tree.build(points[:m_idx+1])
            v_right = Tree.build(points[m_idx+1:])
            v_val = points[m_idx]
            return Node(v_left, v_right, v_val)

    def get_in_range(self, y_min, y_max):
        results = []
        v_div = self.get_pivot(y_min, y_max)

        if v_div is None:
            return []
        if v_div.is_leaf() and (y_min <= v_div.val[1] <= y_max):
            return [v_div.val]

        above = self.get_above(v_div.left, y_min)
        below = self.get_below(v_div.right, y_max)
        return above + below


    def get_above(self, root, y_min):
        if root is None:
            return []

        if root.is_leaf() and root.val[1] >= y_min:
            return [root.val]

        if root.val[1] < y_min:
            return self.get_above(root.right, y_min)

        return self.get_above(root.left, y_min) + self.get_leaves(root.right)


    def get_below(self, root, y_max):
        if root is None:
            return []

        if root.is_leaf() and root.val[1] <= y_max:
            return [root.val]

        if root.val[1] <= y_max:
            return self.get_below(root.right, y_max) + self.get_leaves(root.left)

        return self.get_below(root.left, y_max)


    def get_pivot(self, y_min, y_max):
        return self.get_pivot_calculator(self.root, y_min, y_max)


    def get_pivot_calculator(self, root, y_min, y_max):
        if root is None:
            return None

        if y_min > root.val[1]:
            return self.get_pivot_calculator(root.right, y_min, y_max)

        if y_max < root.val[1]:
            return self.get_pivot_calculator(root.left, y_min, y_max)

        return root


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

    def print_graph(self, special_nodes):
        G = nx.Graph()
        self.get_graph(self.root, G)
        poss = hierarchy_pos(G, self.root.val[1])
        nx.draw_networkx(G, pos=poss, default=True, node_color='y')

        plt.show()

    def get_graph(self, root, G):
        # G.add_node(root.val[1])
        #
        if root.left is not None:
            n_name = root.left.val if root.left.is_leaf() else root.left.val[1]
            G.add_edge(root.val[1], n_name)
            self.get_graph(root.left, G)

        if root.right is not None:
            n_name = root.right.val if root.right.is_leaf() else root.right.val[1]
            G.add_edge(root.val[1], n_name)
            self.get_graph(root.right, G)