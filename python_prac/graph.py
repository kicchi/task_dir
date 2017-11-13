#coding: utf-8
import networkx as nx
import matplotlib.pyplot as plt

#グラフの宣言
G = nx.DiGraph()

#属性付きのノードを作成
G.add_node(1)
G.add_node(2)

#エッジの追加
G.add_edge(1,2)

nx.draw_networks(G)
plt.show()
