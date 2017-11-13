#coding: utf-8
import networkx as nx
import matplotlib.pyplot as plt

#グラフの宣言
G = nx.Graph()

#属性付きのノードを作成
G.add_node(1, color = 'red')
G.add_node(2, color = 'green')
G.add_node(3, color = 'green')
G.add_node(4, color = 'green')
G.add_node(5, color = 'green')
G.add_node(6, color = 'green')
G.add_node(7, color = 'green')
G.add_node(8, color = 'green')
G.add_node(9, color = 'green')

#エッジの追加
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(4,5)
G.add_edge(5,6)
G.add_edge(6,1)

G.add_edge(1,7)
G.add_edge(7,8)
G.add_edge(7,9)

nx.draw_networkx(G)
plt.show()
