# graph_utils.py
from shapely.geometry import LineString, box
import networkx as nx

def build_graph(detections, lines):
    G = nx.Graph()

    # Add nodes
    for d in detections:
        G.add_node(d['id'], bbox=d['bbox'])

    # Add edges if line intersects bounding boxes
    for l in lines:
        line = LineString([(l[0], l[1]), (l[2], l[3])])
        connected_nodes = []
        for node, data in G.nodes(data=True):
            x1, y1, x2, y2 = data['bbox']
            bbox_poly = box(x1, y1, x2, y2)
            if line.intersects(bbox_poly):
                connected_nodes.append(node)

        # Connect all nodes intersected by the line
        for i in range(len(connected_nodes)):
            for j in range(i + 1, len(connected_nodes)):
                G.add_edge(connected_nodes[i], connected_nodes[j])

    return G
