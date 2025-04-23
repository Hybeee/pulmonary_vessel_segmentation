import sknw
import SimpleITK as sitk
import numpy as np
from viewer_3d import point_on_segment

def get_graph(skeleton):
    return sknw.build_sknw(skeleton)

def filter_nodes_helper(node, bboxs):
    for bbox in bboxs:
        z_start, z_end = bbox[0].start, bbox[0].stop
        y_start, y_end = bbox[1].start, bbox[1].stop
        x_start, x_end = bbox[2].start, bbox[2].stop

        if (x_start <= node[2] <= x_end) and (y_start <= node[1] <= y_end) and (z_start <= node[0] <= z_end):
            return True

    return False


def filter_nodes(nodes, bboxs):
    filtered_nodes = []

    for node in nodes:
        if not filter_nodes_helper(node, bboxs):
            filtered_nodes.append(node)

    return np.array(filtered_nodes)

def traverse_component(starting_point, closest_node_id):
    """
    starting_point in this case is the point of intersection.
    closest_node_id is the node closest to the intersection point.
    Traversal should be done in this direction.
    """

def main():
    skeleton = sitk.ReadImage('dataset/skeleton/005_vein_mask_skeleton.nii.gz')
    skeleton = sitk.GetArrayFromImage(skeleton)
    intersections = sitk.ReadImage('dataset/intersections/005_vessel_intersections_bbox.nii.gz')
    intersections = sitk.GetArrayFromImage(intersections)
    bboxs = [[slice(103, 223, None), slice(179, 275, None), slice(164, 241, None)],
            [slice(117, 225, None), slice(187, 294, None), slice(282, 369, None)]]

    graph = get_graph(skeleton=skeleton)

    # print(graph.nodes[0]['o'])

    # nodes = np.array([graph.nodes[node_id]['o'] for node_id in graph.nodes])

    # Stores nodes which are relevant wrt. traversing the graph
    # relevant_nodes = filter_nodes(nodes, bboxs)

    # print(f"Number of filtered nodes: {len(graph.nodes) - relevant_nodes.shape[0]}")

    edge_points = graph[0][7]['pts']

    # print(f"First: {edge_points[0]}\nLast: {edge_points[-1]}")
    # print(f"Node 1: {graph.nodes[0]['o']}\nNode 2: {graph.nodes[7]['o']}")

    print(f"{edge_points[0]}\n{edge_points[1]}")

    point_on_line = edge_points[0] + 0.5 * (edge_points[1] - edge_points[0])
    print(point_on_line)

    print(point_on_segment(point_on_line, edge_points[0], edge_points[1]))

if __name__ == "__main__":
    main()