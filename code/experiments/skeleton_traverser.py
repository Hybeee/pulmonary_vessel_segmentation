import sknw
import SimpleITK as sitk
import numpy as np

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

def main():
    skeleton = sitk.ReadImage('dataset/skeleton/005_vein_mask_skeleton.nii.gz')
    skeleton = sitk.GetArrayFromImage(skeleton)
    intersections = sitk.ReadImage('dataset/intersections/005_vessel_intersections_bbox.nii.gz')
    intersections = sitk.GetArrayFromImage(intersections)
    bboxs = [[slice(103, 223, None), slice(179, 275, None), slice(164, 241, None)],
            [slice(117, 225, None), slice(187, 294, None), slice(282, 369, None)]]

    graph = get_graph(skeleton=skeleton)

    # print(graph.nodes[0]['o'])

    nodes = np.array([graph.nodes[node_id]['o'] for node_id in graph.nodes])

    # Stores nodes which are relevant wrt. traversing the graph
    relevant_nodes = filter_nodes(nodes, bboxs)
    
    print(f"Number of filtered nodes: {len(graph.nodes) - relevant_nodes.shape[0]}")

if __name__ == "__main__":
    main()