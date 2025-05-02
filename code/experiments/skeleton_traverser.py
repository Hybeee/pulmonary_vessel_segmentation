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

def get_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def traverse_component(intersection_point, graph, endpoints, edge_segment_start_index, visited_nodes, step_size=3):
    """
    New implementation.
    Intersection can be an edge_segment_point or on a specific edge_segment.
    If intersection_point == edge_segment_point for a given index, then index marks the first edge_segment's index where intersection appears as the starting point
    - or in other words: the index of the second appearance of intersection_point in current_edge
    """

    inner_node, outer_node = endpoints
    prev_node = inner_node
    next_node = outer_node
    current_position = intersection_point

    current_segment_start = -1
    current_segment_end = -1
    current_segment_index = edge_segment_start_index
    remaining_step_size = step_size

    future_nodes = []
    visited_nodes.append(prev_node)

    while next_node is not None:
        current_edge = graph[prev_node][next_node]['pts']

        while current_segment_index < len(current_edge) - 1:
            current_segment_start = current_edge[current_segment_index]
            current_segment_end = current_edge[current_segment_index + 1]

            segment_length = get_distance(current_segment_start, current_segment_end)
            if not np.array_equal(current_position, current_segment_start): # only true for the very first loop, when current_position is initialized as intersection_point
                segment_length = get_distance(current_position, current_segment_end)

            if segment_length == 0:
                current_segment_index += 1
                continue

            if remaining_step_size > segment_length:
                current_segment_index += 1
                remaining_step_size -= segment_length
            else:
                current_position = current_segment_end
                current_segment_index += 1
                remaining_step_size = step_size

            if current_segment_index == (len(current_edge) - 1):
                    current_position = current_segment_end

        visited_nodes.append(next_node)
        next_node_neighbours = [n for n in graph[next_node].keys() if n != prev_node]
        next_node_neighbours = [(next_node, n) for n in next_node_neighbours if n not in visited_nodes]

        if len(next_node_neighbours) == 0:
            if len(future_nodes) == 0:
                break
            else:
                remaining_step_size = 0
        else:
            future_nodes = next_node_neighbours + future_nodes
        
        prev_node, next_node = future_nodes[0]
        future_nodes = future_nodes[1:]
        current_position = prev_node # probably redundant


def traverse_component_old(intersection_point, graph, visited_nodes, endpoints, edge_segment_start_index, step_size=1):
    """
    Implements a DFS for traversing current component.
    NOTE: if two starting points have the same node as their closest nodes they might be on the same edge. What happens in this case? probably gets explored through one of
    the nodes and then the second one stops immeaditely? Not this function's responsibility.

    starting_point in this case is the point of intersection.
    closest_node_id is the node closest to the intersection point.
    Traversal should be done in this direction.

    How it works:
        - step on the edge with step_size until a node ('next node') is reached.
            - This can be done by considering on which edge segment we're currently are, taking end-start as a vector and stepping in that direction
                - If this step doesnt reach end we step again
                - If step reaches the end then we save the remaining step_size and take the next line segment as our path, thus start' = end, end' = next edge point in
                  graph[prev_node][next_node]['pts']
            - If next node - in which case end = next_node - is reached then we update the already traversed node's list with next_node and
              set prev_node = next_node and update the to-be-traversed list with next_node's (now set as prev_node) neighbours.
              The new next_node is the first element in to-be-traversed list.
            - This is done until to-be-traversed is empty.
            - NOTE: If in step 2 next_node's (now set as prev_node) list of neighbours is empty then the remaining step_size is set back to step_size, since we're resuming
              the traversal from a different point, thus not continuing the current path.
    """

    inner_node, outer_node = endpoints # inner node being inside bounding box and outer_node being outside the bounding box
    prev_node = inner_node
    next_node = outer_node
    current_position = intersection_point

    current_segment_start = -1
    current_segment_end = -1
    current_segment_index = edge_segment_start_index
    remaining_step_size = 0 # NOTE: is this needed? If not handled different step_sizes might occur if edge_length % step_size != 0

    future_nodes = [] # when reaching next node: .append(list(graph[next_node].keys())) BEFORE setting next_node as prev node
    visited_nodes.append(prev_node) # in reality prev node is never visited - used so that traversal doesn't continue this way.

    while next_node is not None:
        current_edge = graph[prev_node][next_node]['pts'] # list of segments

        # TRAVERSING EDGE BETWEEN PREV_NODE AND NEXT_NODE
        while current_segment_index < len(current_edge) - 1: # current_edge[len(current_edge) - 2] <=> -2nd element. -> the starting point of the last edge.
            current_segment_start = current_edge[current_segment_index]
            current_segment_end = current_edge[current_segment_index + 1]

            segment_vector = current_segment_end - current_segment_start
            segment_length = get_distance(current_segment_start, current_segment_end)
            segment_unit_vector = segment_vector / segment_length

            # just to make sure
            if segment_length == 0:
                current_segment_index += 1
                continue

            if remaining_step_size > 0:
                if remaining_step_size > segment_length:
                    current_segment_index += 1
                    remaining_step_size -= segment_length
                else:
                    current_position = current_segment_start + segment_unit_vector * remaining_step_size
                    remaining_step_size = 0
            else:
                remaining_length = get_distance(current_position, current_segment_end)
                if step_size > remaining_length:
                    current_segment_index += 1
                    remaining_step_size = step_size - remaining_length
                else:
                    current_position += segment_unit_vector * step_size
        
        # When the program exits the edge traversing loop current_position = next_node
        visited_nodes.append(next_node)
        next_node_neighbours = [n for n in graph[next_node].keys() if n != prev_node]
        next_node_neighbours = [(next_node, n) for n in next_node_neighbours if n not in visited_nodes] # maybe remaining_step_size should be saved here!

        if len(next_node_neighbours) == 0:
            if len(future_nodes) == 0:
                break
            else:
                remaining_step_size = 0
        else:
            future_nodes = next_node_neighbours + future_nodes
        
        prev_node, next_node = future_nodes[0]
        current_position = prev_node

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

    print(f"Number of filtered nodes: {len(graph.nodes)} - {relevant_nodes.shape[0]} = {len(graph.nodes) - relevant_nodes.shape[0]}")

    edge_points = graph[0][7]['pts']

    print(edge_points)

    # print(f"First: {edge_points[0]}\nLast: {edge_points[-1]}")
    # print(f"Node 1: {graph.nodes[0]['o']}\nNode 2: {graph.nodes[7]['o']}")


if __name__ == "__main__":
    main()