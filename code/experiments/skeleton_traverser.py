import sknw
import SimpleITK as sitk
import numpy as np
from viewer_3d import point_on_segment, add_graph_to_plot
import pyvista as pv
from collections import defaultdict

class Intersection:
    """
    Stores three intersection related points:
        - Point of intersection (point in the 3d space)
        - Endpoints of the edge on which the point of intersection is:
            - previous node - found inside the bounding box (id)
            - next node - found outside the bounding box (id)
    """

    def __init__(self, intersection, prev_node, next_node, segment_index, is_node):
        self.intersection = intersection
        self.prev_node = prev_node
        self.next_node = next_node
        self.segment_index = segment_index
        self.is_node = is_node

    def __str__(self):
        return f"Intersection: {self.intersection}\nNode: {self.is_node}"

def get_graph(skeleton):
    return sknw.build_sknw(skeleton)

def is_node_inside_bboxs(node, bboxs):
    """
    Determines whether a given node is inside a bounding box - returns True - or not - returns False.
    Points of the bounding boxes are considered as False.
    Similar to filter_nodes_helper, this uses a stronger condition.
    """
    for bbox in bboxs:
        z_start, z_end = bbox[0].start, bbox[0].stop
        y_start, y_end = bbox[1].start, bbox[1].stop
        x_start, x_end = bbox[2].start, bbox[2].stop

        if (x_start < node[2] < x_end) and (y_start < node[1] < y_end) and (z_start < node[0] < z_end):
            return True

    return False

def filter_nodes_helper(node, bboxs):
    """
    Determines whether a given node is inside a bounding box - returns True - or not - returns False.
    Points of the bounding boxes are considered as True.
    """
    for bbox in bboxs:
        z_start, z_end = bbox[0].start, bbox[0].stop
        y_start, y_end = bbox[1].start, bbox[1].stop
        x_start, x_end = bbox[2].start, bbox[2].stop

        if (x_start <= node[2] <= x_end) and (y_start <= node[1] <= y_end) and (z_start <= node[0] <= z_end):
            return True

    return False

def filter_nodes_helper_2(node, bboxs):
    """
    Determines whether a given node is inside a bounding box - returns True - or not - returns False.
    Points of the bounding boxes are considered as True.
    """
    for bbox in bboxs:
        z_start, z_end = bbox[0].start, bbox[0].stop
        y_start, y_end = bbox[1].start, bbox[1].stop
        x_start, x_end = bbox[2].start, bbox[2].stop

        if (x_start < node[2] < x_end) and (y_start < node[1] < y_end) and (z_start < node[0] < z_end):
            return True

    return False

def filter_nodes(graph, bboxs):
    result = []

    for node in graph.nodes:
        node_coord = graph.nodes[node]['o']
        if not filter_nodes_helper(node_coord, bboxs):
            result.append(node)
    
    return np.array(result)

def filter_nodes_old(nodes, bboxs):
    filtered_nodes = []

    for node in nodes:
        if not filter_nodes_helper(node, bboxs):
            filtered_nodes.append(node)

    return np.array(filtered_nodes)

def is_node_intersection(node, intersections):
    for intersection in intersections:
        if np.array_equal(node, intersection):
            return True
    
    return False

def create_intersection_objects(intersections, graph, bboxs) -> np.ndarray[Intersection]:
    result = []
    z, y, x = np.where(intersections > 0)
    intersections = np.column_stack((z, y, x))

    for intersection in intersections:
        found = False
        for (u, v) in graph.edges:
            current_edge = graph[u][v]['pts']
            if len(current_edge) == 0:
                continue
            
            # Intersection point is a node -> segment_index = 0
            if (np.array_equal(intersection, graph.nodes[u]['o']) or np.array_equal(intersection, graph.nodes[v]['o'])):
                if np.array_equal(intersection, graph.nodes[u]['o']):
                    intersection_node = u
                elif np.array_equal(intersection, graph.nodes[v]['o']):
                    intersection_node = v
                segment_index = 0
                intersection_obj = Intersection(intersection=intersection,
                                                prev_node=intersection_node,
                                                next_node=intersection_node,
                                                segment_index=segment_index,
                                                is_node=True)
                result.append(intersection_obj)
                found = True
                break

            # Intersection point is on the edge.
            for i in range(len(current_edge) - 1):
                if point_on_segment(intersection, current_edge[i], current_edge[i+1]):
                    prev_node = None
                    next_node = None
                    if filter_nodes_helper(graph.nodes[u]['o'], bboxs) and not filter_nodes_helper(graph.nodes[v]['o'], bboxs):
                        prev_node = u
                        next_node = v
                    else:
                        prev_node = v
                        next_node = u

                    if np.array_equal(intersection, current_edge[i]):
                        segment_index = i
                    elif np.array_equal(intersection, current_edge[i+1]):
                        segment_index = i + 1
                    else:
                        segment_index = i

                    # QUESTION: can you discard an intersection point for which next_node is also an intersection point? will do for now
                    if not is_node_intersection(graph.nodes[next_node]['o'], intersections) and not is_node_intersection(graph.nodes[prev_node]['o'], intersections):
                        intersection_obj = Intersection(intersection=intersection,
                                                        prev_node=prev_node,
                                                        next_node=next_node,
                                                        segment_index=segment_index,
                                                        is_node=False)
                        result.append(intersection_obj)
                    found = True
                    break
            if found:
                break
    
    return np.array(result)

def get_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def traverse_component(intersection_obj, graph, visited_nodes, bboxs, step_size=3):
    """
    New implementation.
    Intersection can be an edge_segment_point or on a specific edge_segment.
    If intersection_point == edge_segment_point for a given index, then index marks the first edge_segment's index where intersection appears as the starting point
    - or in other words: the index of the second appearance of intersection_point in current_edge

    NOTE: Most distances are sqrt(1) or sqrt(2). Does sqrt(3) exist? Might have to check later.
    """

    if intersection_obj.prev_node in visited_nodes and intersection_obj.next_node in visited_nodes:
        return np.array([]), np.array([])

    if intersection_obj.is_node:
        # prev_node = next_node = intersection point's node's id
        neighbours = [n for n in graph[intersection_obj.next_node].keys() if not is_node_inside_bboxs(graph.nodes[n]['o'], bboxs)]
        neighbours = [(intersection_obj.next_node, n) for n in neighbours if n not in visited_nodes]
        future_nodes = neighbours
        if len(future_nodes) == 0:
            visited_nodes.append(intersection_obj.next_node)
            return np.array([intersection_obj.next_node]), np.array([])
        prev_node, next_node = future_nodes[0]
        future_nodes = future_nodes[1:]
    else:
        prev_node = intersection_obj.prev_node
        next_node = intersection_obj.next_node
        future_nodes = []

    current_position = intersection_obj.intersection
    initialized = False

    current_segment_start = -1
    current_segment_end = -1
    current_segment_index = intersection_obj.segment_index
    remaining_step_size = step_size
    
    visited_nodes.append(prev_node)

    traversed_nodes = []
    traversed_paths = []
    accumulator = []
    accumulated = False
    
    while next_node is not None:
        traversed_nodes.append(next_node)
        current_edge = graph[prev_node][next_node]['pts']

        if not np.array_equal(current_edge[0], graph.nodes[prev_node]['o']):
            current_edge = current_edge[::-1] # if previous node is not the first element in the array - in this case it's the last - then the array needs to be reversed
            if not initialized: # this is only needed when component traversal starts - otherwise edges would be skipped if the condition above is true
                current_segment_index = len(current_edge) - current_segment_index

        while current_segment_index < len(current_edge) - 1:
            current_segment_start = current_edge[current_segment_index]
            accumulator.append(current_segment_start)
            current_segment_end = current_edge[current_segment_index + 1]

            segment_length = get_distance(current_segment_start, current_segment_end)
            if not np.array_equal(current_position, current_segment_start) and not initialized: # only true for the very first loop, when current_position is initialized as intersection_point
                accumulator[0] = current_position
                segment_length = get_distance(current_position, current_segment_end)

            initialized = True

            if segment_length == 0:
                current_segment_index += 1
                continue

            if remaining_step_size > segment_length:
                current_segment_index += 1
                remaining_step_size -= segment_length
            else:
                current_position = current_segment_end
                accumulator.append(current_segment_end)
                accumulated = True
                current_segment_index += 1
                remaining_step_size = step_size

            if current_segment_index == (len(current_edge) - 1):
                    current_position = current_segment_end
                    if not accumulated:
                        accumulator.append(current_segment_end)
                        accumulated = True
            
            if accumulated:
                if np.array_equal(intersection_obj.intersection, [103, 257, 213]):
                    print(accumulator)
                traversed_paths.extend(accumulator)
                accumulated = False
                accumulator = []

        visited_nodes.append(next_node)
        next_node_neighbours = [n for n in graph[next_node].keys() if n != prev_node and not is_node_inside_bboxs(graph.nodes[n]['o'], bboxs)]
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
        current_segment_index = 0
        current_position = graph.nodes[prev_node]['o'] # probably redundant | UPDATE: not redundant. For example when a path is traversed and the function returns to a 'crossroad' of the component

    return np.array(traversed_nodes), traversed_paths

def traverse_graph(graph, intersection_obj_list, bboxs):
    intersection_obj_list = filter_intersections(intersection_obj_list)
    visited_nodes = []
    traversed_nodes = []
    traversed_paths = []

    for intersection in intersection_obj_list:
        print(f"Traversing from intersection: {intersection.intersection}")
        traversed_nodes_curr, traversed_paths_curr = traverse_component(intersection, graph, visited_nodes, bboxs, step_size=3)
        traversed_nodes.extend(traversed_nodes_curr)
        traversed_paths.extend(traversed_paths_curr)
    
    return np.array(traversed_nodes), np.array(traversed_paths)

def traverse_graph_from_point(graph, intersection_obj_list, bboxs):
    """
    Only testing for now, traverses from one specific node.
    """
    intersection_obj_list = filter_intersections(intersection_obj_list) # no two intersection objects exist that are on the same edge
    curr_intersection = None
    for intersection in intersection_obj_list:
        # Tried and worked: [225, 234, 321], [148, 238, 164](might be interesting to see)
        if np.array_equal(intersection.intersection, np.array([148, 238, 164])):
            curr_intersection = intersection
    
    # print(f"Intersection: {curr_intersection.intersection}\nPrevious node: {curr_intersection.prev_node}\nNext node: {curr_intersection.next_node}\nSection index: {curr_intersection.segment_index}")

    return traverse_component(curr_intersection, graph, [], bboxs, step_size=3)


def add_bbox_to_plot(plotter):
    bounding_box_1 = pv.Box(bounds=(164, 241, 179, 275, 103, 223))
    bounding_box_2 = pv.Box(bounds=(282, 369, 187, 294, 117, 225))

    plotter.add_mesh(bounding_box_1, color="green", opacity=0.4, style='wireframe')
    plotter.add_mesh(bounding_box_2, color="green", opacity=0.4, style='wireframe')

def add_intersections_to_plot(plotter, intersections):
    z, y, x = np.where(intersections > 0)
    intersection_points = np.column_stack((x, y, z))

    plotter.add_points(intersection_points, color="black", point_size=10, render_points_as_spheres=True)

    # labels = [str(tuple(intersection)) for intersection in intersection_points]
    # plotter.add_point_labels(intersection_points, labels, font_size=12, point_size=10)


def filter_intersections(intersection_obj_list):
    result = []
    node_dict = defaultdict(list)

    for intersection in intersection_obj_list:
        node_dict[(intersection.prev_node, intersection.next_node)].append(intersection)

    for key in node_dict.keys():
        intersections = node_dict[key]
        result.append(intersections[0])
    
    return np.array(result)

def show_viewer(graph, intersections, intersection_obj_list, traversed_nodes, traversed_paths):
    traversed_nodes = np.array([graph.nodes[node]['o'] for node in traversed_nodes])
    traversed_nodes = traversed_nodes[:, [2, 1, 0]]
    traversed_paths = traversed_paths[:, [2, 1, 0]]

    intersection_obj_list = filter_intersections(intersection_obj_list)
    fixed_intersections = np.array([
        intersection.intersection
        for intersection in intersection_obj_list
    ])

    fixed_intersections = fixed_intersections[:, [2, 1, 0]]

    closest_nodes = np.array([
        graph.nodes[intersection_obj.next_node]['o']
        for intersection_obj in intersection_obj_list
    ])

    bboxs = [[slice(103, 223, None), slice(179, 275, None), slice(164, 241, None)],
            [slice(117, 225, None), slice(187, 294, None), slice(282, 369, None)]]

    for closest_node in closest_nodes:
        if filter_nodes_helper_2(closest_node, bboxs):
            print(closest_node)

    closest_nodes = closest_nodes[:, [2, 1, 0]]

    plotter = pv.Plotter()
    add_graph_to_plot(plotter=plotter, vessel_graph=graph)
    add_bbox_to_plot(plotter=plotter)
    # add_intersections_to_plot(plotter=plotter, intersections=intersections)
    # plotter.add_points(fixed_intersections, color="blue", point_size=15, render_points_as_spheres=True)
    # plotter.add_points(closest_nodes, color='purple', point_size=10, render_points_as_spheres=True)
    # plotter.add_points(traversed_nodes, color='green', point_size=10, render_points_as_spheres=True)
    plotter.add_points(traversed_paths, color='purple', point_size=10, render_points_as_spheres=True)
    # plotter.add_points(np.array([218, 255, 96]), color='black', point_size=10, render_points_as_spheres=True)
    # plotter.add_points(np.array([217, 255, 97]), color='green', point_size=10, render_points_as_spheres=True)
    # plotter.add_points(np.array([129, 279, 332][::-1]), color='green', point_size=15, render_points_as_spheres=True)

    # labels = [str(tuple(intersection)) for intersection in traversed_nodes]
    # plotter.add_point_labels(traversed_nodes, labels, font_size=12, point_size=10)
    plotter.show_axes()
    plotter.show()

def main():
    skeleton = sitk.ReadImage('dataset/skeleton/005_vein_mask_skeleton.nii.gz')
    skeleton = sitk.GetArrayFromImage(skeleton)
    intersections = sitk.ReadImage('dataset/intersections/005_vessel_intersections_bbox.nii.gz')
    intersections = sitk.GetArrayFromImage(intersections)
    bboxs = [[slice(103, 223, None), slice(179, 275, None), slice(164, 241, None)],
            [slice(117, 225, None), slice(187, 294, None), slice(282, 369, None)]]

    graph = get_graph(skeleton=skeleton)

    intersection_obj_list = create_intersection_objects(intersections=intersections, graph=graph, bboxs=bboxs)

    # # print(graph[522][621]['pts'])

    # # traversed_nodes = traverse_graph_from_point(graph=graph, intersection_obj_list=intersection_obj_list, bboxs=bboxs)
    traversed_nodes, traversed_paths = traverse_graph(graph=graph, intersection_obj_list=intersection_obj_list, bboxs=bboxs)

    u, c = np.unique(traversed_nodes, return_counts=True)
    dup = u[c > 1]
    print(dup)

    show_viewer(graph=graph, intersections=intersections, intersection_obj_list=intersection_obj_list, traversed_nodes=traversed_nodes, traversed_paths=traversed_paths)


if __name__ == "__main__":
    main()