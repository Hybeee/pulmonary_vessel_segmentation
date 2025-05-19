from collections import defaultdict
import numpy as np
import sknw

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

def filter_nodes(graph, bboxs):
    result = []

    for node in graph.nodes:
        node_coord = graph.nodes[node]['o']
        if not filter_nodes_helper(node_coord, bboxs):
            result.append(node)
    
    return np.array(result)

def point_on_segment(p, a, b, eps=0.1):
    ab = b - a
    ap = p - a
    cross = np.cross(ab, ap)
    if np.linalg.norm(cross) > eps:
        return False
    
    ab_dot_ab = np.dot(ab, ab)
    ap_dot_ab = np.dot(ap, ab)

    return 0 - eps <= ap_dot_ab <= ab_dot_ab + eps

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
            if not initialized and intersection_obj.segment_index != 0: # this is only needed when component traversal starts - otherwise edges would be skipped if the condition above is true
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

def filter_intersections(intersection_obj_list):
    result = []
    node_dict = defaultdict(list)

    for intersection in intersection_obj_list:
        node_dict[(intersection.prev_node, intersection.next_node)].append(intersection)

    for key in node_dict.keys():
        intersections = node_dict[key]
        result.append(intersections[0])
    
    return np.array(result)

def traverse_graph(graph, intersection_obj_list, bboxs):
    intersection_obj_list = filter_intersections(intersection_obj_list)
    visited_nodes = []
    traversed_nodes = []
    traversed_paths = []

    for intersection in intersection_obj_list:
        print(f"Traversing from intersection: {intersection.intersection}")
        traversed_nodes_curr, traversed_paths_curr = traverse_component(intersection, graph, visited_nodes, bboxs, step_size=3)
        traversed_nodes.extend(traversed_nodes_curr)
        traversed_paths.append(traversed_paths_curr)
    
    return np.array(traversed_nodes), traversed_paths