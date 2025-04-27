import pyvista as pv
import SimpleITK as sitk
import numpy as np
import sknw
import math

def get_graph(skeleton):
    return sknw.build_sknw(skeleton)

def get_points(image_path):
    """
    Image shape is: z, y, x\n
    Image has to be binary <=> np.unique(image) = [0 1]
    """
    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)

    z, y, x = np.where(image > 0)
    points = np.column_stack((x, y, z))

    return points

def get_mesh(image_path):

    points = get_points(image_path)

    cloud = pv.PolyData(points)
    surface = cloud.delaunay_3d(alpha=1.0)
    mesh = surface.extract_surface()

    return mesh

def filter_nodes_helper(node, bboxs):
    for bbox in bboxs:
        z_start, z_end = bbox[0].start, bbox[0].stop
        y_start, y_end = bbox[1].start, bbox[1].stop
        x_start, x_end = bbox[2].start, bbox[2].stop

        if (x_start <= node[2] <= x_end) and (y_start <= node[1] <= y_end) and (z_start <= node[0] <= z_end):
            return True

    return False

def filter_nodes(nodes, bboxs):
    result = []

    for node in nodes:
        if not filter_nodes_helper(node, bboxs):
            result.append(node)
    
    return np.array(result)

def filter_nodes_2(graph, bboxs):
    result = []

    for node in graph.nodes:
        node_coord = graph.nodes[node]['o']
        if not filter_nodes_helper(node_coord, bboxs):
            result.append(node)
    
    return np.array(result)

def get_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def point_on_segment(p, a, b, eps=1e-3):
    ab = b - a
    ap = p - a

    cross = np.cross(ab, ap)
    if np.linalg.norm(cross) > eps:
        return False
    
    ab_dot_ab = np.dot(ab, ab)
    ap_dot_ab = np.dot(ap, ab)

    return 0 - eps <= ap_dot_ab <= ab_dot_ab + eps

def get_closest_graph_nodes_2(intersections, graph):
    ret_nodes = []
    z, y, x = np.where(intersections > 0)
    intersections = np.column_stack((z, y, x))

    found = False
    for intersection in intersections:
        for (u, v) in graph.edges:
            current_edge = graph[u][v]['pts']
            if len(current_edge) == 0:
                continue

            for i in range(len(current_edge) - 1):
                if point_on_segment(intersection, current_edge[i], current_edge[i + 1]):
                    print(graph.nodes[u]['o'])
                    print(graph.nodes[v]['o'])
                    ret_nodes.append(u)
                    ret_nodes.append(v)
                    found = True
                    break
            
            if found:
                break
    
    return np.array(ret_nodes)


def get_closest_graph_nodes(intersections, graph, bbox):
    z, y, x = np.where(intersections > 0)
    intersections = np.column_stack((z, y, x))
    nodes = np.array([graph.nodes[node_id]['o'] for node_id in graph.nodes])
    nodes = filter_nodes(nodes, bbox)
    closest_nodes = []
    closest_node = None
    min_distance = None
    for point in intersections:
        for node in nodes:
            if closest_node is None:
               closest_node = node
               min_distance = get_distance(point, closest_node)
            else:
               new_distance = get_distance(point, node)
               if new_distance < min_distance:
                   closest_node = node
                   min_distance = new_distance
        closest_nodes.append(closest_node)
        closest_node = None
        min_distance = None
    
    closest_nodes = np.array(closest_nodes)
    return closest_nodes[:, [2, 1, 0]]

def add_graph_to_plot(plotter: pv.Plotter, vessel_graph):
    nodes = vessel_graph.nodes()
    edges = vessel_graph.edges()

    for (start, end) in edges:
        edge_points = vessel_graph[start][end]['pts']
        edge_points = edge_points[:, [2, 1, 0]]
        line = pv.lines_from_points(edge_points, close=False)
        plotter.add_mesh(line, color="green", opacity=0.2)

    node_points = np.array([nodes[node_id]['o'] for node_id in nodes])
    node_points = node_points[:, [2, 1, 0]]
    plotter.add_points(node_points, color="red", opacity=0.8, point_size=5, render_points_as_spheres=True)

def main():
    skeleton = sitk.ReadImage('dataset/skeleton/005_vein_mask_skeleton.nii.gz')
    skeleton = sitk.GetArrayFromImage(skeleton)
    intersections = sitk.ReadImage('dataset/intersections/005_vessel_intersections_bbox.nii.gz')
    intersections = sitk.GetArrayFromImage(intersections)
    # skeleton = skeleton.transpose(2, 1, 0) # ez miert breakeli a get_graph-ot?
    print(skeleton.shape)
    vessel_graph = get_graph(skeleton)

    print(vessel_graph.nodes[77]['o'])
    print(vessel_graph.nodes[99]['o'])

    plotter = pv.Plotter()

    add_graph_to_plot(plotter=plotter, vessel_graph=vessel_graph)

    intersections = sitk.ReadImage('dataset/intersections/005_vessel_intersections_bbox.nii.gz')
    intersections = sitk.GetArrayFromImage(intersections)
    bbox = [[slice(103, 223, None), slice(179, 275, None), slice(164, 241, None)],
            [slice(117, 225, None), slice(187, 294, None), slice(282, 369, None)]]
    # closest_nodes = get_closest_graph_nodes(intersections=intersections, graph=vessel_graph, bbox=bbox)
    # print(f"Old closest graph nodes shape: {closest_nodes.shape}")

    filtered_node_ids = filter_nodes_2(vessel_graph, bbox)
    print(f"Filtered nodes shape: {filtered_node_ids.shape}")

    closest_nodes_ids = get_closest_graph_nodes_2(intersections, vessel_graph)
    print(f"Closest nodes ids: {closest_nodes_ids}")
    print(f"Intersections shape: {intersections.shape}")
    print(f"Closest nodes shape: {closest_nodes_ids.shape}")

    closest_nodes_filtered_ids = np.array([node_id for node_id in closest_nodes_ids if node_id in filtered_node_ids]) # itt biztos lesz hiba!

    closest_nodes = np.array([vessel_graph.nodes[node_id]['o'] for node_id in closest_nodes_filtered_ids])
    closest_nodes = closest_nodes[:, [2, 1, 0]] # Transposing for the viewer
    print(closest_nodes)
    print(f"Other shape: {closest_nodes.shape}")

    plotter.add_points(closest_nodes, color="purple", opacity=0.8, point_size=10, render_points_as_spheres=True)

    # vessel_mesh = get_mesh('dataset/HiPaS/annotation/vein_nii/005.nii')
    vessel_skeleton_mesh = get_mesh('dataset/skeleton/005_vein_mask_skeleton.nii.gz')
    # # vessel_3d_edge_mesh = get_mesh('dataset/skeleton/005_vessel_edge_3d_sitk.nii.gz')
    # fi_mesh = get_mesh('dataset/intersections/005_vessel_intersections_ne.nii.gz')
    # i2d_points = get_points('dataset/intersections/005_vessel_intersections_2ded.nii.gz')
    # i3d_points = get_points('dataset/intersections/005_vessel_intersections_3ded.nii.gz')
    # i3d_points_sitk = get_points('dataset/intersections/005_vessel_intersections_3ded_sitk.nii.gz')
    i3d_points_bbox = get_points('dataset/intersections/005_vessel_intersections_bbox.nii.gz')

    bounding_box_1 = pv.Box(bounds=(164, 241, 179, 275, 103, 223))
    bounding_box_2 = pv.Box(bounds=(282, 369, 187, 294, 117, 225))

    # plotter.add_mesh(vessel_mesh, color="orange", opacity=0.6, line_width=5)
    # plotter.add_mesh(vessel_skeleton_mesh, color="purple", opacity=0.6, line_width=5)
    plotter.add_points(i3d_points_bbox, color="black", point_size=10, render_points_as_spheres=True)
    plotter.add_mesh(bounding_box_1, color="green", opacity=0.4, style='wireframe')
    plotter.add_mesh(bounding_box_2, color="green", opacity=0.4, style='wireframe')

    print("Plotting...")
    plotter.show_axes()
    plotter.show()



if __name__ == "__main__":
    main()