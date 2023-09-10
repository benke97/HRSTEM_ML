
def connected_edges(edge, adjacency_matrix):
    node_a, node_b = edge
    connected_nodes_a = set(np.nonzero(adjacency_matrix[node_a])[0])
    connected_nodes_b = set(np.nonzero(adjacency_matrix[node_b])[0])
    
    # Nodes connected to both 'a' and 'b'
    common_nodes = connected_nodes_a.intersection(connected_nodes_b)
    # Edges connected to 'a' and 'b'
    connected_edges_a = [(node_a, node) for node in connected_nodes_a if node != node_b]
    connected_edges_b = [(node_b, node) for node in connected_nodes_b if node != node_a]

    extra_edges = []
    for common_node in common_nodes:
        connected_nodes = set(np.nonzero(adjacency_matrix[common_node])[0])
        extra_edges += [(common_node, node) for node in connected_nodes]
    b = connected_edges_a + connected_edges_b
    all_edges = connected_edges_a + connected_edges_b  #+ extra_edges
    #make a copy of all edges where each pair is stored with the smallest value first, then extract the set by removing from the original
    all_edges_copy = [(min(edge), max(edge)) for edge in all_edges]
    all_edges = list(set(all_edges_copy))
    #print(b)
    #print(all_edges)
    return all_edges

def calc_edge_similarity(edge1, edge2, positions):
    # Calculate the vectors representing the edges
    vector1 = positions[edge1[1]] - positions[edge1[0]]
    vector2 = positions[edge2[1]] - positions[edge2[0]]

    # Calculate the lengths of the vectors
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)

    # Calculate the fractional length difference
    length_diff = np.abs(length1 - length2) / max(length1, length2)

    # Calculate the cosine of the angle between the vectors
    cos_angle = np.dot(vector1, vector2) / (length1 * length2)

    # Calculate the rotation as the difference from perfect alignment (cosine of angle is 1)
    rotation_diff = 1 - cos_angle
    #print(length_diff, rotation_diff)
    # Return a scalar value combining the length and rotation differences.
    # In this example, we'll just average them, but you might choose to weight them differently.
    return (1*length_diff + 1*rotation_diff) / 2

def calc_lattice_descriptor_map(idx, positions):
    pos = positions[idx]
    lattice_descriptor_map = np.zeros((128, 128))
    tri = Delaunay(pos)
    adjacency_matrix = np.zeros((len(pos), len(pos)))
    for triangle in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                adjacency_matrix[triangle[i], triangle[j]] = 1
                adjacency_matrix[triangle[j], triangle[i]] = 1  # because it's an undirected graph

    i_upper = np.triu_indices(adjacency_matrix.shape[0], 1)
    edge_indices = np.where(adjacency_matrix[i_upper] == 1)
    edges = list(zip(i_upper[0][edge_indices], i_upper[1][edge_indices]))
    # Plot triangulation
    #plt.triplot(pos[:,0], pos[:,1], tri.simplices)
    #plt.axis('equal')

    # Plot nodes
    #plt.plot(pos[:,0], pos[:,1], 'o')
    similarity_val = []
    edge_midpoints = []
    for edge in edges:
        connected_edges_list = connected_edges(edge, adjacency_matrix)
        sim_vals = []
        for connected_edge in connected_edges_list:
            similarity = calc_edge_similarity(edge, connected_edge, pos)
            sim_vals.append(similarity)
        similarity_val.append(np.min(sim_vals))
        edge_midpoints.append((pos[edge[0]] + pos[edge[1]]) / 2)
            #print(f"Similarity between {edge} and {connected_edge}: {similarity}")

    #generate lattice_descriptor_map by cubic interoplation between all edge midpoints and their similarity values
    grid_x, grid_y = np.mgrid[0:128, 0:128]
    lattice_descriptor_map = griddata(edge_midpoints, similarity_val, (grid_x, grid_y), method='linear', fill_value=0)
    #
    # sobel filter lattice descriptor map with a 5x5 kernel
    edge = edges[3]

    connected_edges_list = connected_edges(edge, adjacency_matrix)
    for connected_edge in connected_edges_list:
        #print(connected_edges_list)
        node_a, node_b = connected_edge
        #plt.plot([pos[node_a, 0], pos[node_b, 0]], [pos[node_a, 1], pos[node_b, 1]], 'r-')

    #plt.show()
    return lattice_descriptor_map.T

            

    # Plot edges from connected_edges_list of the first edge in another color

plot_nr = 370
lattice_descriptor_map = calc_lattice_descriptor_map(plot_nr, positions)    
# Plot the filtered triangulation
#make this a 1,2 subplot with image and density map
#make this a 1,2 subplot with image and density map
#plot image with positions on top

plt.figure(figsize=(10, 5))
plt.imshow(data['images'][23], cmap='gray') # add the origin parameter
plt.scatter(positions[23][:, 0],positions[23][:, 1], s=1, c='r')
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(data['images'][plot_nr], cmap='gray') # add the origin parameter
plt.title('Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(density_maps[plot_nr], cmap='hot', interpolation='nearest') # add the origin parameter
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(lattice_descriptor_map, cmap='hot', interpolation='nearest') # add the origin parameter
plt.scatter(positions[plot_nr][:, 0],positions[plot_nr][:, 1], s=3, c='g')
plt.axis('off')
plt.show()


############################################################################################################
def find_neighboring_regions_and_midpoint(ridge_vertices, vor):

    ridge_points = [key for key, value in vor.ridge_dict.items() if np.array_equal(np.sort(value), np.sort(ridge_vertices))]
    
    if len(ridge_points) == 0 or -1 in ridge_vertices:
        return None, None, None

    region1 = vor.point_region[ridge_points[0][0]]
    region2 = vor.point_region[ridge_points[0][1]]
    if -1 in vor.regions[region1] or -1 in vor.regions[region2]:
        return None, None, None
    vertices = [vor.vertices[i] for i in ridge_vertices if i != -1]
    midpoint = np.mean(vertices, axis=0)

    return region1, region2, midpoint

def bounding_box_center(points):
    # Create a MultiPoint object from your points
    multi_point = MultiPoint(points)
    
    # Get the bounding box (minx, miny, maxx, maxy)
    bounds = multi_point.bounds
    
    # Calculate the center of the bounding box
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    
    return np.array([center_x, center_y])

def calc_similarity(poly1, poly2, vor):
    points1 = vor.vertices[vor.regions[poly1]]
    points2 = vor.vertices[vor.regions[poly2]]

    if np.any(points1 > 128) or np.any(points1 < 0) or np.any(points2 > 128) or np.any(points2 < 0):
        #print('outside of image')
        return 0

    centroid1 = bounding_box_center(points1)
    centroid2 = bounding_box_center(points2)

    points1 = points1 - centroid1
    points2 = points2 - centroid2

    polygon1 = Polygon(points1)
    polygon2 = Polygon(points2)

    intersection = polygon1.intersection(polygon2).area
    union = unary_union([polygon1, polygon2]).area

    iou = intersection / union if union else 0
    #print('iou: ', iou)
    return iou

#%%
def find_neighboring_regions_and_midpoint(ridge_vertices, vor):

    ridge_points = [key for key, value in vor.ridge_dict.items() if np.array_equal(np.sort(value), np.sort(ridge_vertices))]
    
    if len(ridge_points) == 0 or -1 in ridge_vertices:
        return None, None, None

    region1 = vor.point_region[ridge_points[0][0]]
    region2 = vor.point_region[ridge_points[0][1]]
    if -1 in vor.regions[region1] or -1 in vor.regions[region2]:
        return None, None, None
    vertices = [vor.vertices[i] for i in ridge_vertices if i != -1]
    midpoint = np.mean(vertices, axis=0)

    return region1, region2, midpoint

def is_out_of_bounds(poly):
    """Check if any point in the polygon falls outside the range of 0-128."""
    #print(1)
    for point in poly.exterior.coords:
        #print(point)
        if point[0] < 0 or point[0] > 128 or point[1] < 0 or point[1] > 128:
            #print(2)
            return True
    #print(3)
    return False

def calc_iou_split(poly1, poly2, line):
    poly1_split = split(poly1, line)
    poly2_split = split(poly2, line)
    if len(poly1_split.geoms) == 2 and len(poly2_split.geoms) == 2:

        iou_1 = None
        iou_2 = None
        #calculate intersection between poly1_split.geoms[0] and poly2_split.geoms[0] and poly2_splot.geoms[1]
        intersection_1 = poly1_split.geoms[0].intersection(poly2_split.geoms[0]).area
        intersection_2 = poly1_split.geoms[0].intersection(poly2_split.geoms[1]).area
        if intersection_1 > intersection_2:
            #we know poly1[0] => poly2[0] and poly1[1] => poly2[1]
            if is_out_of_bounds(poly1_split.geoms[0]) or is_out_of_bounds(poly2_split.geoms[0]):
                iou_1 = 0
            else:
                union_1 = unary_union([poly1_split.geoms[0], poly2_split.geoms[0]]).area
                iou_1 = intersection_1 / union_1 if union_1 else 0
            if is_out_of_bounds(poly1_split.geoms[1]) or is_out_of_bounds(poly2_split.geoms[1]):
                iou_2 = 0
            else:
                union_2 = unary_union([poly1_split.geoms[1], poly2_split.geoms[1]]).area
                intersection_2 = poly1_split.geoms[1].intersection(poly2_split.geoms[1]).area
                iou_2 = intersection_2 / union_2 if union_2 else 0
        else:
            #we know poly1[0] => poly2[1] and poly1[1] => poly2[0]
            #if any point of poly1_split.geoms[0] and poly2_split.geoms[1] is outside 0-128, set iou_1 to 0
            if is_out_of_bounds(poly1_split.geoms[0]) or is_out_of_bounds(poly2_split.geoms[1]):
                iou_1 = 0
            else:
                union_1 = unary_union([poly1_split.geoms[0], poly2_split.geoms[1]]).area
                intersection_1 = poly1_split.geoms[0].intersection(poly2_split.geoms[1]).area
                iou_1 = intersection_1 / union_1 if union_1 else 0
            if is_out_of_bounds(poly1_split.geoms[1]) or is_out_of_bounds(poly2_split.geoms[0]):
                iou_2 = 0
            else:
                union_2 = unary_union([poly1_split.geoms[1], poly2_split.geoms[0]]).area
                intersection_2 = poly1_split.geoms[1].intersection(poly2_split.geoms[0]).area
                iou_2 = intersection_2 / union_2 if union_2 else 0
        return max(iou_1, iou_2)
    else:
        return 0
        
def calc_similarity2(poly1, poly2, vor):
    points1 = vor.vertices[vor.regions[poly1]]
    points2 = vor.vertices[vor.regions[poly2]]

    point_index1 = np.where(vor.point_region == poly1)[0]
    input_point_1 = vor.points[point_index1[0]]
    point_index2 = np.where(vor.point_region == poly2)[0]    
    input_point_2 = vor.points[point_index2[0]]
    
    distance_vec = input_point_2 - input_point_1
    points2 = points2 - distance_vec
    
    if np.any(points1 > 128) or np.any(points1 < 0) or np.any(points2 > 128) or np.any(points2 < 0):
        #print('outside of image')
        return 0

    polygon1 = Polygon(points1)
    polygon2 = Polygon(points2)

    intersection = polygon1.intersection(polygon2).area
    union = unary_union([polygon1, polygon2]).area

    iou = intersection / union if union else 0
    #print('iou: ', iou)
    return iou

def calc_similarity(poly1, poly2, vor):
    points1 = vor.vertices[vor.regions[poly1]]
    points2 = vor.vertices[vor.regions[poly2]]

    point_index1 = np.where(vor.point_region == poly1)[0]
    input_point_1 = vor.points[point_index1[0]]
    point_index2 = np.where(vor.point_region == poly2)[0]    
    input_point_2 = vor.points[point_index2[0]]

    distance_vec = input_point_2 - input_point_1
    points2 = points2 - distance_vec

    distance_vec = distance_vec / np.linalg.norm(distance_vec)
    input_point_1 = Point(input_point_1)
    input_point_2 = Point(input_point_2)
    line = LineString([(input_point_1.x - distance_vec[0]*182, input_point_1.y - distance_vec[1]*182), 
                    (input_point_1.x + distance_vec[0]*182, input_point_1.y + distance_vec[1]*182)])
    
    polygon1 = Polygon(points1)
    polygon2 = Polygon(points2)

    line2 = LineString([(input_point_1.x - distance_vec[1]*182, input_point_1.y - distance_vec[0]*182), 
                    (input_point_1.x + distance_vec[1]*182, input_point_1.y + distance_vec[0]*182)])    

    iou_split = calc_iou_split(polygon1, polygon2, line)
    iou_split2 = calc_iou_split(polygon1, polygon2, line2)
    iou_split = max(iou_split, iou_split2)
    #print('iou_split: ', iou_split)
    return iou_split

def calc_lattice_descriptor_map(idx,positions):
    pos = positions[idx]
    tri = Delaunay(pos)
    pos = add_midpoints(pos, tri)
    vor = Voronoi(pos)
    #plot voronoi diagram
    #voronoi_plot_2d(vor)
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()
    midpoints = []
    similarities = []

    for edge in vor.ridge_vertices:
        poly1, poly2, midpoint = find_neighboring_regions_and_midpoint(edge, vor)
        #print(poly1, poly2)
        if poly1 is not None and poly2 is not None and midpoint is not None:
            #check that midpoint is within 0-128
            if np.any(midpoint > 128) or np.any(midpoint < 0):
                pass
            else:
                similarity = calc_similarity2(poly1, poly2, vor)
                midpoints.append(midpoint)
                similarities.append(similarity)
        else:
            #print('infinite cell')
            pass

    midpoints = np.array(midpoints)
    x, y = midpoints[:,0], midpoints[:,1]

    grid_x, grid_y = np.mgrid[0:128, 0:128]
    #print(midpoints)
    #print(len(midpoints),len(similarities))
    grid_z = griddata(midpoints, similarities, (grid_x, grid_y), method='linear',fill_value=0)

    size = 5

    smoothed_grid_z = median_filter(grid_z, size)
    smoothed_grid_z_rescale = (smoothed_grid_z.copy() - 0.8)/0.2
    #smoothed_grid_z_rescale[smoothed_grid_z_rescale > 0.1] = 1
    #smoothed_grid_z_rescale[smoothed_grid_z_rescale <= 0.1] = 0
    return smoothed_grid_z_rescale.T