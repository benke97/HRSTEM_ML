#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import triangle
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.ndimage import median_filter
from PI_U_Net import UNet
from Analyzer import Analyzer
from shapely.ops import cascaded_union, unary_union
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import split
import pandas as pd

def preprocess_image(image_path):
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
    normalized_image = normalized_image[np.newaxis, :, :]
    image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
    return image_tensor

def postprocess_output(output_tensor):
    output_numpy = output_tensor.detach().cpu().numpy()
    output_image = np.squeeze(output_numpy)
    return output_image    


localizer = UNet()
loc_data = torch.load("best_model_data.pth")
loaded_model_state_dict = loc_data['model_state_dict']
localizer.load_state_dict(loaded_model_state_dict)

#loaded_validation_loss = loaded_data['validation_loss']

image_path = "data/experimental_data/32bit/02.tif"
image_tensor = preprocess_image(image_path)
image_tensor = image_tensor.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
localizer = localizer.cuda()
# Pass the image tensor through the model
localizer.eval()
with torch.no_grad():
    predicted_output = localizer(image_tensor)

predicted_localization = postprocess_output(predicted_output)
predicted_localization_save = predicted_localization.copy()
analyzer = Analyzer()
pred_positions = analyzer.return_positions_experimental(image_tensor,predicted_localization)
#set all values larger than 0.1 in predicted_localization to 1
predicted_localization[predicted_localization > 0.02] = 1
input_image_np = image_tensor.squeeze(0).squeeze(0).cpu().numpy()

#%%
poss = pred_positions
#load large_dataset.pkl
data = None
with open('large_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
#load positions from data
point_sets = data['dataframes']
index = 493
poss = point_sets[index][['x','y']].to_numpy()*128
def add_midpoints(pos, tri):
    #calculate the midpoint of all edges
    midpoints = []
    for i in range(len(tri.simplices)):
        #get the three points of the triangle
        p1 = pos[tri.simplices[i,0]]
        p2 = pos[tri.simplices[i,1]]
        p3 = pos[tri.simplices[i,2]]
        #calculate the midpoints of the edges
        m1 = (p1+p2)/2
        m2 = (p2+p3)/2
        m3 = (p3+p1)/2
        #add the midpoints to the list
        midpoints.append(m1)
        midpoints.append(m2)
        midpoints.append(m3)
    #convert the list to a numpy array
    midpoints = np.array(midpoints)
    #remove duplicate points
    midpoints = np.unique(midpoints, axis=0)
    #add the midpoints to the list of points
    pos = np.concatenate((pos,midpoints), axis=0)
    #calculate a new triangulation
    tri = Delaunay(pos)
    return pos, tri
trii = Delaunay(poss)
poss, trii = add_midpoints(poss, trii)
#poss, trii = add_midpoints(poss, trii)
#poss, trii = add_midpoints(poss, trii)
#plot
plt.imshow(input_image_np, cmap='gray')
plt.triplot(poss[:,1], poss[:,0], trii.simplices, color='k')
plt.plot(poss[:,1], poss[:,0], 'o', color='r', markersize=2)
plt.xlim(0,128)
plt.ylim(128,0)
plt.show()

#plot the voronoi diagram of the triangulation, maintain aspect ratio, plot original points on top in another color
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(poss)
voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
pos_sim = data['dataframes'][index][['x','y']].to_numpy()*128
labels = data['dataframes'][index]['label']
plt.plot(pos_sim[:,0], pos_sim[:,1], 'o', color='r', markersize=2)
#plt.plot(pred_positions[:,0], pred_positions[:,1], 'o', color='r', markersize=2)
plt.xlim(0,128)
plt.ylim(128,0)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
#%%
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d

def voronoi_ridge_neighbors(vor):
    ridge_dict = defaultdict(list)
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_dict[tuple(sorted((p1, p2)))].extend([v1, v2])

    region_neighbors = defaultdict(set)
    for pr, vr in ridge_dict.items():
        r1, r2 = [vor.point_region[p] for p in pr]
        if all(v >= 0 for v in vr):  # ridge is finite
            region_neighbors[r1].add(r2)
            region_neighbors[r2].add(r1)
    
    return region_neighbors

# Example usage:
neighbors = voronoi_ridge_neighbors(vor)
sim_vals = []
positions = []
for region, neighbor_regions in neighbors.items():
    #print(f"Region {region} has neighboring regions {neighbor_regions}")
    #get vor point of the region
    #print(len(vor.points))
    input_point_index = np.where(vor.point_region == region)[0]
    if len(input_point_index) == 0:
        continue
    input_point = vor.points[input_point_index[0]]
    positions.append(input_point)

    #get the points of the region
    points = vor.vertices[vor.regions[region]]
    if np.any(points > 128) or np.any(points < 0):
        #print('outside of image')
        sim_vals.append(0)
        continue
    #get the voronoi point of the region
    point_index = np.where(vor.point_region == region)[0]
    input_point = vor.points[point_index[0]]
    poly1 = Polygon(points)
    #get the neighboring points of the region
    all_ious = []  # Initialize all_ious here
    for nbor in neighbor_regions:
        #get the points of the neighboring region
        nbor_points = vor.vertices[vor.regions[nbor]]
        poly2 = Polygon(nbor_points)
        if np.any(nbor_points > 128) or np.any(nbor_points < 0) or not poly1.is_valid or not poly2.is_valid:
            #print('outside of image')
            continue
        #get the voronoi point of the neighboring region
        nbor_point_index = np.where(vor.point_region == nbor)[0]
        if len(nbor_point_index) == 0:
            continue
        nbor_input_point = vor.points[nbor_point_index[0]]
        #calculate the distance vector between the two voronoi points
        distance_vec = nbor_input_point - input_point
        #calculate the new points of the neighboring region
        nbor_points = nbor_points - distance_vec
        #calculate intersection over union
        poly2 = Polygon(nbor_points)
        intersection = poly1.intersection(poly2).area
        union = unary_union([poly1, poly2]).area
        iou = intersection / union if union else 0
        #print(iou)
        all_ious.append(iou)

    # Move the following lines out of the inner loop
    if len(all_ious) > 0:
        mean_IoU = np.mean(all_ious)
        std_dev_IoU = np.std(all_ious)
        cv_IoU = std_dev_IoU / mean_IoU
        sim_vals.append(mean_IoU)
    else:
        sim_vals.append(0)
#print(len(sim_vals), len(positions))
#make a grid interpolating between the positions and their corresponding similarity values
grid_x, grid_y = np.mgrid[0:128, 0:128]
lattice_descriptor_map = griddata(positions, sim_vals, (grid_x, grid_y), method='linear', fill_value=0)
from scipy.spatial import ConvexHull
from skimage.draw import polygon
#calculate the convex hull of pos_sim, make a binary mask and multiply to grid_z
hull = ConvexHull(pos_sim)
hull_points = pos_sim[hull.vertices]
x = hull_points[:, 1]
y = hull_points[:, 0]
mask = np.zeros_like(lattice_descriptor_map, dtype=bool)
rr, cc = polygon(y, x)
mask[rr, cc] = True
lattice_descriptor_map = lattice_descriptor_map*mask
from scipy.ndimage import median_filter
# Define a 3x3 mean filter
size =5
# Apply mean filter to smooth the grid
lattice_descriptor_map = median_filter(lattice_descriptor_map, size)
dx, dy = np.gradient(lattice_descriptor_map)
gradient_magnitude = np.sqrt(dx**2 + dy**2)
#dx2,dy2 = np.gradient(gradient_magnitude)
#gradient_magnitude = np.sqrt(dx2**2 + dy2**2)
positions_original = point_sets[index][['x','y']].to_numpy()*128

#plot the lattice descriptor map
plt.imshow(lattice_descriptor_map.T, cmap='hot',vmin=0,vmax=0.3) # add the origin parameter
plt.scatter(positions_original[:,0],positions_original[:,1], s=3, c='g')
plt.xlim(0,128)
plt.ylim(128,0)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()



def plot_voronoi_region(vor, region_index, ax=None, color=None):
    if ax is None:
        ax = plt.gca()

    region = vor.regions[region_index]
    if not -1 in region:  # Check that the region is finite
        polygon = [vor.vertices[i] for i in region]
        ax.fill(*zip(*polygon), color=color)

        ax.plot(vor.points[:, 0], vor.points[:, 1], 'k.')
        # Find and highlight the point generating the region
        point_index = np.where(np.array(vor.point_region) == region_index)[0][0]
        ax.plot(vor.points[point_index, 0], vor.points[point_index, 1], 'go')
region_index = 33  # Change this to the index of the region you want to plot
neighbor_indices = neighbors[region_index]

fig, ax = plt.subplots()

# Plot the chosen region in red
plot_voronoi_region(vor, region_index, ax, 'red')

# Plot the neighboring regions in blue
for neighbor_index in neighbor_indices:
    plot_voronoi_region(vor, neighbor_index, ax, 'blue')
plt.xlim(0,128)
plt.ylim(0,128)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# %%


def find_neighboring_regions_and_midpoint(ridge_vertices, vor):
    # ridge_points gives the indices of the points between which this ridge is
    ridge_points = [key for key, value in vor.ridge_dict.items() if np.array_equal(np.sort(value), np.sort(ridge_vertices))]
    
    if len(ridge_points) == 0 or -1 in ridge_vertices:
        # skip this ridge if it doesn't match an entry in ridge_dict
        # or if it's connected to the vertex at infinity
        return None, None, None
    
    # point_region gives the region corresponding to a given point
    region1 = vor.point_region[ridge_points[0][0]]
    region2 = vor.point_region[ridge_points[0][1]]
    if -1 in vor.regions[region1] or -1 in vor.regions[region2]:
        return None, None, None
    vertices = [vor.vertices[i] for i in ridge_vertices if i != -1]
    midpoint = np.mean(vertices, axis=0)
    if midpoint[0] < 0 or midpoint[0] > 128 or midpoint[1] < 0 or midpoint[1] > 128:
        return None, None, None

    return region1, region2, midpoint

def is_out_of_bounds(poly):
    """Check if any point in the polygon falls outside the range of 0-128."""
    #print(1)
    for point in poly.exterior.coords:
        #print(point)
        if point[0] < 0 or point[0] > 128 or point[1] < 0 or point[1] > 128:
            print(2)
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
        

def calc_similarity(poly1, poly2):
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

def angle_from_midpoint(midpoint, point):
    return np.arctan2(point[1] - midpoint[1], point[0] - midpoint[0])

def sort_points_by_angle(midpoint, points):
    return sorted(points, key=lambda point: angle_from_midpoint(midpoint, point))

def get_midpoints(points):
    midpoints = []
    for i in range(len(points)):
        if i == len(points)-1:
            midpoints.append((points[i] + points[0])/2)
        else:
            midpoints.append((points[i] + points[i+1])/2)
    return np.array(midpoints)

def closest_point(point,axis):
    print(1)

def generate_new_poly(points,center_edge_index,cell_center):
    if center_edge_index < len(points)-2:
            point_1 = points[center_edge_index]
            point_2 = points[center_edge_index+1]
            point_3 = (points[center_edge_index+1]+points[center_edge_index+2])/2
            point_4 = cell_center
            if center_edge_index == 0:
                point_5 = (points[len(points)-1]+points[0])/2
            else:
                point_5 = (points[center_edge_index-1]+points[center_edge_index])/2
    elif center_edge_index == len(points)-2:
            point_1 = points[center_edge_index]
            point_2 = points[center_edge_index+1]
            point_3 = (points[center_edge_index+1]+points[0])/2
            point_4 = cell_center
            if center_edge_index == 0:
                point_5 = (points[len(points)-1]+points[0])/2
            else:
                point_5 = (points[center_edge_index-1]+points[center_edge_index])/2
    elif center_edge_index == len(points)-1:
            point_1 = points[center_edge_index]
            point_2 = points[0]
            point_3 = (points[0]+points[1])/2
            point_4 = cell_center
            if center_edge_index == 0:
                point_5 = (points[len(points)-1]+points[0])/2
            else:
                point_5 = (points[center_edge_index-1]+points[center_edge_index])/2
    return np.array([point_1,point_2,point_3,point_4,point_5])

def project_and_find_closest(line,midpoint, points):
    # create the line segment vector
    line_vector = np.array(line[1]) - np.array(line[0])
    
    # normalize the line_vector
    line_vector_norm = line_vector / np.linalg.norm(line_vector)

    # initialize an array to hold the distances
    distances = []

    # iterate over each point in points
    for point in points:
        if not np.array_equal(point, midpoint):
            projection = np.dot(point - line[0], line_vector_norm) * line_vector_norm + line[0]
            # compute the distance from the projection to the midpoint
            distance = np.linalg.norm(projection - midpoint)
            distances.append(distance)
        else :
            distances.append(np.inf)


    # find the point with the minimum distance to the midpoint
    min_index = np.argmin(distances)
    #print(distances)
    return points[min_index],min_index

def calc_iou(poly1,poly2):
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    intersection = poly1.intersection(poly2).area
    union = unary_union([poly1, poly2]).area
    iou = intersection / union if union else 0
    return iou

def calc_similarity3(poly1, poly2, vor):
    #get the vertices of of the polygons
    points1 = vor.vertices[vor.regions[poly1]]
    points2 = vor.vertices[vor.regions[poly2]]
    #get the voronoi point of the two polygons
    point_index1 = np.where(vor.point_region == poly1)[0]
    input_point_1 = vor.points[point_index1[0]]
    point_index2 = np.where(vor.point_region == poly2)[0]    
    input_point_2 = vor.points[point_index2[0]]
    #sort the points of the polygon clockwise around the voronoi point (from neg x axis)
    points1 = np.array(sort_points_by_angle(input_point_1, points1))
    points2 = np.array(sort_points_by_angle(input_point_2, points2))

    #the midpoints of the polygon segments
    midpoints1 = get_midpoints(points1)
    midpoints2 = get_midpoints(points2)
    midpoints1_lst = midpoints1.tolist()
    midpoints2_lst = midpoints2.tolist()
    midpoints_ext = midpoints1_lst + midpoints2_lst
    
    #element that appears twice
    midpoint = [x for x in midpoints_ext if midpoints_ext.count(x) == 2][0]
    index1 = midpoints1_lst.index(midpoint)
    index2 = midpoints2_lst.index(midpoint)
    #print(midpoints1)
    #print(midpoint)
    #print(index1)
    #print(len(points1))
    if index1 != len(points1)-1:
        #print('yep')
        opposite_midpoint1,opposite_index1 = project_and_find_closest([points1[index1], points1[index1+1]],midpoint, midpoints1)
        #print(points1[index1],points1[index1+1])
    else:
        #print('nope')
        opposite_midpoint1,opposite_index1 = project_and_find_closest([points1[index1], points1[0]],midpoint, midpoints1)
        #print(points1[index1],points1[0])

    if index2 != len(points2)-1:
        #print('yep')
        opposite_midpoint2,opposite_index2 = project_and_find_closest([points2[index2], points2[index2+1]],midpoint, midpoints2)
        #print(points2[index2],points2[index2+1])
    else:
        #print('nope')
        opposite_midpoint2,opposite_index2 = project_and_find_closest([points2[index2], points2[0]],midpoint, midpoints2)
        #print(points2[index2],points2[0])


    poly1 = generate_new_poly(points1,opposite_index1,input_point_1)
    poly2 = generate_new_poly(points1,index1,input_point_1)
    poly3 = generate_new_poly(points2,opposite_index2,input_point_2)
    poly4 = generate_new_poly(points2,index2,input_point_2)

    displacement = input_point_1-input_point_2
    poly3 = poly3 + displacement
    poly4 = poly4 + displacement
    iou1 = calc_iou(poly2,poly3)
    iou2 = calc_iou(poly1,poly4)
    #print("iou1",iou1)
    #print("iou2",iou2)
    #poly2 = generate_new_poly(points1,index1,input_point_1)
    
    #find the segment midpoint furthest away from ridge midpoint
    #take previous and following segment, new poly = cell_center, previous_segment_midpoint, furthest_away_segment_start_point, furthest_away_segment_end_point following_segment_midpoint

    #plot points1 with a number displaying the index of each point
    #sort the points in points1 by angle
    #plt.scatter(points1[:,0], points1[:,1], c='b', s=2)
    #plt.scatter(midpoints1[:,0], midpoints1[:,1], c='g', s=2)
    #plt.scatter(poly3[:,0], poly3[:,1], c='r', s=2)
    #plt.plot(input_point_1[0], input_point_1[1], 'o', color='r', markersize=1)
    #plt.plot(midpoints1[index1][0], midpoints1[index1][1], 'o', color='c', markersize=2)
    #plt.plot(opposite_midpoint1[0], opposite_midpoint1[1], 'o', color='m', markersize=2)
    #for i in range(len(poly3)):
    #    plt.text(poly3[i,0], poly3[i,1], str(i), fontsize=5)
    #plt.xlim(min(points1[:,0])-10,max(points1[:,0])+10)
    #plt.ylim(min(points1[:,1])-10,max(points1[:,1])+10)
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()
    return max(iou1,iou2)

def calc_similarity_4(poly1, poly2, vor):
    print(1)

def plot_regions(poly1, poly2,midpoint, vor):
    #plot the two polygons
    #print(vor.vertices)
    #print(vor.ridge_points)
    #print(vor.ridge_vertices)
    #print(len(vor.ridge_vertices))
    #plot all ridge points and ridge vertices
    for i in range(len(vor.ridge_points)):
        plt.plot(vor.vertices[vor.ridge_vertices[i],0], vor.vertices[vor.ridge_vertices[i],1], 'o', color='r', markersize=2)
    plt.xlim(0,128)
    plt.ylim(0,128)
    plt.gca().set_aspect('equal', adjustable='box')
    #plot the polygons with fill
    plt.fill(vor.vertices[vor.regions[poly1],0], vor.vertices[vor.regions[poly1],1], color='blue', alpha=0.5)
    plt.fill(vor.vertices[vor.regions[poly2],0], vor.vertices[vor.regions[poly2],1], color='green', alpha=0.5)
    #plot the midpoint
    plt.plot(midpoint[0], midpoint[1], 'o', color='k', markersize=2)
    plt.show()

    #ridge_points = np.array([x for x in vor.vertices[vor.ridge_points]])
    #ridge_vertices = np.array([x for x in vor.vertices[vor.ridge_vertices]])
    #print(ridge_points)


vor = Voronoi(poss)

midpoints = []
similarities = []

for edge in vor.ridge_vertices:
    poly1, poly2, midpoint = find_neighboring_regions_and_midpoint(edge, vor)
    if poly1 is not None and poly2 is not None and midpoint is not None:
        #plot_regions(poly1, poly2,midpoint, vor)
        similarity = calc_similarity2(poly1, poly2,vor)
        midpoints.append(midpoint)
        similarities.append(similarity)
        #calc_similarity3(poly1, poly2, vor)
    else:
        print('infinite cell')
#print(max(similarities))
plt.scatter(np.array(midpoints)[:,0], np.array(midpoints)[:,1], c=similarities)
plt.xlim(0,128)
plt.ylim(0,128)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# %%
# Separate x and y coordinates for griddata
midpoints = np.array(midpoints)
#make similarities binary with a threshold at 0.9
similarities = np.array(similarities.copy())
#similarities[similarities < 0.8] = 0
#similarities[similarities >= 0.8] = 1
#Define grid

grid_x, grid_y = np.mgrid[0:128, 0:128]  # adjust the grid density to your liking

# Perform interpolation
grid_z = griddata(midpoints, similarities, (grid_x, grid_y), method='linear',fill_value=0)
#mean filter grid
from scipy.spatial import ConvexHull
from skimage.draw import polygon
#calculate the convex hull of pos_sim, make a binary mask and multiply to grid_z
hull = ConvexHull(pos_sim)
hull_points = pos_sim[hull.vertices]
x = hull_points[:, 1]
y = hull_points[:, 0]
mask = np.zeros_like(grid_z, dtype=bool)
rr, cc = polygon(y, x)
mask[rr, cc] = True
grid_z = grid_z*mask
from scipy.ndimage import median_filter
# Define a 3x3 mean filter
size =5
# Apply mean filter to smooth the grid
smoothed_grid_z = median_filter(grid_z, size)
dx, dy = np.gradient(smoothed_grid_z)
gradient_magnitude = np.sqrt(dx**2 + dy**2)
smoothed_grid_z_rescale = (smoothed_grid_z.copy() - 0.7)/0.2
# Create the interpolated plot
plt.figure(figsize=(10, 10))
plt.imshow(gradient_magnitude.T, extent=(0, 128, 0, 128), origin='lower', cmap='hot',vmin=0, vmax=1, aspect='equal')
#plt.scatter(pred_positions[:,0], pred_positions[:,1], c='r', s=1)
#plt.scatter(midpoints[:,0], midpoints[:,1], c='b', s=1)
#color is based on labels blue = 1, red = 0
plt.scatter(pos_sim[:,0], pos_sim[:,1], c=labels, s=5)
# Add a color bar
plt.colorbar(label='Similarity')

# Set labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Linearly Interpolated Map of All Midpoints')
plt.xlim(0,128)
plt.ylim(0,128)
# Display the plot
plt.show()

# %%

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

def add_midpoints(pos, tri):
    #calculate the midpoint of all edges
    midpoints = []
    for i in range(len(tri.simplices)):
        #get the three points of the triangle
        p1 = pos[tri.simplices[i,0]]
        p2 = pos[tri.simplices[i,1]]
        p3 = pos[tri.simplices[i,2]]
        #calculate the midpoints of the edges
        m1 = (p1+p2)/2
        m2 = (p2+p3)/2
        m3 = (p3+p1)/2
        #add the midpoints to the list
        midpoints.append(m1)
        midpoints.append(m2)
        midpoints.append(m3)
    #convert the list to a numpy array
    midpoints = np.array(midpoints)
    #remove duplicate points
    midpoints = np.unique(midpoints, axis=0)
    #add the midpoints to the list of points
    pos = np.concatenate((pos,midpoints), axis=0)
    #calculate a new triangulation

    return pos

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
                similarity = calc_similarity(poly1, poly2, vor)
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
    return smoothed_grid_z_rescale.T
# %%d
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from scipy.ndimage import median_filter

def voronoi_ridge_neighbors(vor):
    ridge_dict = defaultdict(list)
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_dict[tuple(sorted((p1, p2)))].extend([v1, v2])

    region_neighbors = defaultdict(set)
    for pr, vr in ridge_dict.items():
        r1, r2 = [vor.point_region[p] for p in pr]
        if all(v >= 0 for v in vr):  # ridge is finite
            region_neighbors[r1].add(r2)
            region_neighbors[r2].add(r1)
    
    return region_neighbors

def calc_lattice_descriptor_map(idx,positions):
    pos = positions[idx]
    tri = Delaunay(pos)
    pos = add_midpoints(pos, tri)
    pos = add_midpoints(pos, tri)
    vor = Voronoi(pos)

    neighbors = voronoi_ridge_neighbors(vor)
    sim_vals = []
    positions = []
    for region, neighbor_regions in neighbors.items():
        input_point_index = np.where(vor.point_region == region)[0]
        if len(input_point_index) == 0:
            continue
        input_point = vor.points[input_point_index[0]]
        positions.append(input_point)

        #get the points of the region
        points = vor.vertices[vor.regions[region]]
        if np.any(points > 128) or np.any(points < 0):
            #print('outside of image')
            sim_vals.append(0)
            continue
        #get the voronoi point of the region
        point_index = np.where(vor.point_region == region)[0]
        input_point = vor.points[point_index[0]]
        poly1 = Polygon(points)
        #get the neighboring points of the region
        all_ious = []  # Initialize all_ious here
        for nbor in neighbor_regions:
            #get the points of the neighboring region
            nbor_points = vor.vertices[vor.regions[nbor]]
            poly2 = Polygon(nbor_points)
            if np.any(nbor_points > 128) or np.any(nbor_points < 0) or not poly1.is_valid or not poly2.is_valid:
                #print('outside of image')
                continue
            #get the voronoi point of the neighboring region
            nbor_point_index = np.where(vor.point_region == nbor)[0]
            if len(nbor_point_index) == 0:
                continue
            nbor_input_point = vor.points[nbor_point_index[0]]
            #calculate the distance vector between the two voronoi points
            distance_vec = nbor_input_point - input_point
            #calculate the new points of the neighboring region
            nbor_points = nbor_points - distance_vec
            #calculate intersection over union
            poly2 = Polygon(nbor_points)
            intersection = poly1.intersection(poly2).area
            union = unary_union([poly1, poly2]).area
            iou = intersection / union if union else 0
            #print(iou)
            all_ious.append(iou)

        # Move the following lines out of the inner loop
        if len(all_ious) > 0:
            mean_IoU = np.mean(all_ious)
            #std_dev_IoU = np.std(all_ious)
            #cv_IoU = std_dev_IoU / mean_IoU
            sim_vals.append(mean_IoU)
        else:
            sim_vals.append(0)
    #print(len(sim_vals), len(positions))
    #make a grid interpolating between the positions and their corresponding similarity values
    grid_x, grid_y = np.mgrid[0:128, 0:128]
    lattice_descriptor_map = griddata(positions, sim_vals, (grid_x, grid_y), method='linear', fill_value=0)
    #calculate the convex hull of pos_sim, make a binary mask and multiply to grid_z
    hull = ConvexHull(pos_sim)
    hull_points = pos_sim[hull.vertices]
    x = hull_points[:, 1]
    y = hull_points[:, 0]
    mask = np.zeros_like(lattice_descriptor_map, dtype=bool)
    rr, cc = polygon(y, x)
    mask[rr, cc] = True
    lattice_descriptor_map = lattice_descriptor_map*mask
    # Define a 3x3 mean filter
    size =5
    # Apply mean filter to smooth the grid
    lattice_descriptor_map = median_filter(lattice_descriptor_map, size)
    dx, dy = np.gradient(lattice_descriptor_map)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    
    return gradient_magnitude.T