import numpy as np
from scipy.spatial import distance
import pandas as pd
from shapely.geometry import Polygon, Point
import random
import math
import matplotlib.pyplot as plt

class Particle_Interface_Point_Set_Generator:
    def __init__(self):
        self.particle_lattice_constant = 0.07
        self.support_lattice_constant = 0.1
        self.lattices = {
            "hex" : [0,np.pi/6],
            "square" : [0,np.pi/4],
            "rhombic" : [0,np.arccos(1/np.sqrt(3)), -np.arccos(1/np.sqrt(3)),np.pi/2]
        }
        self.interface_length = 0

    def set_particle_lattice_constant(self,val):
        self.particle_lattice_constant = val

    def set_support_lattice_constant(self,val):
        self.support_lattice_constant = val

    def set_interface_length(self,val):
        self.interface_length = val

    def rotate_vector(self, vec, angle_degrees):
        angle_rad = np.radians(angle_degrees)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        return rotation_matrix @ vec

    def normalize(self,vec):
        return vec / np.linalg.norm(vec)
    
    def skewed_random_float(self,min_value, max_value, alpha, beta): #beta distribution
        random_float = random.betavariate(alpha, beta)
        return min_value + (max_value - min_value) * random_float
    
    def generate_3_point_interface(self):
        A = np.array([0,0])
        B = np.array([1,0])
        k = random.randint(0, 1)
        if k:
            AB = B-A
            normalized_AB = self.normalize(AB)

            # Rotate the vector AB by the specified angle to get the direction of BC
            angle_degrees = self.skewed_random_float(0, 90, 1,3) #beta distr. alpha = 1 beta = 2 => linear, beta= 3 osv. färre stora vinklar
            direction_BC = self.normalize(self.rotate_vector(normalized_AB, angle_degrees))
            length_BC = np.random.rand()
            C = B + direction_BC * length_BC
            return A,B,C
        else:
            AB = A-B
            normalized_AB = self.normalize(AB)
            # Rotate the vector AB by the specified angle to get the direction of BC
            angle_degrees = random.randint(-90,0)
            direction_BC = self.normalize(self.rotate_vector(normalized_AB, angle_degrees))
            length_BC = np.random.rand()
            C = A + direction_BC * length_BC        
            return C,A,B
        
    def signed_angle_between_vectors(self, v1, v2):
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        
        # Keep the angle in the range of -pi to pi
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
            
        return angle
    
    def generate_particle_polygon(self,A,B,C):
        a = self.signed_angle_between_vectors(A-C,np.array([1,0]))
        center = np.array([(A[0]+C[0])/2,(A[1]+C[1])/2])
        radius = math.sqrt((A[0]-C[0])**2+(A[1]-C[1])**2)/2
        num_angles = random.randint(3,5)
        if a < 0:
            random_angles = [random.uniform(0-math.pi-a,-a) for _ in range(num_angles)]
        else:
            random_angles = [random.uniform(2*math.pi -a,math.pi-a) for _ in range(num_angles)]       
        random_angles.sort()

        new_points = []
        scaling = random.uniform(1,math.sqrt(2))    #Scaling if you want :)
        for angle in random_angles:
            new_points.append([center[0]+scaling*radius*math.cos(angle),center[1]+scaling*radius*math.sin(angle)])
        
        new_points = [np.array(point) for point in new_points]
        particle = np.vstack([A, B, C, *new_points])
        return particle
    
    def generate_support_polygon(self,A,B,C):
        dist = C[0]-A[0]
        height = random.uniform(1,1.7)
        D = np.array((A[0]-(3-dist)/2,A[1])) #vänster
        #print(A[0], dist, "D:",D)
        E = D-[0,height] #nere vänster
        G = np.array((C[0]+(3-dist)/2,C[1])) #höger
        F = G - [0,G[1]-E[1]]#nere höger
        return np.vstack([A, B, C, G, F, E, D])    
 
    
    def point_inside_polygon(self, point, polygon):
        x, y = point  # Extract x and y coordinates of the point
        n = len(polygon)  # Number of vertices in the polygon
        inside = False  # Initialize the inside flag to False

        p1x, p1y = polygon[0]  # Get the first vertex of the polygon
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]  # Get the next vertex in the sequence
            
            # Check if the point lies within the y bounds of the current edge (p1, p2)
            if y > min(p1y, p2y) and y <= max(p1y, p2y):
                # Check if the point is to the left of the current edge (p1, p2)
                if x <= max(p1x, p2x):
                    # Check if the current edge (p1, p2) is not horizontal
                    if p1y != p2y:
                        # Calculate the x-coordinate of the intersection between the ray and the current edge (p1, p2)
                        x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    
                    # Check if the ray intersects the current edge (p1, p2)
                    if p1x == p2x or x <= x_intersection:
                        # Toggle the inside flag if the ray intersects the current edge (p1, p2)
                        inside = not inside
            
            # Move on to the next edge by updating p1 to be the current p2
            p1x, p1y = p2x, p2y

        # Return True if the point is inside the polygon, and False otherwise
        return inside

    def plot_polygons_and_points(self, particle, support, filtered_points_particle):
        fig, ax = plt.subplots()

        # Plot particle polygon
        particle_path = plt.Polygon(particle, edgecolor='blue', fill=None, lw=1, linestyle="--", label='Particle')
        ax.add_patch(particle_path)

        # Plot support polygon
        support_path = plt.Polygon(support, edgecolor='red', fill=None, lw=1, linestyle="--", label='Support')
        ax.add_patch(support_path)

        # Plot filtered_points_particle
        filtered_points_particle = np.array(filtered_points_particle)
        ax.scatter(filtered_points_particle[:, 0], filtered_points_particle[:, 1], c='black', label='Points')

        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.axis('off')

        plt.show()

    def apply_random_displacement(self, point_set, max_displacement):
        displaced_points = []
        for i, point in enumerate(point_set):
            angle = random.uniform(0, 2 * math.pi)
            magnitude = random.uniform(0, max_displacement)
            dx = magnitude * math.cos(angle)
            dy = magnitude * math.sin(angle)
            displaced_point = [point[0] + dx, point[1] + dy]
            
            # Clip the coordinates within the unit square
            clipped_point = [min(max(displaced_point[0], 0), 1), min(max(displaced_point[1], 0), 1)]
            displaced_points.append(clipped_point)

        return displaced_points
    

    def generate_lattice_points(self,lattice_constant,lattice_type,rotation_angle):
        if lattice_type == "hex":
            a = lattice_constant*0.8
            hx = a
            hy = a * np.sqrt(3) / 2
        elif lattice_type == "square":
            a = lattice_constant*0.8
            hx = a
            hy = a
        elif lattice_type == "rhombic":
            a = lattice_constant*2
            hx = a / 2
            hy = a / np.sqrt(8)
        # Calculate the number of points in the x and y directions
        Nx = int(3 / hx) + 1
        Ny = int(3 / hy) + 1

        lattice_points = []

        for i in range(Ny):
            for j in range(Nx):
                x = hx * j
                y = hy * i
                if lattice_type == "hex" or lattice_type == "rhombic":
                    if i % 2 == 1:
                        x += hx / 2

                # Center point set on (0,0)
                x -= 1.5
                y -= 1.5
                
                lattice_points.append((x, y))

        # Rotate the entire set of points using matrix multiplication
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle),  np.cos(rotation_angle)]])
        lattice_points_rotated = np.dot(rotation_matrix, np.array(lattice_points).T).T

        # Filter points that lie within the unit square
        lattice_points_filtered = [(x, y) for x, y in lattice_points_rotated if 0 <= x <= 1 and 0 <= y <= 1]

        return lattice_points_filtered        

    def normalize_and_split(self, poly1, poly2):
        # Concatenate the points of the two polygons
        combined_points = np.concatenate((poly1, poly2), axis=0)

        # Find the minimum and maximum values for x and y coordinates
        min_x, min_y = np.min(combined_points, axis=0)
        max_x, max_y = np.max(combined_points, axis=0)

        # Normalize the points by scaling the x and y components to fit inside the unit square
        scale_x, scale_y = 1 / (max_x - min_x), 1 / (max_y - min_y)
        scale = min(scale_x, scale_y)
        normalized_points = (combined_points - np.array([min_x, min_y])) * scale

        # Split the points back into separate polygons
        num_points_poly1 = len(poly1)
        new_poly1, new_poly2 = np.split(normalized_points, [num_points_poly1])

        return new_poly1, new_poly2
    
    def remove_close_points(self,set1, set2, limit):
            """
            Given two point sets, check if any points within the sets are closer than the limit, then randomly remove either of the two points.

            Args:
            set1 (list): List of tuples representing points in set 1
            set2 (list): List of tuples representing points in set 2
            limit (float): Minimum allowed distance between points

            Returns:
            tuple: Two lists of filtered points from set1 and set2
            """
            # Initialize new sets
            new_set1 = set1.copy()
            new_set2 = set2.copy()

            # Create a list to store pairs of points that are too close
            close_pairs = []

            # Iterate through the points in both sets
            for point1 in set1:
                for point2 in set2:
                    # Calculate the Euclidean distance between the points
                    dist = distance.euclidean(point1, point2)

                    # If the distance is less than the limit
                    if dist < limit:
                        # Add the pair of points to the list of close pairs
                        pair = (point1, point2)
                        reversed_pair = (point2, point1)

                        if pair not in close_pairs and reversed_pair not in close_pairs:
                            close_pairs.append(pair)

            # Remove a random point from each pair in the list of close pairs
            for point1, point2 in close_pairs:
                point_to_remove = random.choice([point1, point2])

                if point_to_remove in new_set1:
                    new_set1.remove(point_to_remove)
                elif point_to_remove in new_set2:
                    new_set2.remove(point_to_remove)

            return new_set1, new_set2

    def closest_distance_to_polygon_edge(self, polygon, point_set):
        polygon = Polygon(polygon)
        distances = []
        
        # Convert the point set into Shapely Points
        shapely_points = [Point(p) for p in point_set]
        
        # For each point, find the closest distance to the polygon's edge
        for point in shapely_points:
            # Find the distance from the point to the polygon's exterior
            distance = polygon.exterior.distance(point)
            distances.append(distance)
        
        return distances
    
    def calculate_distances(self,point_set, given_point):
        distances = []
        x1, y1 = given_point

        for point in point_set:
            x2, y2 = point
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(distance)

        return distances
    
    def move_point_set_down(self,point_set1, point_set2, a):
        moved_point_set1 = np.array(point_set1, dtype=float)
        while True:
            min_distance = float('inf')
            for point1 in moved_point_set1:
                for point2 in point_set2:
                    dist = distance.euclidean(point1, point2)
                    if dist < min_distance:
                        min_distance = dist

            if min_distance < a:
                break
            else:
                moved_point_set1[:, 1] -= 0.001  # Move all points in set1 down by 0.001 in y-axis

        return moved_point_set1.tolist()
    
    def rotate_and_displace_point_set(self, point_set,benchmark=False):
        # Set the center of the unit square
        center_x, center_y = 0.5, 0.5

        # Rotate the point set by a random angle around the center
        theta = np.random.uniform(0, 2 * np.pi)
        if benchmark:
            theta = 0
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Generate random dx and dy values
        dx, dy = np.random.uniform(0, 0.2, size=2)
        if benchmark:
            dx = 0
            dy = 0

        # Apply the rotation and displacement to the point set
        updated_points = point_set.apply(
            lambda row: self.rotate_and_translate(row, center_x, center_y, rotation_matrix, dx, dy),
            axis=1,
            result_type='expand'
        )
        updated_points.columns = ['x', 'y']
        point_set[['x', 'y']] = updated_points

        # Remove points outside the unit square and create a new DataFrame
        new_point_set = pd.DataFrame(columns=point_set.columns)
        for index, row in point_set.iterrows():
            if 0 <= row['x'] <= 1 and 0 <= row['y'] <= 1:
                # Use pd.concat with a DataFrame containing a single row
                row_as_df = pd.DataFrame([row], columns=point_set.columns)
                new_point_set = pd.concat([new_point_set, row_as_df], ignore_index=True)

        return new_point_set

    def rotate_and_translate(self, row, center_x, center_y, rotation_matrix, dx, dy):
        point = np.array([row['x'] - center_x, row['y'] - center_y])
        rotated_point = rotation_matrix @ point
        displaced_point = rotated_point + np.array([center_x + dx, center_y + dy])
        return pd.Series(displaced_point, index=['x', 'y'])
    
    def generate_random_point_set(self,benchmark=False):
        
        while True:
            filtered_points_particle = []
            filtered_points_support = []
            A,B,C = self.generate_3_point_interface()
            particle = self.generate_particle_polygon(A,B,C)
            support = self.generate_support_polygon(A,B,C)
            self.set_interface_length(distance.euclidean(support[4],support[5]))
            #print(self.interface_length)
            particle,support = self.normalize_and_split(particle,support)
            A_norm, B_norm, C_norm = [particle[0],particle[1],particle[2]]
            interface_center = np.mean(np.array([A_norm,B_norm,C_norm]),axis=0)

            particle_lattice_type = random.choice(list(self.lattices.keys()))
            if benchmark:
                particle_lattice_type = "rhombic"
            particle_lattice_rotation = random.choice(self.lattices[particle_lattice_type])
            particle_lattice = self.generate_lattice_points(self.particle_lattice_constant,particle_lattice_type,particle_lattice_rotation)

            support_lattice_type = random.choice(list(self.lattices.keys()))
            if benchmark:
                support_lattice_type = "rhombic"
            support_lattice_rotation = random.choice(self.lattices[support_lattice_type])
            support_lattice = self.generate_lattice_points(self.support_lattice_constant,support_lattice_type,support_lattice_rotation)
            #particle_lattice = self.apply_random_displacement(particle_lattice,self.particle_lattice_constant/20)
            #support_lattice = self.apply_random_displacement(support_lattice,self.support_lattice_constant/20)


            for point in particle_lattice:
                if self.point_inside_polygon(point, particle):
                    filtered_points_particle.append(point)
            for point in support_lattice:
                if self.point_inside_polygon(point, support):
                    filtered_points_support.append(point)
            #manipulate point set
            filtered_points_particle,filtered_points_support = self.remove_close_points(filtered_points_particle,filtered_points_support,0.035)
            
            filtered_points_particle = self.apply_random_displacement(filtered_points_particle,self.particle_lattice_constant/15)
            filtered_points_support = self.apply_random_displacement(filtered_points_support,self.support_lattice_constant/15)
            if len(filtered_points_particle) > 0 and len(filtered_points_support) > 0:
                filtered_points_particle = self.move_point_set_down(filtered_points_particle,filtered_points_support,0.09)
                filtered_points_particle,filtered_points_support = self.remove_close_points(filtered_points_particle,filtered_points_support,0.05)
                
                if len(filtered_points_particle) > 0 and len(filtered_points_support) > 0:
                    point_set=np.vstack([filtered_points_particle,filtered_points_support])

                    #print(particle_lattice_type,support_lattice_type)
                    
                    #Label points, 1 = particle, 0 = support
                    ones_list = [1] * len(filtered_points_particle)
                    zeros_list = [0] * len(filtered_points_support)
                    labels = ones_list + zeros_list
                    

                    particle_lattice_constant_list = [self.particle_lattice_constant] * len(filtered_points_particle)
                    support_lattice_constant_list = [self.support_lattice_constant] * len(filtered_points_support)
                    lattice_constants = particle_lattice_constant_list + support_lattice_constant_list


                    particle_lattice_type_list = [particle_lattice_type] * len(filtered_points_particle)
                    support_lattice_type_list = [support_lattice_type] * len(filtered_points_support)
                    lattice_types = particle_lattice_type_list + support_lattice_type_list


                    particle_lattice_rotation_list = [particle_lattice_rotation] * len(filtered_points_particle)
                    support_lattice_rotation_list = [support_lattice_rotation] * len(filtered_points_support)
                    lattice_rotations = particle_lattice_rotation_list + support_lattice_rotation_list

                    particle_edge_distances = self.closest_distance_to_polygon_edge(particle,filtered_points_particle)
                    support_edge_distances = self.closest_distance_to_polygon_edge(support,filtered_points_support)
                    edge_distances = particle_edge_distances + support_edge_distances
                    
                    particle_interface_center_distances = self.calculate_distances(filtered_points_particle,interface_center)
                    support_interface_center_distances = self.calculate_distances(filtered_points_support,interface_center)
                    interface_center_distances = particle_interface_center_distances + support_interface_center_distances
                    self.plot_polygons_and_points(particle,support,point_set)
                    
                    
                    point_set_dict = {
                        'x': point_set[:,0],
                        'y': point_set[:,1],
                        'distance_to_edge':edge_distances,
                        'distance_to_interface_center':interface_center_distances,
                        'label':labels,
                        'lattice_constant':lattice_constants,
                        'lattice_type': lattice_types,
                        'lattice_rotation':lattice_rotations
                    }

                    point_set_df = pd.DataFrame(point_set_dict)
                    point_set_without_augment = self.rotate_and_displace_point_set(point_set_df,True)
                    point_set = self.rotate_and_displace_point_set(point_set_df,False)
                    label_1_points = point_set[point_set["label"] == 1][['x', 'y']].values
                    label_0_points = point_set[point_set["label"] == 0][['x', 'y']].values
                    if  label_1_points.shape[0] > 3 and label_0_points.shape[0] > 3 and len(label_1_points) < len(label_0_points):
                        break
            else:
                print("filtered_points_particle is empty. Restarting the process...")
        
    
        return point_set, point_set_without_augment