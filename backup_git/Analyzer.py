import numpy as np
from skimage.segmentation import clear_border
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure
import math
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial.distance import cdist


class Analyzer:
    def __init__(self):
        self.image_list = []
        self.gt_list = []
        self.pred_list = []
        self.point_sets = []
        self.atomaps_positions = []
        self.SSIM_list = []

    def set_SSIM_list(self,SSIM_list):
        self.SSIM_list = SSIM_list

    def set_atomaps_positions(self,atomaps_positions):
        self.atomaps_positions = atomaps_positions

    def set_image_list(self,image_list):
        self.image_list = image_list
    
    def set_gt_list(self,gt_list):
        self.gt_list = gt_list
    
    def set_pred_list(self,pred_list):
        self.pred_list = pred_list

    def set_point_sets(self,point_sets):
        self.point_sets = point_sets

    def preprocess_gt(self,gt):
        gt_mask = np.where((gt > 0.01),1,0)
        gt_mask = clear_border(gt_mask)
        gt = gt*gt_mask
        return gt,gt_mask
    
    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def weighted_centroid_position(self,gt,gt_mask):
        labeled_image = measure.label(gt_mask)
        regions = measure.regionprops(labeled_image, intensity_image=gt)
        weighted_centroids = [region.weighted_centroid for region in regions]
        return weighted_centroids

    def calculate_circ_intersections(self,centers,radii):
        centers = np.asarray(centers)
        radii = np.asarray(radii)

        intersections = []

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.linalg.norm(centers[j] - centers[i])

                # If the distance between the centers is greater than the sum of the radii, the circles don't intersect
                if d > radii[i] + radii[j]:
                    continue
                # If one circle is contained within the other, they don't intersect
                elif d < abs(radii[i] - radii[j]):
                    continue
                # If the circles are identical, they don't intersect
                elif d == 0 and radii[i] == radii[j]:
                    continue
                # If the circles are tangent (touch at exactly one point)
                elif d == radii[i] + radii[j]:
                    intersections.append(centers[i] + radii[i] * (centers[j] - centers[i]) / d)
                else:
                    a = (radii[i]**2 - radii[j]**2 + d**2) / (2 * d)
                    h = np.sqrt(radii[i]**2 - a**2)
                    x3 = centers[i] + a * (centers[j] - centers[i]) / d
                    rx = -h * (centers[j] - centers[i]) / d

                    intersections.append(x3 + np.array([-rx[1], rx[0]]))
                    intersections.append(x3 + np.array([rx[1], -rx[0]]))

        # Remove duplicates
        intersections = np.unique(intersections, axis=0)

        return intersections
    
    def point_within_pixel(self, point, pixel):
        px_x, px_y = pixel
        pt_x, pt_y = point

        if pt_x >= px_x and pt_x <= px_x + 1 and pt_y >= px_y and pt_y <= px_y + 1:
            return True
        else:
            return False  
          
    def unravel_predicted_positions(self,gt,gt_mask,gt_idx,gt_positions):
        labeled_image = measure.label(gt_mask)
        regions = measure.regionprops(labeled_image, intensity_image=gt)
        error_list =[]
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            #print(minr,minc,maxr,maxc)
            radii = []
            centers = []
            for row in range(minr, maxr):
                for col in range(minc, maxc):
                    center = (row + 0.5, col + 0.5)
                    intensity = gt[row,col]
                    if intensity == 0:
                        continue
                    radius = -np.log(intensity)
                    centers.append(center)
                    radii.append(radius)
            radii = np.array(radii)
            centers = np.array(centers)
            intersections = self.calculate_circ_intersections(centers,radii)
            #if intersections.size != 0:
            #    intersections -= [0.5, 0.5]
            #intersections -= [0.5,0.5]
            intersections = np.array([point for point in intersections if minr <= point[0] < maxr and minc <= point[1] < maxc])
            if intersections.size != 0:  # Add this line
                intersections -= [minr, minc]
            else:
                continue
            
            #weighted_centroid = np.array(region.weighted_centroid)
            #weighted_centroid -= [0.5,0.5]
            #gt_positions -= [0.5,0.5]
            gt_pos = next((pos for pos in gt_positions if minr-1 <= pos[1] < maxr+1 and minc-1 <= pos[0] < maxc+1), None)
            if gt_pos is None:
                continue
            #print("gt_pos:",gt_pos,"centroid:", weighted_centroid)
            bbox_image = gt[minr-1:maxr+1, minc-1:maxc+1]
            dbscan = DBSCAN(eps=0.1, min_samples=3)
            dbscan.fit(intersections) 
            labels = dbscan.labels_
            clusters = [intersections[labels == i] for i in range(max(labels) + 1)]
            clusters = sorted(clusters, key=len, reverse=True)

            flat_index = np.argmax(bbox_image)

            row, col = np.unravel_index(flat_index, bbox_image.shape)
                
            avg_points = [np.mean(cluster, axis=0)+[1,1] for cluster in clusters]
            
            predicted_position = [0,0]
            if avg_points == []:
                predicted_position = [row+0.5,col+0.5]
            else:
                good_points=[]
                for idx,point in enumerate(avg_points):
                    if self.point_within_pixel(point,[row,col]):
                        good_points.append(idx)
                
                if good_points == []:
                    min_dist = float('inf')
                    min_dist_idx = 0
                    for idx,point in enumerate(avg_points):
                        dist = self.euclidean_distance(point,[row+0.5,col+0.5])
                        if dist < min_dist:
                            min_dist = dist
                            min_dist_idx = idx
                    predicted_position = avg_points[min_dist_idx]
                else:
                    #avg_points is already sorted [0] -> largest cluster
                    predicted_position = avg_points[good_points[0]]

            
            gt_pos =  np.array(gt_pos)-[minc,minr]+[0.5,0.5]  
            predicted_position = np.array(predicted_position)-[0.5,0.5]
            #print("Gt_pos:",gt_pos," pred_pos: ", predicted_position)
            error = self.euclidean_distance(gt_pos,np.array([predicted_position[1],predicted_position[0]]))
            #print(error)
            error_list.append(error)
        
            if gt_idx in {1, 2} and regions.index(region) in {1, 2}:
                #print(gt_pos,predicted_position,error)
                plt.figure()
                plt.imshow(gt,cmap="gray")
                plt.scatter(gt_positions[:,0]-0.5,gt_positions[:,1]-0.5,c='g',s=3)
                #plt.scatter(weighted_centroid[1]-0.5,weighted_centroid[0]-0.5,c='r',s=3)
                fig, ax = plt.subplots()
                bbox_image = gt[minr-1:maxr+1, minc-1:maxc+1]
                ax.imshow(bbox_image, cmap='gray')
                plt.show() 


                plt.figure()
                plt.imshow(bbox_image, cmap='gray') 
                plt.scatter(intersections[:, 1]+0.5, intersections[:, 0]+0.5, c=labels, cmap='viridis',s=2)

                #weighted_centroid -= [minr, minc]
                if intersections.size > 0:
                    ax.scatter( intersections[:, 1]+0.5,intersections[:, 0]+0.5, color='blue', s=2)  # Scatter plot of intersections
                    #ax.scatter(weighted_centroid[1]+0.5,weighted_centroid[0]+0.5,c='r',s=2)
                    ax.scatter(predicted_position[1],predicted_position[0],c='m',s=10)
                    if gt_pos is not None:  # If a gt_pos was found inside the bounding box
                        #gt_pos -= [minc, minr]
                        ax.scatter(gt_pos[0],gt_pos[1],c='g',s=30)
                #print(gt_pos,predicted_position)
                plt.show()
        return error_list

                    
    def return_positions_experimental(self,image, prediction):
        tot_positions = 0
        gt,gt_mask = self.preprocess_gt(prediction)
        labeled_image = measure.label(gt_mask)
        regions = measure.regionprops(labeled_image, intensity_image=gt)
        positions = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            #print(minr,minc,maxr,maxc)
            radii = []
            centers = []
            for row in range(minr, maxr):
                for col in range(minc, maxc):
                    center = (row + 0.5, col + 0.5)
                    intensity = gt[row,col]
                    if intensity == 0:
                        continue
                    radius = -np.log(intensity)
                    centers.append(center)
                    radii.append(radius)
            radii = np.array(radii)
            centers = np.array(centers)
            intersections = self.calculate_circ_intersections(centers,radii)

            intersections = np.array([point for point in intersections if minr <= point[0] < maxr and minc <= point[1] < maxc])
            if intersections.size != 0:  # Add this line
                intersections -= [minr, minc]
            else:
                continue

            bbox_image = gt[minr-1:maxr+1, minc-1:maxc+1]
            dbscan = DBSCAN(eps=0.1, min_samples=3)
            dbscan.fit(intersections) 
            labels = dbscan.labels_
            clusters = [intersections[labels == i] for i in range(max(labels) + 1)]
            clusters = sorted(clusters, key=len, reverse=True)

            flat_index = np.argmax(bbox_image)

            row, col = np.unravel_index(flat_index, bbox_image.shape)
                
            avg_points = [np.mean(cluster, axis=0)+[1,1] for cluster in clusters]
            
            predicted_position = [0,0]
            if avg_points == []:
                predicted_position = [row+0.5,col+0.5]
            else:
                good_points=[]
                for idx,point in enumerate(avg_points):
                    if self.point_within_pixel(point,[row,col]):
                        good_points.append(idx)
                
                if good_points == []:
                    min_dist = float('inf')
                    min_dist_idx = 0
                    for idx,point in enumerate(avg_points):
                        dist = self.euclidean_distance(point,[row+0.5,col+0.5])
                        if dist < min_dist:
                            min_dist = dist
                            min_dist_idx = idx
                    if min_dist > 0.5:
                        predicted_position = [row+0.5,col+0.5]
                    else:
                        predicted_position = avg_points[min_dist_idx]
                else:
                    #avg_points is already sorted [0] -> largest cluster
                    predicted_position = avg_points[good_points[0]]

            predicted_position = np.array(predicted_position)-[0.5,0.5] + [minr, minc]
            positions.append(predicted_position)
        
        positions = np.array(positions)
        #plt.figure()
        #plt.imshow(image,origin="lower",cmap='gray')
        #plt.scatter(positions[:, 1]-0.5,positions[:, 0]-0.5, c='r', s=5)
        #plt.axis('off')

        return positions-[0.5,0.5]


    def calculate_average_error(self):
        tot_positions = 0
        gt_example = np.zeros((np.shape(self.pred_list[0])))
        weighted_centroid_mean_distances = []
        error_list = []
        for idx, gt in enumerate(self.pred_list):
            print(idx)
            point_set = self.point_sets[idx]
            true_positions = point_set[['x', 'y']].to_numpy()*128
            
            true_gt,true_gt_mask = self.preprocess_gt(self.gt_list[idx])
            true_centroid_positions = self.weighted_centroid_position(true_gt,true_gt_mask)
            tot_positions += true_positions.shape[0]
            gt,gt_mask = self.preprocess_gt(gt)

            weighted_centroid_positions = self.weighted_centroid_position(gt,gt_mask)

            for weighted_centroid_position in weighted_centroid_positions:
                min_distance = float('inf')
                for position in true_centroid_positions:
                    distance = self.euclidean_distance(weighted_centroid_position, position)
                    if distance < min_distance and distance <= 2:
                        min_distance = distance
                if min_distance != float('inf'):
                    weighted_centroid_mean_distances.append(min_distance)

            errors = self.unravel_predicted_positions(gt,gt_mask,idx,true_positions)
            error_list.append(errors)

            image_example = self.image_list[idx]
            gt_example = gt
        
        flat_error_list = [item for sublist in error_list for item in sublist]
        mean = np.mean(flat_error_list)
        std = np.std(flat_error_list)

        print("Mean error:", mean)
        print("Standard Deviation:", std)
            #error.append()
        #y, x = zip(*weighted_centroid_positions) 
        #print(gt_example.size)
        #plt.figure()
        #plt.subplot(1,2,1)
        #plt.imshow(image_example,cmap="gray")
        #plt.subplot(1,2,2)
        #plt.imshow(gt_example,cmap="gray")
        #plt.scatter(true_positions[:,0]-0.5,true_positions[:,1]-0.5,c='r',s=2)
        #plt.scatter(x,y,c='b',s=2)
        #print(tot_positions)
        return mean, std
    



#DUMP

#            if idx in {2,4,6,8,10} and regions.index(region) in {7, 9}:
                #plt.figure()
                #plt.imshow(gt,cmap="gray")
                #plt.scatter(gt_positions[:,0]-0.5,gt_positions[:,1]-0.5,c='g',s=3)
                #plt.scatter(weighted_centroid[1]-0.5,weighted_centroid[0]-0.5,c='r',s=3)
#                fig, ax = plt.subplots()
#                bbox_image = gt[minr-1:maxr+1, minc-1:maxc+1]
#                ax.imshow(bbox_image, cmap='gray') 
 #               intersections -= [minr, minc]

  #              dbscan = DBSCAN(eps=0.085, min_samples=5)
   #             dbscan.fit(intersections) 
    #            labels = dbscan.labels_
     #           clusters = [intersections[labels == i] for i in range(max(labels) + 1)]
      #          clusters = sorted(clusters, key=len, reverse=True)

       #         flat_index = np.argmax(bbox_image)
                #print(clusters[0])
                # Convert the 1D index to 2D indices
        #        row, col = np.unravel_index(flat_index, bbox_image.shape)
                
         #       avg_points = [np.mean(cluster, axis=0)+[1,1] for cluster in clusters]
                
          #      predicted_position = [0,0]
           #     if avg_points == []:
           #         predicted_position = [row+0.5,col+0.5]
         #       else:
          #          good_points=[]
         #           for idx,point in enumerate(avg_points):
         #               if self.point_within_pixel(point,[row,col]):
         #                   good_points.append(idx)
                    
          #          predicted_position = [0,0]
          #          if good_points == []:
           #             min_dist = float('inf')
         #               min_dist_idx = 0
          #              for idx,point in enumerate(avg_points):
          #                  dist = self.euclidean_distance(point,[2.5,2.5])
          #                  if dist < min_dist:
          #                      min_dist = dist
          #                      min_dist_idx = idx
          #              predicted_position = avg_points[good_points[min_dist_idx]]
          #          else:
                        #avg_points is already sorted [0] -> largest cluster
          #              predicted_position = avg_points[good_points[0]]

          #      print(gt_pos-[minc, minr]+[minc, minr],predicted_position-[1,1]+[minc, minr])



                
          #      print(row,col,avg_points)
                
          #      gt_row = minr-1+row
          #      gt_col= minc-1+col
#

           #     plt.figure()
          #      plt.imshow(bbox_image, cmap='gray') 
           #     plt.scatter(intersections[:, 1]+0.5, intersections[:, 0]+0.5, c=labels, cmap='viridis',s=2)

           #     weighted_centroid -= [minr, minc]
           #     if intersections.size > 0:
             #       ax.scatter( intersections[:, 1]+0.5,intersections[:, 0]+0.5, color='blue', s=2)  # Scatter plot of intersections
            #        ax.scatter(weighted_centroid[1]+0.5,weighted_centroid[0]+0.5,c='r',s=2)
             #       ax.scatter(predicted_position[1]-0.5,predicted_position[0]-0.5,c='m',s=10)
        #            if gt_pos is not None:  # If a gt_pos was found inside the bounding box
            #            gt_pos -= [minc, minr]
             #           ax.scatter(gt_pos[0]+0.5,gt_pos[1]+0.5,c='g',s=30)
        #        #    else:
                #        print("WTF")
            #print(f'Number of intersections: {np.shape(np.array(intersections))}')

#def compare_atomaps(self,images,)


    def assign_nearest(self, list1, list2, max_distance=2):
        list1, list2 = np.array(list1), np.array(list2)
        
        if list2.size == 0:
            return [(None, None)] * len(list1)
        
        tree = cKDTree(list2)
        matches = tree.query_ball_point(list1, max_distance)

        assignment = []
        
        for i, match in enumerate(matches):
            if len(match) == 1:
                assignment.append(tuple(list2[match[0]]))
            else:
                min_distance, _ = tree.query(list1[i])  # Get the smallest distance for this point
                print(f"Point {list1[i]} has {len(match)} matches within {max_distance}. Smallest distance: {min_distance}")
                assignment.append((None, None))
        
        return assignment

    def calculate_error(self, list1, list2):
        distances = []
        for p1, p2 in zip(list1, list2):
            if p2 == (None, None):
                distances.append(np.inf)
            else:
                distances.append(np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2)))
        return distances
    
    def comparison_atomaps(self):
        unet_positions = []
        for image, prediction in zip(self.image_list,self.pred_list):
            unet_pos = self.return_positions_experimental(image,prediction)
            unet_positions.append(unet_pos)
        print(type(self.point_sets))
        true_positions = self.point_sets[0][['x', 'y']].to_numpy()*128
        atomaps_positions = self.atomaps_positions
        
        errors_unet = []
        errors_atomaps = []
        for positions_unet,positions_atomaps in zip(unet_positions,atomaps_positions):
            positions_unet = np.array(positions_unet)[:, ::-1]
            nn_unet = self.assign_nearest(positions_unet,true_positions,3)
            nn_atomaps = self.assign_nearest(positions_atomaps,true_positions,3)
            unet_errors = self.calculate_error(positions_unet,nn_unet)
            atomaps_error = self.calculate_error(positions_atomaps,nn_atomaps)
            errors_unet.append(unet_errors)
            errors_atomaps.append(atomaps_error)
        
        result = {"point_set":self.point_sets,
                  "true_positions":true_positions,
                  "unet_predictions":unet_positions,
                  "unet_errors":errors_unet,
                  "atomaps_predictions":atomaps_positions,
                  "atomaps_errors":errors_atomaps,
                  "SSIM":self.SSIM_list}
        #add "contains inf" to dict
        return result
        
    def plot_how_it_works(self, prediction, region_idx):
        gt,gt_mask = self.preprocess_gt(prediction)
        labeled_image = measure.label(gt_mask)
        regions = measure.regionprops(labeled_image, intensity_image=gt)
        positions = []
        minr, minc, maxr, maxc = regions[region_idx].bbox
        radii = []
        centers = []
        for row in range(minr, maxr):
            for col in range(minc, maxc):
                center = (row + 0.5, col + 0.5)
                intensity = gt[row,col]
                if intensity == 0:
                    continue
                radius = -np.log(intensity)
                centers.append(center)
                radii.append(radius)
        radii = np.array(radii)
        centers = np.array(centers)
        intersections = self.calculate_circ_intersections(centers,radii)

        intersections = np.array([point for point in intersections if minr <= point[0] < maxr and minc <= point[1] < maxc])
        if intersections.size != 0:
            intersections -= [minr, minc]

        bbox_image = gt[minr-1:maxr+1, minc-1:maxc+1]
        fig, ax = plt.subplots()
        ax.imshow(gt, cmap='gray',origin='upper')

        # Create a Rectangle patch
        rect = patches.Rectangle((minc-0.5, minr-0.5), maxc - minc, maxr - minr, linewidth=0.5, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)
        ax.axis('off')
        # Show the figure
        plt.show()
        

        point_set = self.point_sets[0]
        true_positions = point_set[['x', 'y']].to_numpy()*128
        gt_pos = next((pos for pos in true_positions if minr-1 <= pos[1] < maxr+1 and minc-1 <= pos[0] < maxc+1), None)

        fig, ax = plt.subplots()
        ax.imshow(bbox_image, cmap='gray',origin='lower')
        # Add circles
        for center, radius in zip(centers, radii):
            circle = plt.Circle((center[1] - minc + 0.5, center[0] - minr + 0.5), radius, fill=False, edgecolor='darkorange', linewidth=2)
            ax.add_artist(circle)

        if len(intersections) > 0:
            plt.scatter(intersections[:, 1]+0.5, intersections[:, 0]+0.5, marker='x', color='springgreen')
            plt.plot(gt_pos[0]+0.5,gt_pos[1]+0.5,marker='x',color='r',markersize=10)
        plt.axis('equal')
        plt.axis('off')
        plt.show()

        dbscan = DBSCAN(eps=0.1, min_samples=3)
        dbscan.fit(intersections) 
        labels = dbscan.labels_
        clusters = [intersections[labels == i] for i in range(max(labels) + 1)]
        clusters = sorted(clusters, key=len, reverse=True)


        return intersections,centers,radii,bbox_image,minr,minc,maxr,maxc,gt_pos

    def compare_unet_atomaps(self,unet_positions,atomaps_positions,true_positions):
        unet_errors = []
        atomaps_errors = []
        for positions_unet,positions_atomaps in zip(unet_positions,atomaps_positions):
            nn_unet = self.assign_nearest(positions_unet,true_positions,3)
            nn_atomaps = self.assign_nearest(positions_atomaps,true_positions,3)
            unet_error = self.calculate_error(positions_unet,nn_unet)
            atomaps_error = self.calculate_error(positions_atomaps,nn_atomaps)
            unet_errors.append(unet_error)
            atomaps_errors.append(atomaps_error)
            print(unet_error,atomaps_error)
        return unet_errors,atomaps_errors