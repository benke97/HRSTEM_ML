import pickle
from Point_Set_Generator import Particle_Interface_Point_Set_Generator
from Image_Maker import Image_Maker
import time
from IPython.display import display, clear_output
import numpy as np
import pandas
import matplotlib.pyplot as plt
from time import sleep

class Data_Generator:
    def __init__(self):
        self.noise_ranges = [(1,5), (5, 10),(10,1000)]
        self.number_of_images = 0

    def display_progress(self,loop_idx,elapsed_time):
        def progress_bar(percentage, width=30):
            filled = int(width * percentage)
            head = '>' if filled < width else ''
            empty = width - filled - len(head)
            bar = f"[{'=' * filled}{head}{' ' * empty}]"
            return bar
        percentage_complete = (loop_idx + 1) / self.number_of_images
        bar = progress_bar(percentage_complete)

        average_iteration_time = elapsed_time / (loop_idx + 1)
        eta = average_iteration_time * (self.number_of_images - (loop_idx + 1))
        eta_minutes = eta / 60

        clear_output(wait=True)
        print(f'Generating {self.number_of_images} images ...')
        print(f'Image {loop_idx+1}/{self.number_of_images}')
        print(bar)
        print(f'Estimated time remaining: {eta_minutes:.2f} minutes')
    
    def set_noise_ranges(self,noise_ranges):
        self.noise_ranges = noise_ranges

    def random_value_from_ranges(self):
        # Choose a random index from the list of ranges
        selected_index = np.random.randint(0, len(self.noise_ranges))

        # Use the index to select the desired range
        selected_range = self.noise_ranges[selected_index]

        # Generate a random number from the selected range
        random_value = np.random.uniform(selected_range[0], selected_range[1])

        return random_value
    
    def generate_data(self,number_of_images,save_name):
        self.number_of_images = number_of_images
        point_set_generator = Particle_Interface_Point_Set_Generator()
        imager = Image_Maker()
        
        total_time = 0
        point_sets = []
        images = []
        gaussian_ground_truths = []
        exp_ground_truths = []
        segmented_images = []
        
        for i in range(self.number_of_images):
            start_time = time.time()
            
            point_set_generator.set_particle_lattice_constant(np.random.uniform(0.06, 0.14))
            point_set_generator.set_support_lattice_constant(np.random.uniform(0.08, 0.15))
            point_set = point_set_generator.generate_random_point_set()
            point_sets.append(point_set)

            imager.set_scale(np.random.uniform(5,15))
            imager.set_poisson(self.random_value_from_ranges())
            imager.set_perlin_scale(np.random.randint(5,30))
            image, gaussian_ground_truth, exp_ground_truth, segmented_image = imager.generate_image(point_sets[i])
            images.append(image)
            #print(np.max(image))
            gaussian_ground_truths.append(gaussian_ground_truth)
            exp_ground_truths.append(exp_ground_truth)
            segmented_images.append(segmented_image)

            end_time = time.time()
            time_taken = end_time - start_time
            total_time += time_taken
            self.display_progress(i,total_time)
        clear_output(wait=True)
        sleep(0.001)
        print(f'Data generation complete, {self.number_of_images} images succesfully generated. Saving as {save_name}.pkl ...')
        data = {
            'dataframes': point_sets,
            'images': images,
            'exponential_ground_truths':exp_ground_truths,
            'gaussian_ground_truths': gaussian_ground_truths,
            'segmented_images': segmented_images
        }
        with open(f'{save_name}.pkl', 'wb') as f:
            pickle.dump(data, f)
        clear_output(wait=True)
        sleep(0.001)
        print(f'Data saved succesfully.')

    def visualize_data(self,file_name):
        with open(f'{file_name}.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        
        loaded_dataframe_list = loaded_data['dataframes']
        loaded_dataframe_list[0].info()

        loaded_image_list = loaded_data['images']
        exp_ground_truths = loaded_data['exponential_ground_truths']
        gaussian_ground_truths = loaded_data['gaussian_ground_truths']
        def find_lattice_type_indexes():
            lattice_types = ['hex', 'square', 'rhombic']
            indexes = {lattice_type: [] for lattice_type in lattice_types}

            for idx, df in enumerate(loaded_dataframe_list):
                lattice_type = df[df['label'] == 1]['lattice_type'].iloc[0]
                if lattice_type in lattice_types and len(indexes[lattice_type]) < 3:
                    indexes[lattice_type].append(idx)
                    if all(len(indexes[l]) == 3 for l in lattice_types):
                        break
            return indexes

        lattice_type_indexes = find_lattice_type_indexes()

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, lattice_type in enumerate(lattice_type_indexes.keys()):
            np.random.shuffle(lattice_type_indexes[lattice_type])
            for j, idx in enumerate(lattice_type_indexes[lattice_type]):
                image = loaded_image_list[idx]
                axes[i, j].imshow(image, cmap='gray',origin='lower')
                if j == 0:
                    axes[i, j].set_title(f'Particle Lattice Type: {lattice_type}')
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()

    #write a function to visualize a random image, its expoenential ground truth, and its segmented image
    def visualize_random_image(self,file_name):
        with open(f'{file_name}.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        loaded_dataframe_list = loaded_data['dataframes']
        loaded_image_list = loaded_data['images']
        exp_ground_truths = loaded_data['exponential_ground_truths']
        gaussian_ground_truths = loaded_data['gaussian_ground_truths']
        segmented_images = loaded_data['segmented_images']
        random_index = np.random.randint(0,len(loaded_dataframe_list))
        positions = loaded_dataframe_list[random_index][['x','y']].to_numpy()*128
        image = loaded_image_list[random_index]
        exp_ground_truth = exp_ground_truths[random_index]
        gaussian_ground_truth = gaussian_ground_truths[random_index]
        segmented_image = segmented_images[random_index]
        plt.subplot(1,3,1)
        plt.imshow(image, cmap="gray", origin='lower')
        plt.title('Image') 
        plt.subplot(1,3,2)
        plt.imshow(exp_ground_truth, cmap="gray", origin='lower')
        plt.title('Exponential Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(segmented_image, cmap="gray", origin='lower')
        plt.scatter(positions[:,0],positions[:,1],s=1,c='r')
        plt.title('Segmented Image')
        plt.show()

        plt.figure()
        plt.imshow(segmented_image, cmap="gray", origin='lower')
        plt.scatter(positions[:,0],positions[:,1],s=1,c='r')
        plt.title('Segmented Image')
        plt.show()


    def generate_benchmark_images(self,save_name):
        print("hello")
        
        point_set_generator = Particle_Interface_Point_Set_Generator()
        imager = Image_Maker()
        
        point_set_generator.set_particle_lattice_constant(np.random.uniform(0.06, 0.11))
        point_set_generator.set_support_lattice_constant(np.random.uniform(0.08, 0.13))
        point_set = point_set_generator.generate_random_point_set(benchmark=True)

        imager.set_scale(np.random.uniform(5,15))
        imager.set_poisson(10000)
        imager.set_perlin_scale(np.random.randint(5,30))
        images,distorted_images, gaussian_ground_truth, exp_ground_truth, ssim_list, ssim_distorted = imager.generate_benchmark_images(point_set)
        
        
        data = {
            'dataframe': point_set,
            'image': images,
            'distorted_images': distorted_images,
            'SSIM': ssim_list,
            'SSIM_distorted': ssim_distorted,
            'exponential_ground_truth':exp_ground_truth,
            'gaussian_ground_truth': gaussian_ground_truth
        }
        with open(f'{save_name}.pkl', 'wb') as f:
            pickle.dump(data, f)
    
    def visualize_benchmark_image(self,file_name):
        with open(f'{file_name}.pkl', 'rb') as f:
            loaded_data = pickle.load(f)

        images = loaded_data['image']
        distorted_images =loaded_data['distorted_images']
        ssim_values = loaded_data['SSIM']
        ssim_distorted = loaded_data['SSIM_distorted']

        if len(images) != len(ssim_values):
            raise ValueError("i dont feel so good")
        
        for image,distorted_image, ssim,ssim_d in zip(images,distorted_images, ssim_values,ssim_distorted):
                plt.subplot(1,2,1)
                plt.imshow(image, cmap="gray", origin='lower')
                plt.title(f'SSIM: {ssim}') 
                plt.subplot(1,2,2)
                plt.imshow(distorted_image, cmap="gray", origin='lower')
                plt.title(f'SSIM: {ssim_d}')
                plt.show() 

    
    def generate_benchmark_set(self,image_name):
        with open(f'{image_name}.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
        imager = Image_Maker()