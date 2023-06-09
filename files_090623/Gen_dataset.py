#%%
from Data_Generator import Data_Generator

number_of_images = 5000

dataset_name = "dataset_workstation_4"
generator = Data_Generator()
generator.generate_data(number_of_images,dataset_name)
#%%
generator.visualize_data(dataset_name)
generator.visualize_random_image(dataset_name)
# %%
from Data_Generator import Data_Generator
image_name = "1"
generator = Data_Generator()
generator.generate_benchmark_images(image_name)
#generator.generate_benchmark_set(image_name)
generator.visualize_benchmark_image(image_name)
# %%