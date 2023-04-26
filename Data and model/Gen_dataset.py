#%%
from Data_Generator import Data_Generator

number_of_images = 1000
dataset_name = "data_boi"
generator = Data_Generator()
generator.generate_data(number_of_images,dataset_name)
#%%
generator.visualize_data(dataset_name)

# %%
