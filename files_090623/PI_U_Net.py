#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

import random
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import cv2

random.seed(1337)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Downsampling
        self.conv1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bridge
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 512)

        # Upsampling
        self.upconv6 = upconv_block(512, 256)
        self.conv6 = conv_block(512, 256)
        self.upconv7 = upconv_block(256, 128)
        self.conv7 = conv_block(256, 128)
        self.upconv8 = upconv_block(128, 64)
        self.conv8 = conv_block(128, 64)

        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        x7 = self.conv4(x6)
        x8 = self.conv5(x7)

        x9 = self.upconv6(x8)
        x10 = torch.cat([x9, x5], dim=1)
        x11 = self.conv6(x10)

        x12 = self.upconv7(x11)
        x13 = torch.cat([x12, x3], dim=1)
        x14 = self.conv7(x13)

        x15 = self.upconv8(x14)
        x16 = torch.cat([x15, x1], dim=1)
        x17 = self.conv8(x16)

        out = self.output(x17)

        return out

with open('dataset_workstation.pkl', 'rb') as f:
    data = pickle.load(f)

class CustomDataset(Dataset):
    def __init__(self,images, ground_truths):
        self.images = images
        self.ground_truths = ground_truths

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        raw_image = self.images[idx]
        ground_truth = self.ground_truths[idx]

        normalized_image = np.maximum((raw_image - raw_image.min()) / (raw_image.max() - raw_image.min()), 0)
        normalized_image = normalized_image[np.newaxis, :, :]
        ground_truth = ground_truth[np.newaxis, :, :]
        image = torch.tensor(normalized_image, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.float32)

        return image, ground_truth
    
def count_black_pixels(image):
    # Assuming that black pixels have a value of 0
    black_pixels = np.count_nonzero(image == 0)
    return black_pixels

def plot_losses(train_losses, validation_losses):
    # Start from the 5th epoch (Python uses 0-based indexing)
    train_losses = train_losses[4:]
    validation_losses = validation_losses[4:]

    epochs = range(5, len(train_losses) + 5)  # Adjust the range of your epochs accordingly

    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid()
    plt.show()

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
 #%%   
def main():
    def scale_and_normalize_image(image):
        scaling_factor = np.random.uniform(14000, 30000)
        scaled_image = image * scaling_factor
        normalized_image = scaled_image / 30000
        return normalized_image
    
    train_images, validation_images, train_ground_truths, validation_ground_truths = train_test_split(
        data['images'], data['exponential_ground_truths'], test_size=0.25, random_state=1337)

    train_dataset = CustomDataset(train_images, train_ground_truths)
    validation_dataset = CustomDataset(validation_images, validation_ground_truths)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
    image, ground_truth = validation_dataset[0]
    #print(count_black_pixels(ground_truth))
    
    def custom_loss(y_pred,y_true):
        # Standard MSE loss
        mse = torch.mean((y_true - y_pred) ** 2)

        # Create a mask that is 1 where y_true is 0, and 0 otherwise
        mask = (y_true == 0).float()

        # Apply the mask to y_pred
        false_positive_preds = y_pred * mask

        # Calculate the penalty for false positives
        false_positive_penalty = torch.mean(false_positive_preds ** 2)*5

        return mse + false_positive_penalty
    
    def modified_chi_squared_loss(pred, target):
        ground_truth = target.unsqueeze(1)
        difference = pred - ground_truth
        max_ground_truth = torch.max(ground_truth)
        return torch.sum((difference ** 2) / (ground_truth + max_ground_truth / 10))
    
    def weighted_mse_loss(pred, target, weights=None):
        diff = pred - target
        squared_diff = diff ** 2

        if weights is not None:
            squared_diff = squared_diff * weights

        loss = torch.mean(squared_diff)
        return loss
    
    def loss_weight_schedule(y_true, epoch, start_weight=65, end_weight=1, decay_stop_epoch=5, threshold=0.1):
        weight_map = (y_true > threshold).float() * start_weight + 1
        decay_rate = (end_weight / start_weight) ** (1 / (decay_stop_epoch - 1))
        
        if epoch < decay_stop_epoch:
            decayed_weight = start_weight * (decay_rate ** epoch)
        else:
            decayed_weight = end_weight

        weight_map = weight_map * decayed_weight / start_weight
        return weight_map

    def dynamic_weighted_mse_loss(y_pred, y_true, epoch, num_epochs):
        weight_map = loss_weight_schedule(y_true,epoch, start_weight=250,threshold=0.1)
        weighted_mse = (y_pred - y_true)**2 * weight_map
        loss = torch.mean(weighted_mse)
        if epoch > 3:
            mask = (y_true == 0).float()

            # Apply the mask to y_pred
            false_positive_preds = y_pred * mask
            torch.where(false_positive_preds < 0.01, 0, false_positive_preds)
            # Calculate the penalty for false positives
            false_positive_penalty = torch.mean(false_positive_preds ** 2)*5
        else:
            false_positive_penalty = 0

        return loss + false_positive_penalty
    
    def visualize_output(image, ground_truth, predicted_output, exp_input, exp_input_pred):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        # Plot input image
        axs[0, 0].imshow(image.squeeze(0), cmap='gray')
        axs[0, 0].set_title("Input Image")
        axs[0, 0].axis("off")

        # Plot ground truth
        axs[0, 1].imshow(ground_truth.squeeze(0), cmap='gray')
        axs[0, 1].set_title("Ground Truth")
        axs[0, 1].axis("off")

        # Plot predicted output
        axs[0, 2].imshow(predicted_output.squeeze(0), cmap='gray')
        axs[0, 2].set_title("Predicted Output")
        axs[0, 2].axis("off")

        # Plot exp_input
        axs[1, 0].imshow(exp_input, cmap='gray')
        axs[1, 0].set_title("Exp Input")
        axs[1, 0].axis("off")

        # Plot exp_input_pred
        axs[1, 1].imshow(exp_input_pred, cmap='gray')
        axs[1, 1].set_title("Exp Input Pred")
        axs[1, 1].axis("off")

        # Leave last slot blank
        axs[1, 2].axis("off")

        plt.show()

    def warmup_lr_scheduler(epoch):
        if epoch < 5:
            return 0.001
        else:
            return 0.005


    model = UNet()
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_epochs = 100
    learning_rate = 0.0001 # Set the desired learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)
    
    train_losses = []
    validation_losses = []
    min_validation_loss = float("inf")
    best_epoch = 0
    patience = 10
    early_stopping_counter = 0
    #scheduler = StepLR(optimizer,1,0)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training", unit="batch")


        for images, targets in train_progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            # Forward pass
            predictions = model(images)
            # Calculate the loss
            #weights = (targets > 0.1).float() * 56 + 1  # Create the weight tensor
            #print(weights.max())
            #loss = weighted_mse_loss(predictions, targets, weights=weights)
            loss = dynamic_weighted_mse_loss(predictions, targets, epoch,num_epochs)
            #loss = custom_loss(predictions,targets)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        validation_loss = 0
        
        validation_progress_bar = tqdm(validation_dataloader, desc=f"Epoch {epoch+1} - Validation", unit="batch")
        
        with torch.no_grad():
            for images, targets in validation_progress_bar:
                images = images.to(device)
                targets = targets.to(device)

                predictions = model(images)
                #weights = (targets > 0.1).float() * 56 + 1
                #loss = weighted_mse_loss(predictions,targets, weights=weights)
                loss = dynamic_weighted_mse_loss(predictions, targets, epoch,num_epochs)
                #loss = custom_loss(predictions,targets)
                validation_loss += loss.item()

        validation_loss /= len(validation_dataloader)
        validation_losses.append(validation_loss)       
        
        if validation_loss < min_validation_loss:
            early_stopping_counter = 0
            min_validation_loss = validation_loss
            best_epoch = epoch+1
            best_model_data = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'validation_loss': min_validation_loss
            }
            torch.save(best_model_data, "best_model_data.pth")   
        else:
            early_stopping_counter += 1
            if early_stopping_counter > patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        model_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'validation_loss': min_validation_loss
        }
        torch.save(model_data, f"model_data_epoch_{epoch + 1}.pth")
        
        model.eval()
        with torch.no_grad():
            sample_idx = 2 # Choose a random sample
            image, ground_truth = validation_dataset[sample_idx]
            image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            prediction = model(image_tensor)
            predicted_output = prediction.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
            
            #experimental
            image_path = "data/experimental_data/32bit/03.tif"
            exp_tensor = preprocess_image(image_path)
            exp_tensor = exp_tensor.unsqueeze(0)
            exp_tensor = exp_tensor.to(device)
            exp_pred = model(exp_tensor)
            pred_exp = postprocess_output(exp_pred)
            exp_im = exp_tensor.squeeze(0).squeeze(0).cpu().numpy()
            
            visualize_output(image, ground_truth, predicted_output,exp_im,pred_exp)
        
        #scheduler.step()
        #current_lr = optimizer.param_groups[0]['lr']
        #current_lr = optimizer.param_groups[0]['lr']
        
        #decay_rate = (1 / 60) ** (1 / (5 - 1))
        #if epoch < 5:
        #    decayed_weight = 60 * (decay_rate ** epoch)
        #else:
        #    decayed_weight = 1
        
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Validation Loss: {validation_loss}")
    
    return model,train_losses, validation_losses
#%%
if __name__ == '__main__':
    torch.cuda.empty_cache()
    trained_model, train_losses, validation_losses = main()
    
    loaded_data = torch.load("best_model_data.pth")

# Extract the epoch number, model state dictionary, and validation loss
    loaded_epoch = loaded_data['epoch']
    loaded_model_state_dict = loaded_data['model_state_dict']
    loaded_validation_loss = loaded_data['validation_loss']
    
    plot_losses(train_losses, validation_losses)

    trained_model.load_state_dict(loaded_model_state_dict)

    image_path = "data/experimental_data/32bit/10.tif"
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    torch.cuda.reset_max_memory_allocated(device)
    torch.cuda.reset_max_memory_cached(device)
    # Pass the image tensor through the model
    trained_model.eval()
    with torch.no_grad():
        predicted_output = trained_model(image_tensor)

    predicted_segmentation = postprocess_output(predicted_output)
    input_image_np = image_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # Plot the input image and predicted segmentation
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image_np, cmap='gray')
    plt.title(f'Epoch{loaded_epoch}')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_segmentation, cmap='gray')
    plt.title('Predicted Segmentation')

    plt.show()
    #%%
    loaded_data = torch.load("model_data_epoch_24.pth")
    model = UNet()
    # Extract the epoch number, model state dictionary, and validation loss
    loaded_epoch = loaded_data['epoch']
    loaded_model_state_dict = loaded_data['model_state_dict']
    #loaded_validation_loss = loaded_data['validation_loss']
    model.load_state_dict(loaded_model_state_dict)
    image_path = "data/experimental_data/32bit/09.tif"
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    model = model.cuda()
    # Pass the image tensor through the model
    model.eval()
    with torch.no_grad():
        predicted_output = model(image_tensor)

    predicted_segmentation = postprocess_output(predicted_output)
    input_image_np = image_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # Plot the input image and predicted segmentation
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image_np, cmap='gray')
    plt.title(f'Epoch{loaded_epoch}')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_segmentation, cmap='gray')
    plt.title('Predicted Segmentation')

    plt.show()
    # %%
    #get predictions for each image and save them along with the input data
    #load the data
    with open('dataset_workstation_3.pkl', 'rb') as f:
        data = pickle.load(f)
    #load the model
    model = UNet()
    loaded_data = torch.load("model_data_epoch_24.pth")
    loaded_model_state_dict = loaded_data['model_state_dict']
    model.load_state_dict(loaded_model_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #generate the predictions
    predictions = []
    i = 0
    for image in data['images']:
        print(i)
        normalized_image = np.maximum((image - image.min()) / (image.max() - image.min()), 0)
        normalized_image = normalized_image[np.newaxis, :, :]
        image_tensor = torch.tensor(normalized_image, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            predicted_output = model(image_tensor)
        predicted_segmentation = postprocess_output(predicted_output)
        predictions.append(predicted_segmentation)
        i += 1
    #save the predictions
    predictions = [np.where(prediction >= 0.01, 1, 0) for prediction in predictions]
    
    data['predictions'] = predictions
    with open('dataset_workstation_3_with_predictions.pkl', 'wb') as f:
        pickle.dump(data, f)
    # %%