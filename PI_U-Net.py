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
from torch.optim.lr_scheduler import StepLR
import cv2

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1),
                nn.ReLU(inplace=True)
            )
        
        def upconv_block(in_channels,out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels,kernel_size=2,stride=2)
        

        #Downsampling
        self.conv1 = conv_block(1,64)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = conv_block(64,128)
        self.pool2 = nn.MaxPool2d(2,2)

        #Bridge
        self.conv3 = conv_block(128,256)
        self.conv4 = conv_block(256,256)

        #Upsampling
        self.upconv5 = upconv_block(256, 128) #this makes it 128. Then we concat the 128 from the downsampling making 256
        self.conv5 = conv_block(256, 128) #Therefore in_channels is 256
        self.upconv6 = upconv_block(128, 64)
        self.conv6 = conv_block(128, 64)

        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)

        x5 = self.conv3(x4)
        x6 = self.conv4(x5)

        x7 = self.upconv5(x6)
        x8 = torch.cat([x7, x3], dim=1)
        x9 = self.conv5(x8)
        
        x10 = self.upconv6(x9)
        x11 = torch.cat([x10, x1], dim=1)
        x12 = self.conv6(x11)

        out = self.output(x12)

        return out

with open('data_boi.pkl', 'rb') as f:
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
    epochs = range(1, len(train_losses) + 1)
    
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
    
    train_images, validation_images, train_ground_truths, validation_ground_truths = train_test_split(
        data['images'], data['exponential_ground_truths'], test_size=0.2, random_state=1337)

    train_dataset = CustomDataset(train_images, train_ground_truths)
    validation_dataset = CustomDataset(validation_images, validation_ground_truths)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=0)
    image, ground_truth = validation_dataset[0]
    print(count_black_pixels(ground_truth))

    def weighted_mse_loss(pred, target, weights=None):
        diff = pred - target
        squared_diff = diff ** 2

        if weights is not None:
            squared_diff = squared_diff * weights

        loss = torch.mean(squared_diff)
        return loss
    
    def visualize_output(image, ground_truth, predicted_output):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot input image
        axs[0].imshow(image.squeeze(0), cmap='gray')
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        # Plot ground truth
        axs[1].imshow(ground_truth.squeeze(0), cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        # Plot predicted output
        axs[2].imshow(predicted_output.squeeze(0), cmap='gray')
        axs[2].set_title("Predicted Output")
        axs[2].axis("off")

        plt.show()


    model = UNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    learning_rate = 0.001  # Set the desired learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 35
    
    train_losses = []
    validation_losses = []
    min_validation_loss = float("inf")


    scheduler = StepLR(optimizer, step_size=10, gamma=0.0002)

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
            weights = (targets > 0).float() * 45 + 1  # Create the weight tensor
            #print(weights.max())
            loss = weighted_mse_loss(predictions, targets, weights=weights)

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
                weights = (targets > 0).float() * 45 + 1
                loss = weighted_mse_loss(predictions,targets, weights=weights)
                validation_loss += loss.item()

        validation_loss /= len(validation_dataloader)
        validation_losses.append(validation_loss)       
        
        if validation_loss < min_validation_loss:
            torch.save(model.state_dict(), "best_weights.pth")
            min_validation_loss = validation_loss       
        
        model.eval()
        with torch.no_grad():
            sample_idx = 1 # Choose a random sample
            image, ground_truth = validation_dataset[sample_idx]
            image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            prediction = model(image_tensor)
            predicted_output = prediction.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU

            visualize_output(image, ground_truth, predicted_output)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Validation Loss: {validation_loss}, Current lr: {current_lr}")
    
    return model,train_losses, validation_losses
#%%
if __name__ == '__main__':
    trained_model, train_losses, validation_losses = main()
    plot_losses(train_losses, validation_losses)
    trained_model.load_state_dict(torch.load("best_weights.pth"))

    image_path = "data/experimental_data/32bit/02.tif"
    image_tensor = preprocess_image(image_path)
    
    # Add the batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the image tensor to the appropriate device (GPU or CPU)
    image_tensor = image_tensor.to(device)

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
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_segmentation, cmap='gray')
    plt.title('Predicted Segmentation')

    plt.show()
#%%