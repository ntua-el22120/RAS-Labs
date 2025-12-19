import torch
import torch.nn as nn
import cnn
import torch.optim as optim
import json
import cv2

# Load model
model = cnn.SimpleCNN().to(cnn.device)
model.load_state_dict(torch.load("simple_cnn_weights.pth", weights_only=True))
# Define training options
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Define size of train set
num_pictures = 5000

# Split to batches
batch_size = 20
batches = []
train_set_path = "C:\\Python Projects\\BlenderProc\\train_set\\"

for i in range(0, num_pictures, batch_size):
    batch_images = []
    batch_labels = []

    for j in range(i, i + batch_size, 1):
        # Load image from dataset
        image = cv2.imread(train_set_path+str(j)+".png")
        image_tensor = cnn.image_to_tensor(image)
        batch_images.append(image_tensor)
        # Load label from dataset
        with open(train_set_path+str(j)+".json", "r") as f:
            label = json.load(f)["name"]
        label_tensor = cnn.label_to_tensor(label)
        batch_labels.append(label_tensor)

    batches.append((batch_images, batch_labels))


# Train
for epoch in range(30):
    running_loss = 0
    for (batch_images, batch_labels) in batches:

        # Images: shape (batch_size, C, H, W)
        batch_images_tensor = torch.cat(batch_images, dim=0).to(cnn.device)
        # Labels: shape (batch_size,)
        batch_labels_tensor = torch.cat(batch_labels, dim=0).to(cnn.device)
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(batch_images_tensor)
        # Calculate loss
        loss = criterion(output, batch_labels_tensor)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Avg Loss: {running_loss/len(batches):.4f}")

torch.save(model.state_dict(), "simple_cnn_weights.pth")