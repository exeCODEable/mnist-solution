# --------------------------------------------
# MNIST Neural Network Challenge.
# Credit : Nicolas Ronette's video
# https://www.youtube.com/watch?v=mozBidd58VQ
# --------------------------------------------

# --------------------------------------------
# Importing the dependences.
# Torch
from torch import nn , save, load  # our Neural Network Class, Save and Load features.
from torch.optim import Adam  # our Optimizer
from torch.utils.data import DataLoader  # our Data Loader
# Torch Vision
from torchvision import datasets # to download our MNIST dataset
from torchvision.transforms import ToTensor # to transfrom our images into tensors
#  Needed to be able to Load / Use the 'model.pt' file
import torch
from PIL import Image

# Downloading the Dataset
# MNIST has 10 classes '0-9' << this is what I need to figure out for the glyphs.


# >>>> Setting up Training and Dataset
# The Training dataset
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
# Create our dataset using the Loader. into baches of 32 images.
dataset = DataLoader(train, 32)
# The images are 1, 28, 28  // 10 Classes (features) 0-9

# >>>> Creating our Image Classifer Class Neural Network Class
class ImageClassifier(nn.Module):
    # The init funciton.
    def __init__(self):
        super().__init__()
        # Create our model.
        self.model = nn.Sequential(
            # Convenultion NN layer.
            # 1 = black and white images.
            # 32 = the batch number
            # 3,3 = the images are 3 x 3
            # Shaves off 2 pixels of the height and width of each image.
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(), # Activation layer.
            # Shaves off 2 pixels of the hieght and width of each image.
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),  # Activation layer.
            # Shaves off 2 pixels of the hieght and width of each image.
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # Activation layer.
            nn.Flatten(), # Flatter the layers.
            # input shape-64: The final channel output from the last Conv Layer
            # 28-6 : the image size (minus) the shaved pixels.
            # output-10: the number of classes
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    # the forward function (akin to the Call method in TensorFow.)
    def forward(self, x):
        return self.model(x)


# >>> Create an Instance of the NN, Loss fucntion, optimizer

# CUDA or CPU
'''
device = torch.device("cuda=0" if torch.cuda.is_available() else 'cpu') 
model.to(device)
'''

# CPU -- I don't have access to CUDA/GPU
device = 'cpu'

# The Setup
clf = ImageClassifier().to(device) # NN Instance CPU.
opt = Adam(clf.parameters(), lr=1e-3) # Optimizer with lr(Learing Rate)
loss_fn = nn.CrossEntropyLoss() # Our Loss Function.


# Creating the Training Function
def training_function():
    # train for 10 epoch
    for epoch in range(10):
        for batch in dataset:
            x, y = batch  # to unpack the data
            x, y = x.to(device), y.to(device)  # sending the unpacked data to the your device.
            yhat = clf(x)  # to make a prediction
            loss = loss_fn(yhat, y)  # to claculate our loss

            # Applying Backpropagation
            opt.zero_grad()  # to zero out the gradient
            loss.backward()  # calculate the gradients, backwards.
            opt.step()  # taking a step to apply gradient descent

        # Print out the loss for every batch.
        print(f'Epoch: {epoch + 1} -- Loss {loss.item()}')

        # Saving the model to be used on its own.
        with open('model_1.pt', 'wb') as f:
            save(clf.state_dict(), f)


# Use the model_1.pt to predict the test images.
def pred_model():
    # open up the model pt file
    with open('model_1.pt', 'rb') as f:
        # Load the weights into the classifier
        clf.load_state_dict(load(f))
        # Load the test images
        test_image = 'img_3.jpg'
        img = Image.open(test_image)
        # Convert to tensor
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)

        """ 
        The test images are:
            img_1 is '2'
            img_2 is '0'
            img_3 is '9'
        """

        # Print prediction to screen.
        print(torch.argmax(clf(img_tensor))) # Classifer (image tensor) prediction




# Main Guard
if __name__ == "__main__":
   # Call the training function
   # training_function() # Commented so that it doesn't runt the training again.
   # Call the prediction model.
   pred_model()

