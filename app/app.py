import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 64 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Load Trained Model
model = CNN()

model.load_state_dict(
    torch.load(
        "../model/mnist_cnn.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prediction Function
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 64 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Load Trained Model
model = CNN()

model.load_state_dict(
    torch.load(
        "../model/mnist_cnn.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prediction Function
def predict(image):

    image = Image.fromarray(image.astype("uint8"))

    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(output, dim=1).item()

    return f"✅ Predicted Digit: {prediction}"

# Gradio Interface
interface = gr.Interface(
    fn=predict,

    inputs=gr.Image(
        type="numpy",
        image_mode="L"
    ),

    outputs=gr.Textbox(label="Prediction"),

    title="🧠 Handwritten Digit Recognition",

    description="""
Upload or draw a handwritten digit image (0–9)
and the CNN model will predict the digit.
""",

    theme="soft"
)

# Launch App
interface.launch()
# Gradio Interface
interface = gr.Interface(
    fn=predict,

    inputs=gr.Image(
        type="numpy",
        image_mode="L"
    ),

    outputs=gr.Textbox(label="Prediction"),

    title="🧠 AI-Powered Handwritten Digit Recognition System",

    description="""
Upload or draw a handwritten digit image (0–9)
and the CNN model will predict the digit.
""",

    theme="soft"
)

# Launch App
interface.launch()