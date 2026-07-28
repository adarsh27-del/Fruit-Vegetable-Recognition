import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("Model5_model.h5")

# Class names
class_names = [
    "Apple","Banana","Beetroot","Bell Pepper","Cabbage","Capsicum",
    "Carrot","Cauliflower","Chilli Pepper","Corn","Cucumber",
    "Eggplant","Garlic","Ginger","Grapes","Jalepeno","Kiwi",
    "Lemon","Lettuce","Mango","Onion","Orange","Paprika",
    "Pear","Peas","Pineapple","Pomegranate","Potato","Raddish",
    "Soy Beans","Spinach","Sweetcorn","Sweetpotato","Tomato",
    "Turnip","Watermelon"
]

IMG_SIZE = (64, 64)

def predict(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    index = np.argmax(prediction)

    return {
        class_names[index]: float(np.max(prediction))
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Fruit & Vegetable Recognition",
    description="Upload an image to identify the fruit or vegetable."
)

demo.launch()