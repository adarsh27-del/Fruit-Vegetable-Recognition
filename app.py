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

IMG_SIZE = (180, 180)

def predict(image):
    image = image.convert("RGB")
    image = image.resize((180, 180))
    image = np.array(image, dtype=np.float32)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)
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

import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)