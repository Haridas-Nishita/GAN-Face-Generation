from flask import Flask, Response
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Define the generator model architecture
def generator_model(z_dim=100):
    gen_input = keras.Input(shape=(z_dim,), name='generator_network')
    # Define the layers as per the training script
    # ...
    model_gen = keras.Model(inputs=gen_input, outputs=fake_images_gen)
    return model_gen

# Load the generator model
gen_model = generator_model(z_dim=100)
gen_model.load_weights('generator_model_weights.h5')

@app.route('/generate_image')
def generate_image():
    print("Generating image...")
    z_dim = 100
    noise = np.random.normal(0, 1, (1, z_dim))
    generated_image = gen_model.predict(noise)[0]

    # Post-process the image
    generated_image = np.squeeze(generated_image)
    generated_image = ((generated_image * 255)).astype(np.uint8)

    # Convert the array to a PIL Image with 'RGB' mode
    image = Image.fromarray(generated_image, 'RGB')

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG')
    buffer.seek(0)

    # Encode the image in base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Create an HTML response with the image
    html = f'<img src="data:image/jpeg;base64,{img_base64}"/>'
    return html

if __name__ == '__main__':
    app.run(debug=True)