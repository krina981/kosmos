import streamlit as st
import requests
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForVision2Seq

st.title('Image Captioning')

# Get inputs from user
image_url = st.text_input('Enter Image URL:')
prompt = st.text_input('Enter Prompt:')

if st.button('Process'):
    # Load image from URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Load model and processor
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224", load_in_4bit=True, device_map={"":0})

    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")

    # Generate completion
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Post-process generation
    processed_text, entities = processor.post_process_generation(generated_text)

    # Draw bounding boxes on image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for entity, _, box in entities:
        box = [round(i, 2) for i in box[0]]
        x1, y1, x2, y2 = tuple(box)
        x1, x2 = x1*width, x2*width
        y1, y2 = y1*height, y2*height
        draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
        draw.text(xy=(x1, y1), text=entity)

    # Display processed image and text
    st.image(image, caption=processed_text, use_column_width=True)
