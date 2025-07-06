import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os

# Set your Gemini API key in Streamlit secrets or environment variable
# It's recommended to use st.secrets for deployment
GEMINI_KEY = os.getenv("GEMINI_KEY") or st.secrets.get("GEMINI_KEY")

if not GEMINI_KEY:
    st.error("Gemini API key not found. Please set it in your environment variables or Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=GEMINI_KEY)

st.set_page_config(layout="wide") # Use wide layout for better side-by-side display
st.title("Gemini Image Generation")

# --- Session State Initialization ---
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "image_history" not in st.session_state:
    st.session_state.image_history = []
if "last_generated_image" not in st.session_state:
    st.session_state.last_generated_image = None
if "last_text_response" not in st.session_state:
    st.session_state.last_text_response = ""

# --- Initial Image Generation ---
st.header("Start a new image generation")
uploaded_images = st.file_uploader("Upload one or more images (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
initial_prompt = st.text_area("Enter your initial prompt for image generation", key="initial_prompt_area")

if st.button("Generate Initial Image", key="generate_initial"):
    if not initial_prompt:
        st.warning("Please enter an initial prompt.")
    else:
        st.info("Generating initial image... This may take a moment.")
        try:
            contents = [initial_prompt]
            if uploaded_images:
                images_for_content = [Image.open(img) for img in uploaded_images]
                contents.extend(images_for_content)
                st.session_state.image_history = images_for_content # Store for follow-up

            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )

            generated_text = ""
            generated_image = None

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    generated_text += part.text + "\n"
                elif part.inline_data is not None:
                    generated_image = Image.open(BytesIO(part.inline_data.data))

            if generated_image:
                st.session_state.last_generated_image = generated_image
                st.session_state.last_text_response = generated_text
                st.session_state.generated_images.append(generated_image) # Keep track of all generated images
                st.success("Initial image generated successfully!")
            else:
                st.warning("No image was generated. The model might have only returned text.")
                st.session_state.last_text_response = generated_text
        except Exception as e:
            st.error(f"An error occurred during initial generation: {e}")

## Refine Your Image

# --- Follow-up Image Generation ---
if st.session_state.last_generated_image:
    col1, col2 = st.columns([0.6, 0.4]) # Adjust column ratios as needed

    with col1:
        st.subheader("Last Generated Image")
        st.image(st.session_state.last_generated_image, caption="Current Image", use_container_width=True)
        if st.session_state.last_text_response:
            st.markdown(f"**Description:**\n{st.session_state.last_text_response}")

    with col2:
        st.subheader("Modify/Refine")
        followup_prompt = st.text_area(
            "Enter your next prompt to modify the image:",
            key="followup_prompt_area",
            height=150
        )
        followup_uploaded_images = st.file_uploader(
            "Upload additional images for context (optional)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="followup_uploader"
        )

        if st.button("Generate Next Version", key="generate_next_version"):
            if not followup_prompt:
                st.warning("Please enter a prompt for the next version.")
            else:
                st.info("Generating next image version... This may take a moment.")
                try:
                    contents_followup = [followup_prompt]

                    # Add the last generated image to the content for modification
                    if st.session_state.last_generated_image:
                        contents_followup.append(st.session_state.last_generated_image)

                    # Add any newly uploaded images for context
                    if followup_uploaded_images:
                        new_context_images = [Image.open(img) for img in followup_uploaded_images]
                        contents_followup.extend(new_context_images)
                        # Optionally, update image_history if you want these to persist
                        st.session_state.image_history.extend(new_context_images)


                    response_followup = client.models.generate_content(
                        model="gemini-2.0-flash-preview-image-generation",
                        contents=contents_followup,
                        config=types.GenerateContentConfig(
                            response_modalities=['TEXT', 'IMAGE']
                        )
                    )

                    generated_text_followup = ""
                    generated_image_followup = None

                    for part in response_followup.candidates[0].content.parts:
                        if part.text is not None:
                            generated_text_followup += part.text + "\n"
                        elif part.inline_data is not None:
                            generated_image_followup = Image.open(BytesIO(part.inline_data.data))

                    if generated_image_followup:
                        st.session_state.last_generated_image = generated_image_followup
                        st.session_state.last_text_response = generated_text_followup
                        st.session_state.generated_images.append(generated_image_followup)
                        st.success("Image refined successfully!")
                        # Rerun to update the display
                        st.rerun()
                    else:
                        st.warning("No new image was generated. The model might have only returned text.")
                        st.session_state.last_text_response = generated_text_followup
                except Exception as e:
                    st.error(f"An error occurred during follow-up generation: {e}")

# --- Display All Generated Images (Optional) ---
if st.session_state.generated_images:
    st.markdown("---")
    st.subheader("All Generated Images in this Session")
    # Display images in a grid or gallery format if desired
    # For simplicity, we'll just show them stacked
    for i, img in enumerate(st.session_state.generated_images):
        st.image(img, caption=f"Generated Image {i+1}", width=200) # Smaller size for history