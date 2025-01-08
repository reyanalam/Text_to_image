import streamlit as st
from backend.image_generator import *
# Streamlit UI
st.title("Text-to-Layout Generator")
st.markdown(
    "Enter a description of the layout you want (e.g., 'Generate a frontend layout with a navigation bar and sidebar')."
)

# Text input
user_input = st.text_input("Enter your layout description:")

if st.button("Generate Layout"):
    if user_input.strip():
        st.write("Generating layout...")

        # Generate the layout image
        layout_image = generate_image(user_input)

        # Display the generated image
        st.image(layout_image, caption="Generated Layout", use_column_width=True)
    else:
        st.warning("Please enter a valid description.")
