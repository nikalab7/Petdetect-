"""Local interactive demo: streamlit run app.py."""
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from core import load_checkpoint, predict_image

st.set_page_config(page_title='PetBreed AI', page_icon='🐾')
st.title('🐾 PetBreed AI')
st.write('Upload a photo of a cat or dog to estimate its breed. Predictions may be wrong.')

@st.cache_resource
def get_model():
    return load_checkpoint('artifacts/best.pt', torch.device('cpu'))

if not Path('artifacts/best.pt').exists():
    st.error('Trained checkpoint not found at `artifacts/best.pt`. Restore the checkpoint file and try again.')
    st.stop()

model, saved = get_model()

cat_breeds = {
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair',
    'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll',
    'Russian Blue', 'Siamese', 'Sphynx',
}
supported_breeds = sorted(saved['class_names'])
cats = [breed for breed in supported_breeds if breed in cat_breeds]
dogs = [breed for breed in supported_breeds if breed not in cat_breeds]

st.caption(
    'This model recognizes only the 37 breeds in the list below. '
    'Other breeds and animals may be assigned an incorrect label.'
)
with st.expander(f"See all {len(supported_breeds)} supported breeds"):
    cat_col, dog_col = st.columns(2)
    with cat_col:
        st.markdown(f"**Cats ({len(cats)})**")
        st.markdown('\n'.join(f'- {breed}' for breed in cats))
    with dog_col:
        st.markdown(f"**Dogs ({len(dogs)})**")
        st.markdown('\n'.join(f'- {breed}' for breed in dogs))

file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
if file:
    try:
        if file.size > 10 * 1024 * 1024:
            st.error('Maximum file size is 10 MB.')
            st.stop()
        image = Image.open(file)
        image.load()
        image = image.convert('RGB')
        if image.width * image.height > 20_000_000:
            st.error('Image dimensions are too large.')
            st.stop()
        st.image(image, caption='Uploaded photo', width='stretch')
        results = predict_image(model, image, saved['image_size'], torch.device('cpu'), saved['class_names'])
        st.subheader('Top 3 predictions')
        for breed, probability in results:
            st.write(f'**{breed}** — {probability:.1%}')
            st.progress(probability)
        st.caption('Softmax probability is not a calibrated measure of certainty. The model only knows 37 pet breeds.')
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f'Could not read this image: {exc}')
