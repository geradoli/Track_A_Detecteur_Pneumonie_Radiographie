import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
# TODO : importer votre modele et GradCAM

# ── Configuration de la page ──
st.set_page_config(
    page_title="Detecteur de pneumonie",
    page_icon="🔬",
    layout="wide"
)

# ── Sidebar ──
st.sidebar.title("A propos")
st.sidebar.info(
    "Outil d'aide a la decision — ne remplace pas un diagnostic medical. Consultez un medecin."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modele :** ViT-Base")
st.sidebar.markdown("**Classes :** NORMAL, PNEUMONIA")

# ── Page principale ──
st.title("Detecteur de pneumonie")
st.markdown(
    "Uploadez une image pour obtenir une prediction "
    "avec score de confiance et visualisation GradCAM."
)

# ── Upload d'image ──
uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptes : JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Layout en 2 colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analyse")

        with st.spinner("Analyse en cours..."):
            # TODO :
            # 1. Appliquer val_transform a l'image
            # 2. Predire avec le modele
            # 3. Calculer les probabilites (softmax)
            # 4. Generer le GradCAM
            # 5. Afficher les resultats
            pass

        # Exemple d'affichage des resultats :
        # st.metric(label="Prediction", value="PNEUMONIA", delta="92.3%")
        # st.image(gradcam_overlay, caption="Heatmap GradCAM", use_container_width=True)

    # ── Disclaimer en bas ──
    st.markdown("---")
    st.warning("Outil d'aide a la decision — ne remplace pas un diagnostic medical. Consultez un medecin.")

else:
    st.info("Uploadez une image pour commencer l'analyse.")
