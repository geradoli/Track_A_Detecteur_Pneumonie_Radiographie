%%writefile app.py
import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="Detecteur de pneumonie", page_icon="🔬", layout="wide")

# ================================================================
# Helpers
# ================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def denormalize(tensor):
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)

def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class ViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(pixel_values=x).logits

# ================================================================
# Chargement modèle (cached)
# ================================================================
@st.cache_resource
def load_model_and_cam():
    # Cherche un checkpoint local, sinon utilise le modèle de base HuggingFace
    results_dir = './vit_xray_results'
    if os.path.exists(results_dir):
        checkpoints = sorted([
            os.path.join(results_dir, d)
            for d in os.listdir(results_dir)
            if d.startswith('checkpoint')
        ])
        source = checkpoints[-1] if checkpoints else "google/vit-base-patch16-224"
    else:
        source = "google/vit-base-patch16-224"

    model = AutoModelForImageClassification.from_pretrained(
        source,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    wrapped      = ViTWrapper(model)
    target_layer = wrapped.model.vit.encoder.layer[-1].layernorm_before
    cam = GradCAM(
        model=wrapped,
        target_layers=[target_layer],
        reshape_transform=vit_reshape_transform
    )
    return wrapped, cam

# ================================================================
# Sidebar
# ================================================================
st.sidebar.title("A propos")
st.sidebar.info("Outil d'aide a la decision — ne remplace pas un diagnostic medical. Consultez un medecin.")
st.sidebar.markdown("---")
seuil = st.sidebar.slider("Seuil de decision", 0.10, 0.90, 0.40, 0.05,
                           help="Abaisser = plus de recall, plus de faux positifs")
st.sidebar.markdown("---")
st.sidebar.markdown("**Modele :** ViT-Base")
st.sidebar.markdown("**Classes :** NORMAL, PNEUMONIA")

# ================================================================
# Page principale
# ================================================================
st.title("Detecteur de pneumonie")
st.markdown("Uploadez une image pour obtenir une prediction avec score de confiance et visualisation GradCAM.")

uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"],
                                  help="Formats acceptes : JPG, JPEG, PNG")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Chargement du modele..."):
        wrapped, cam = load_model_and_cam()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analyse")

        with st.spinner("Analyse en cours..."):

            # 1. Prétraitement
            img_tensor = val_transform(image)

            # 2. Prédiction
            with torch.no_grad():
                logits = wrapped(img_tensor.unsqueeze(0))

            # 3. Softmax → probabilités
            probs_np    = torch.softmax(logits, dim=-1).numpy()[0]
            p_normal    = float(probs_np[0])
            p_pneumonia = float(probs_np[1])
            prediction  = 'PNEUMONIA' if p_pneumonia >= seuil else 'NORMAL'
            confidence  = p_pneumonia if prediction == 'PNEUMONIA' else p_normal

            # 4. GradCAM
            grayscale_cam = cam(
                input_tensor=img_tensor.unsqueeze(0),
                targets=[ClassifierOutputTarget(1)]
            )
            img_rgb = denormalize(img_tensor).permute(1, 2, 0).numpy()
            img_rgb = np.clip(img_rgb, 0, 1).astype(np.float32)
            overlay = show_cam_on_image(img_rgb, grayscale_cam[0], use_rgb=True)

        # 5. Affichage résultats
        emoji = "🔴" if prediction == 'PNEUMONIA' else "🟢"
        st.metric("Prediction", f"{emoji} {prediction}",
                  f"Confiance : {confidence*100:.1f}%")

        st.markdown("**Probabilites :**")
        st.progress(p_normal,    text=f"🟢 NORMAL      {p_normal*100:.1f}%")
        st.progress(p_pneumonia, text=f"🔴 PNEUMONIA  {p_pneumonia*100:.1f}%")
        st.caption(f"Seuil actuel : {seuil}")

        if 0.35 <= p_pneumonia <= 0.65:
            st.warning("Zone d'incertitude — revision humaine recommandee.")

    # GradCAM pleine largeur
    st.markdown("---")
    st.subheader("Visualisation GradCAM — zones d'attention du modele")
    col3, col4 = st.columns(2)
    with col3:
        st.image(img_rgb, caption="Image pretraitee (224x224)", use_container_width=True)
    with col4:
        st.image(overlay, caption="Heatmap GradCAM (rouge = attention max)", use_container_width=True)

    st.info("Rouge/jaune = zones activees  |  Bleu = zones ignorees  |  Ideal : activation sur les poumons")
    st.markdown("---")
    st.warning("Outil d'aide a la decision — ne remplace pas un diagnostic medical. Consultez un medecin.")

else:
    st.info("Uploadez une image pour commencer l'analyse.")