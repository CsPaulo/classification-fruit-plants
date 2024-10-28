import streamlit as st
import cv2
import mahotas
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from eli5.sklearn import PermutationImportance
import eli5
from streamlit import components



# Carregar o modelo
def get_model():
    return pickle.load(open('C:/Users/cspau/Desktop/coisas do pc/Aprendendo Python/GitHub/leaf-diagnostic-haralick/etc/best_xgboost_model.dat', 'rb'))

# Converter dados da imagem
def convert_byteio_image(string):
    array = np.frombuffer(string, np.uint8)
    image = cv2.imdecode(array, flags=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Função para interpretar e exibir previsões e importância das características
def interpret_and_display_predictions(model, image, features, class_names):
    prediction_probs = model.predict_proba([features])
    class_idx = np.argmax(prediction_probs)
    prediction = f"Folha classificada como '{class_names[class_idx]}' com {prediction_probs[0][class_idx]:.2%} de certeza"
    st.markdown(f"<h4 style='text-align: center; color: white;'>{prediction}</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    
    col1.image(image, use_column_width=True)

    # Exibir importância das características usando eli5
    explanation = eli5.show_weights(model)
    raw_html = explanation._repr_html_()
    
    with col2:
        components.v1.html(raw_html, width=180, height=340)
     


# Título 
st.markdown("<h1 style='text-align: center; color: white;'>Classificação de Plantas Frutíferas</h1>", unsafe_allow_html=True)

# Barra lateral para upload de imagens
st.sidebar.title('Configurações')
uploaded_images = st.sidebar.file_uploader("Escolha até 10 imagens (JPG ou JPEG)", type=['jpg', 'jpeg'], accept_multiple_files=True, key="upload_images")

model = get_model()

if uploaded_images:
    images_and_features = []
    
    st.markdown("<h3 style='text-align: center; color: white;'>Imagens e Previsões</h3>", unsafe_allow_html=True)

    for idx, uploaded_image in enumerate(uploaded_images, start=1):
        bytes_data = uploaded_image.getvalue()
        image = convert_byteio_image(bytes_data)

        if image.shape != (256, 256):
            image = cv2.resize(image, (256, 256))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Características da imagem
        features = mahotas.features.haralick(gray_image, compute_14th_feature=True, return_mean=True).reshape(14,)
        
        images_and_features.append((image, features))

    for idx, (img, features) in enumerate(images_and_features, start=1):
        class_names = ['Acerola', 'Amora', 'Bacuri', 'Banana', 'Caja', 
          'Caju', 'Goiaba', 'Graviola', 'Mamão', 'Manga',
          'Maracuja', 'Pinha']  
        
        # Exibe previsões e importância das características para cada imagem
        interpret_and_display_predictions(model, img, features, class_names)
