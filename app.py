import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import os # Importar 'os' para checagem de arquivo

# --- 1. Carregar o Modelo ---
@st.cache_resource
def carrega_modelo():
    """Baixa o modelo do GDrive e o carrega na memória."""
    
    #https://drive.google.com/file/d/1ZunVB45Rxqma-QfQoASyQ-52TErIuErH/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1ZunVB45Rxqma-QfQoASyQ-52TErIuErH'
    output_path = 'TCC_modelo_quantizado_float16.tflite'
    
    # Baixar se não existir
    if not os.path.exists(output_path):
        st.write("Baixando modelo do Google Drive (só na primeira vez)...")
        gdown.download(url, output_path, quiet=False)
        st.write("Download concluído.")
    
    # Carregar o interpretador TFLite
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    return interpreter

# --- 2. Carregar e Pré-processar a Imagem (Com a lógica correta) ---
def carrega_e_prepara_imagem(interpreter):
    """Lida com o upload e o pré-processamento da imagem."""
    
    uploaded_file = st.file_uploader('Arraste e solte uma imagem ou clique aqui para selecionar uma', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image_pil = Image.open(io.BytesIO(image_data))
        
        # Garantir que a imagem é RGB (remove canal Alpha, se houver)
        if image_pil.mode == 'RGBA':
            image_pil = image_pil.convert('RGB')

        # --- Início da Lógica de Correção Final ---
        # Seguindo EXATAMENTE a lógica do seu Colab que funcionou

        # 1. Redimensionar para (256, 256)
        image_resized = image_pil.resize((256, 256))

        # 2. Converter para Array 3D (256, 256, 3)
        image_array = np.array(image_resized, dtype=np.float32)

        # 3. Aplicar o pré-processamento InceptionV3 no Array 3D
        # Converte os pixels de [0, 255] para [-1, 1]
        image_preprocessed = tf.keras.applications.inception_v3.preprocess_input(image_array)

        # 4. Adicionar a dimensão do "lote" (batch) para criar o Array 4D
        image_batch = np.expand_dims(image_preprocessed, axis=0)

        # 5. Garantir o tipo de dado final (float16)
        input_details = interpreter.get_input_details()
        input_dtype = input_details[0]['dtype']
        
        if input_dtype == np.float16:
            image_final = image_batch.astype(np.float16)
        else:
            image_final = image_batch
        
        # --- Fim da Lógica de Correção ---

        st.image(image_pil, caption="Imagem Enviada", width=256)
        st.success('Imagem foi carregada e processada com sucesso')
        return image_final
    
    return None

# --- 3. Fazer a Previsão ---
def previsao(interpreter, image):
    """Executa o modelo e exibe os resultados."""
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Ordem correta das classes (benign=0, malignant=1)
    classes = ['Benigno', 'Maligno']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = (100 * output_data[0]).round(2)

    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', 
                 text='probabilidades (%)', title='Probabilidade de Câncer Mamário')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)

# --- 4. Função Principal ---
def main():
    st.set_page_config(
        page_title="Classificador de Câncer Mamário",
        page_icon="🔬",
    )
    
    st.title("🔬 Classificador de Câncer Mamário em Animais")
    st.write("""
    Este aplicativo utiliza um modelo de Deep Learning (InceptionV3) 
    para classificar se uma imagem histopatológica indica um tumor 
    Benigno ou Maligno.
    """)

    interpreter = carrega_modelo()
    
    image_para_modelo = carrega_e_prepara_imagem(interpreter)

    if image_para_modelo is not None: 
        previsao(interpreter, image_para_modelo)

# --- 5. Ponto de Entrada (Corrigido) ---
if __name__ == "__main__":
    main()