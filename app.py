import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(page_title='DriveGuard AI', page_icon='🚗', layout='wide')

st.markdown('''<style>
.block-container{max-width:1200px;padding-top:2rem}
.hero{padding:32px;border-radius:22px;background:linear-gradient(135deg,#0f172a,#334155);color:white;margin-bottom:24px}
.card{background:white;padding:20px;border-radius:16px;border:1px solid #e5e7eb;box-shadow:0 5px 18px rgba(15,23,42,.06)}
</style>''', unsafe_allow_html=True)

MODEL_PATH='EfficientNet_B0.keras'
CLASSES=['Closed','Open','no_yawn','yawn']
RESULTS=pd.DataFrame({
 'Model':['CNN','MobileNetV3-Small','EfficientNet-B0'],
 'Accuracy':[72.29,84.53,90.53], 'Precision':[72.86,85.79,91.19],
 'Recall':[72.29,84.53,90.53], 'F1-Score':[72.11,84.15,90.44]})

@st.cache_resource

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model=load_model()
    model_ok=True
except Exception as e:
    model=None; model_ok=False; model_error=str(e)

with st.sidebar:
    st.markdown('## 🚗 DriveGuard AI')
    st.caption('Intelligent Driver Safety System')
    page=st.radio('Navigation',['🏠 Dashboard','🔍 Drowsiness Detection','📊 Model Performance','📘 About Project'])
    st.divider()
    st.success('🟢 Model loaded') if model_ok else st.error('🔴 Model unavailable')

st.markdown('''<div class="hero"><h1>🚗 DriveGuard AI</h1><p>AI-Powered Driver Drowsiness Detection & Intelligent Road Safety Assistance</p></div>''',unsafe_allow_html=True)

if page=='🏠 Dashboard':
    st.header('Intelligent Road Safety at a Glance')
    st.write('Deep-learning based visual analysis for eye/yawn images.')
    a,b,c,d=st.columns(4)
    a.metric('Best Model','EfficientNet-B0'); b.metric('Test Accuracy','90.53%'); c.metric('Classes','4'); d.metric('Model Status','Online' if model_ok else 'Offline')
    st.markdown('### 🧠 System Flow')
    st.info('Input → Preprocessing → EfficientNet-B0 → Prediction → Safety Message')
    st.success('🏆 EfficientNet-B0 achieved the highest reported test accuracy: 90.53%.')

elif page=='🔍 Drowsiness Detection':
    st.header('🔍 Drowsiness Detection')
    st.write('Upload an image or use your device camera for a single-image prediction.')
    if not model_ok:
        st.error('Model could not be loaded.'); st.code(model_error); st.stop()
    source=st.radio('Input method',['📁 Upload Image','📷 Camera'],horizontal=True)
    image=None
    if source=='📁 Upload Image':
        f=st.file_uploader('Choose an image',type=['jpg','jpeg','png'])
        if f: image=Image.open(f).convert('RGB')
    else:
        f=st.camera_input('Take a picture')
        if f: image=Image.open(f).convert('RGB')
    if image:
        left,right=st.columns(2)
        with left: st.image(image,caption='Input Image',use_container_width=True)
        x=np.asarray(image.resize((224,224)),dtype=np.float32)/255.0
        probs=model.predict(np.expand_dims(x,0),verbose=0)[0]
        i=int(np.argmax(probs)); label=CLASSES[i]; conf=float(probs[i])*100
        with right:
            st.metric('Prediction',label); st.metric('Confidence',f'{conf:.2f}%')
            if label in ['Closed','yawn']:
                st.warning('⚠️ Possible drowsiness-related visual sign detected. Please take a break if tired.')
            else:
                st.success('✅ No strong drowsiness-related visual sign predicted.')
        df=pd.DataFrame({'Class':CLASSES,'Probability (%)':[float(p)*100 for p in probs]}).sort_values('Probability (%)',ascending=False)
        st.subheader('📊 Class Probabilities'); st.dataframe(df.style.format({'Probability (%)':'{:.2f}%'}),use_container_width=True,hide_index=True)
        st.caption('Mini-project demonstration; not a certified automotive safety system.')

elif page=='📊 Model Performance':
    st.header('📊 Model Performance Comparison')
    a,b,c=st.columns(3); a.metric('CNN','72.29%'); b.metric('MobileNetV3-Small','84.53%'); c.metric('🏆 EfficientNet-B0','90.53%')
    st.dataframe(RESULTS.style.format({k:'{:.2f}%' for k in ['Accuracy','Precision','Recall','F1-Score']}),use_container_width=True,hide_index=True)
    st.success('🏆 EfficientNet-B0 is the best-performing model in the reported experiment.')

else:
    st.header('📘 About Project')
    st.markdown('### Problem Statement')
    st.write('AI Agent for Driver Drowsiness Detection and Intelligent Road Safety Assistance.')
    a,b=st.columns(2)
    with a:
        st.markdown('### 🎯 Objective'); st.write('Detect visual signs related to eye closure and yawning and present a clear safety-oriented result.')
        st.markdown('### 🗂️ Dataset'); st.write('Yawn-Eye Dataset — Closed, Open, no_yawn, yawn')
    with b:
        st.markdown('### 🤖 Models'); st.write('CNN • MobileNetV3-Small • EfficientNet-B0')
        st.markdown('### 📏 Evaluation'); st.write('Accuracy • Precision • Recall • F1-score')
    st.info('Dataset → Preprocessing → Training → Comparison → Best Model → Prediction → Safety Message')
    st.warning('This is a college mini-project demonstration, not a certified automotive safety or medical device.')

st.divider(); st.caption('DriveGuard AI • CNN | MobileNetV3-Small | EfficientNet-B0')
