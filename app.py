import streamlit as st
import PIL.Image as Image
from io import BytesIO 
import pandas as pd

from torchvision import transforms
from inference import label_to_food, preprocess_image, predict_V0, predict_V1, predict_V2, predict_V3

st.set_page_config(
    page_title="Food-Vision",
    page_icon="😋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.header('Food Vision', divider='rainbow')
st.write("These models are trained on the famous Food101 dataset. Here is the [link]() with the types of foods the models are able to predict")

uploaded_file = st.file_uploader("Choose a file", type = ['png', 'jpg'])
preprocessed_image = None
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    image = Image.open(BytesIO(bytes_data))
    # st.image(image, caption='Uploaded image')
    preprocessed_image = preprocess_image(image)

V3_column, V2_column, V1_column, V0_column = st.columns(4)

with V3_column:
    if st.button('Predict with Model V3'):
        if preprocessed_image != None:
            with st.spinner('Wait for it...'):
                label, confidence = predict_V3(preprocessed_image)
            food = label_to_food(label)
            st.success(f'Predicted {food} with a confidence of {confidence: .2f}%')
        else:
            st.write('Please upload an image')

     


with V2_column:
    if st.button('Predict with Model V2'):
        if preprocessed_image != None:
            with st.spinner('Wait for it...'):
                label, confidence = predict_V2(preprocessed_image)
            food = label_to_food(label)
            st.success(f'Predicted {food} with a confidence of {confidence: .2f}%')
        else:
            st.write('Please upload an image')
    

with V1_column:
    if st.button('Predict with Model V1'):
        if preprocessed_image != None:
            with st.spinner('Wait for it...'):
                label, confidence = predict_V1(preprocessed_image)
            food = label_to_food(label)
            st.success(f'Predicted {food} with a confidence of {confidence: .2f}%')
        else:
            st.write('Please upload an image')


with V0_column:
    if st.button('Predict with Model V0'):
        if preprocessed_image != None:
            with st.spinner('Wait for it...'):
                label, confidence = predict_V0(preprocessed_image)
            food = label_to_food(label)
            st.success(f'Predicted {food} with a confidence of {confidence: .2f}%')
        else:
            st.write('Please upload an image')

st.divider()
st.subheader("Model Comparison")
left_co, cent_co,last_co = st.columns([1,4,1])

data_df = pd.DataFrame(
    {
        "Models": ["3️⃣ V3","2️⃣ V2","1️⃣ V1","0️⃣ V0"],
        "Description":["Transfer Learning with Resnet-101", "Transfer Learning with Effecient Net V2 S",
                       "Standard CNN", "Standard MLP"],
        "Train Accuracy(Approximated %)": ["60", "45", "33", "11"],
        "Early Stopping": ["❌", "✅", "✅", "✅"],
        "LR Scheduler": [ "✅", "❌", "❌", "❌"]
    }
)

with cent_co:
    st.data_editor(
        data_df,
        hide_index=True,
    )

st.divider()
st.subheader("Metric Comparison")
V3_met, V2_met, V1_met, V0_met = st.columns(4)

with V3_met:
    st.image('metrics/model_V3.png', caption='Metrics for Model V3')

with V2_met:
    st.image('metrics/model_V2.png', caption='Metrics for Model V2')

with V1_met:
    st.image('metrics/model_V1.png', caption='Metrics for Model V1')

with V0_met:
    st.image('metrics/model_V0.png', caption='Metrics for Model V0')