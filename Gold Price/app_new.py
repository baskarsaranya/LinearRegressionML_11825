import gradio as gr
import pickle
import numpy as np

scaler = pickle.load(open('C:/Users/baska/Desktop/New Data_journey/Data/Youtube_Stackoverflow/Gold Price/scaler.pkl','rb'))
model = pickle.load(open('C:/Users/baska/Desktop/New Data_journey\Data/Youtube_Stackoverflow/Gold Price/regressor.pkl','rb'))

def calculate_goldrate(usd_inr):
    scaled_input = scaler.transform(np.array(usd_inr).reshape(1,-1))
    return round(model.predict(scaled_input)[0][0],2)

demo = gr.Interface(
    fn=calculate_goldrate,
    inputs=["number"],
    outputs=["number"],
    title="How much is 1g of gold in India NOW?"
)

demo.launch(share=True)