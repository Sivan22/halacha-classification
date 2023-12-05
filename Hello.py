import streamlit as st
from streamlit.logger import get_logger
from transformers import  pipeline

LOGGER = get_logger(__name__)
model = "sivan22/halacha-siman-classifier"


def get_predicts_local(input)->str:
    classifier = pipeline("text-classification",model=model,top_k=5)
    predicts = classifier(input)
    return predicts

def get_predicts_online(input)->str:
    import requests
    API_URL = "https://api-inference.huggingface.co/models/" + model
    headers = {"Authorization": f"Bearer {'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'}"}
    def query(input_text):
        response = requests.post(API_URL, headers=headers, json='{inputs:' +input_text+'}')
        if response.status_code == 503:
            response = requests.post(API_URL, headers=headers, json='{{inputs:' +input_text+'}{wait_for_model:true}{top_k:5}}')        
        return response.json()
    predicts = query(input)
    return predicts

def run():
    st.set_page_config(
        page_title="Halacha classification",
        page_icon="",
    )

    st.write("# חיפוש בשולחן ערוך")
    use_local = st.checkbox("חיפוש לא מקוון")
    user_input = st.text_input('כתוב כאן את שאלתך', placeholder='כמה נרות מדליקים בחנוכה')
    if st.button('השב'):
        if(user_input!="" ):
            if use_local:
                for prediction in get_predicts_local(user_input)[0][:5]:
                        st.write('סימן ' + str(prediction['label']))
            else:            
                for prediction in get_predicts_online(user_input)[0][:5]:
                    st.write('סימן ' + str(prediction['label']))

    

if __name__ == "__main__":
    run()
