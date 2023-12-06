import streamlit as st
from streamlit.logger import get_logger
from transformers import  pipeline
import datasets
import pandas as pd
from huggingface_hub import login

LOGGER = get_logger(__name__)
model = "sivan22/halacha-siman-seif-classifier"


ds = datasets.load_dataset('sivan22/orach-chaim')
df = ds['train'].to_pandas()
def clean(s)->str:
    return s.replace(" ","")
df['seif']= df['seif'].apply(clean)



def get_predicts_local(input)->str:
    classifier = pipeline("text-classification",model=model,top_k=None)
    predicts = classifier(input)
    return predicts

def get_predicts_online(input)->str:
    import requests
    API_URL = "https://api-inference.huggingface.co/models/" + model
    headers = {"Authorization": f"Bearer {'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'}"}
    def query(input_text):
        response = requests.post(API_URL, headers=headers, json='{{inputs:' +input_text+'}{wait_for_model:true}}')        
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
    if st.button('חפש') and user_input!="":       
        get_predicts = get_predicts_local if use_local else get_predicts_online
        #print(get_predicts(user_input)[0][0:5])
        for prediction in get_predicts(user_input)[0][:5]:
            rows = df[((df["bookname"] == " שלחן ערוך - אורח חיים ") |
                        (df["bookname"] ==" משנה ברורה")) &
                      (df["siman"] == prediction['label'].split(' ')[0])&
                      (df["seif"] == prediction['label'].split(' ')[1]) ]
            rows.sort_values(["bookname"],ascending=False, inplace=True) 
         
            print(prediction['label'].split(' '))
            st.write('סימן ' + str(prediction['label']), rows[['text','sek','seif','siman','bookname']])

    

if __name__ == "__main__":
    run()
