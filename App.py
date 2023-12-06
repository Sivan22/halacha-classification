import streamlit as st
from streamlit.logger import get_logger
from transformers import  pipeline
import datasets
import pandas as pd


LOGGER = get_logger(__name__)
model = "sivan22/halacha-siman-seif-classifier"

@st.cache_data
ds = datasets.load_from_disk('sivan22/orach-chaim')
df = ds['train'].to_pandas()
def clean(s)->str:
    return s.replace(" ","")
df['seif']= df['seif'].apply(clean)

@st.cache_resource
def get_predicts(input)->str:
    classifier = pipeline("text-classification",model=model,top_k=None)
    predicts = classifier(input)
    return predicts

def run():
    st.set_page_config(
        page_title="Halacha classification",
        page_icon="",
    )

    st.write("# חיפוש בשולחן ערוך")
    user_input = st.text_input('כתוב כאן את שאלתך', placeholder='כמה נרות מדליקים בחנוכה')
    if st.button('חפש') and user_input!="":       
        for prediction in get_predicts(user_input)[0][:5]:
            rows = df[((df["bookname"] == " שלחן ערוך - אורח חיים ") |
                        (df["bookname"] ==" משנה ברורה")) &
                      (df["siman"] == prediction['label'].split(' ')[0])&
                      (df["seif"] == prediction['label'].split(' ')[1]) ]
            rows.sort_values(["bookname"],ascending=False, inplace=True) 
            st.write('סימן ' + str(prediction['label']), rows[['text','sek','seif','siman','bookname']])

    

if __name__ == "__main__":
    run()
