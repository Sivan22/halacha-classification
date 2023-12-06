import streamlit as st
from streamlit.logger import get_logger
from transformers import  pipeline
import datasets
import pandas as pd
from huggingface_hub import login


LOGGER = get_logger(__name__)
<<<<<<< HEAD


@st.cache_data
def get_df() ->object:
    ds = datasets.load_from_disk('sivan22/orach-chaim')
    df = ds['train'].to_pandas()
    def clean(s)->str:
        return s.replace(" ","")
    df['seif']= df['seif'].apply(clean)
    return df

@st.cache_resource
def get_model()->object:
    model = "sivan22/halacha-siman-seif-classifier"
    classifier = pipeline("text-classification",model=model,top_k=None,device='cuda')
    return classifier

def get_predicts(classifier,input)->str:
=======
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
>>>>>>> 47b5e4bb255746c8628096b9bb5ce012a12f1263
    predicts = classifier(input)
    return predicts

def run():
    st.set_page_config(
        page_title="חיפוש חכם בשולחן ערוך",
        page_icon="",
    )
    st.write("# חיפוש בשולחן ערוך")
    classifier = get_model()
    
    df = get_df()
    user_input = st.text_input('כתוב כאן את שאלתך', placeholder='כמה נרות מדליקים בכל לילה מלילות החנוכה')    
    num_of_results = st.sidebar.slider('מספר התוצאות שברצונך להציג:',1,25,5)
    
    if st.button('חפש') and user_input!="":       
        for prediction in get_predicts(classifier,user_input)[0][:num_of_results]:
            rows = df[((df["bookname"] == " שלחן ערוך - אורח חיים ") |
                        (df["bookname"] ==" משנה ברורה")) &
                      (df["siman"] == prediction['label'].split(' ')[0])&
                      (df["seif"] == prediction['label'].split(' ')[1]) ]
            rows.sort_values(["bookname"],ascending=False, inplace=True) 
            st.write('סימן ' + str(prediction['label']), rows[['text','bookname','sek','seif','siman',]])

    

if __name__ == "__main__":
    run()
