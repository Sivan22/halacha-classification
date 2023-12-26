import streamlit as st
from streamlit.logger import get_logger
from transformers import  pipeline
import datasets
import pandas as pd
from huggingface_hub import login


LOGGER = get_logger(__name__)


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
    model = "sivan22/halacha-siman-seif-classifier-new"
    classifier = pipeline("text-classification",model=model,top_k=None)
    return classifier

def get_predicts(classifier,input)->str:
    predicts = classifier(input)
    return predicts

def run():
    
    st.set_page_config(
        page_title=" חיפוש חכם בשולחן ערוך",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"    
    )
    
    st.write("# (אורח חיים) חיפוש חכם בשולחן ערוך")
    
    classifier = get_model()    
    df = get_df()
    
    user_input = st.text_input('כתוב כאן את שאלתך', placeholder='כמה נרות מדליקים בכל לילה מלילות החנוכה')    
    num_of_results = st.sidebar.slider('מספר התוצאות שברצונך להציג:',1,25,5)
    
    if (st.button('חפש') or user_input) and user_input!="":
        predictions = get_predicts(classifier,user_input)[0][:num_of_results]
        for prediction in predictions:
            siman = prediction['label'].split(' ')[0]
            seif = prediction['label'].split(' ')[1]
            rows = df[((df["bookname"] == " שלחן ערוך - אורח חיים ") | (df["bookname"] ==" משנה ברורה")) &
                      (df["siman"] == siman) &
                      (df["seif"] == seif) ]
            rows = rows.sort_values(["bookname"],ascending=False) 
            st.write(('סימן ' + siman + ' סעיף ' + seif), rows[['text','bookname','sek','seif','siman',]])
            
        feedback_picker = st.sidebar.selectbox("עזור לי להשתפר! מהי התוצאה הנכונה ביותר לדעתך?",[ str(i+1)+') '+p['label']  for i,p in enumerate(predictions)])
        if st.sidebar.button("אישור"):           
            LOGGER.info("TEXT: " +user_input + "\t" +"LABEL: "+feedback_picker)
            st.sidebar.write("תודה על המשוב!")

if __name__ == "__main__":
    run()
