# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from transformers import  pipeline

LOGGER = get_logger(__name__)
model = "sivan22/halacha-siman-classifier"


def get_predicts_local(input)->str:
    classifier = pipeline( model=model)
    predicts = classifier(input)
    return predicts

def get_predicts_online(input)->str:
    import requests
    API_URL = "https://api-inference.huggingface.co/models/sivan22/halacha-siman-seif-classifier"
    headers = {"Authorization": f"Bearer {'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'}"}
    def query(input_text):
        response = requests.post(API_URL, headers=headers, json='{inputs:' +input_text+'}')
        if response.status_code == 503:
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
    user_input = st.text_input('כתוב כאן את שאלתך', placeholder='כמה נרות מדליקים בחנוכה')
    if st.button('Predict'):
        if(user_input!="" ):
              for prediction in get_predicts_online(user_input):
                  st.write('סימן ' + str(prediction))
            

if __name__ == "__main__":
    run()
