# Semantic search for Halacha
![image](https://github.com/Sivan22/halacha-classification/assets/89018301/04e3eae0-c990-4795-843e-a879a4ef26b0)


this app showcases the capabilities of an AI-based approach to searching Jewish law (Halacha).

By now, it only returns results from the "orach-chaim" volume of the book "shulchan-aruch".

# חיפוש הלכתי חכם
 תוכנה זו נועדה להדגים את היכולות של גישה מבוססת מודלים של בינה מלאכותית עבור חיפוש תורני חכם. 

כרגע, התוכנה מחפשת רק בספר "שולחן ערוך " חלק "אורח חיים"

## Semantic search through text classification

Searches are based on a text-classification model, with categories being chapters and paragraphs within the book.

[link](https://huggingface.co/sivan22/halacha-siman-seif-classifier) to the moedl card on huggingFace.co.

## גישה מבוססת סיווג
החיפוש מתבצע באמצעות מודל שאומן למשימה של סיווג טקסט, הקטגוריות לסיווג הן הסימן והסעיף בו נמצא הנידון שבחיפוש.
# installation
להתקנה:

<code>
git clone https://github.com/Sivan22/halacha-classification.git
cd halacha-classification
pip install -r requirements.txt
streamlit run App.py<code/>
