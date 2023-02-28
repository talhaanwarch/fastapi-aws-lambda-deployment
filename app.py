import pickle
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum

app=FastAPI()
handler=Mangum(app)

# Load the saved model
clf_path =  "models/logisticregression.pkl"
with open(clf_path, 'rb') as file:
    classifier = pickle.load(file)

# Load the saved vectorizer
vect_path =  "models/countvectorizer.pkl"
with open(vect_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Load the saved encoder
enc_path =  "models/labelencoder.pkl"
with open(enc_path, 'rb') as file:
    encoder = pickle.load(file)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
# if not os.path.exists('nltkdata'):
#   os.mkdir('nltkdata')
#   nltk.download('wordnet',download_dir='nltkdata')
#   nltk.download('stopwords',download_dir='nltkdata')
#   nltk.download('omw-1.4',download_dir='nltkdata')

nltk.data.path.append("nltkdata")

lemmatizer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
stop = stopwords.words('english')

import re
import string
def preprocessing(text):
  #apply preprocessing
  text=text.lower()
  text=re.sub('\d+', '',text)
  text=re.sub('@user', '',text)
  text=re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
          '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',text)
  
  text=text.translate(str.maketrans('', '', string.punctuation))
  text=re.sub('[^\w\s]', '',text)

  text=[lemmatizer.lemmatize(y) for y in w_tokenizer.tokenize(text)]
  text=[item for item in text if item not in stop]
  text= " ".join(text)
  return text

def predict_result(text):
  text=preprocessing(text)
  vect=vectorizer.transform([text])
  pred=classifier.predict(vect)
  result=encoder.inverse_transform(pred)
  return result[0]

@app.get('/')
def my_function(text:str):
  pred=predict_result(text)
  return JSONResponse({"prediction":pred})

if __name__=="__main__":
  uvicorn.run(app,host="0.0.0.0",port=9000)

