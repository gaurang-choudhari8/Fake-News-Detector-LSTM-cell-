from flask import Flask, escape, request, render_template
import pickle
import nltk
import re
from nltk.corpus import stopwords
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import one_hot



nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]



voc_size=10000









import numpy as np



model = pickle.load(open("fin_model.pkl", 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods=['GET','POST'])
def prediction():
    if request.method == "POST":
       news = str(request.form['news'])
       


       review=re.sub('[^a-zA-Z]',' ',news)
       review=review.lower()
       review=review.split()
       review=[ps.stem(word) for word in review]
       review=' '.join(review)
       corpus.append(review)
       onehot_repr=[one_hot(words,voc_size)for words in corpus]
       sent_length=20
       embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
       X_final=np.array(embedded_docs)
       y_pred = model.predict(X_final)
       y_pred=np.round(y_pred).astype(int)

       return render_template("prediction.html", prediction_text="News is -> {}".format(y_pred))
        
    else:
        return render_template("prediction.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run()    