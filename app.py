import string
from flask import Flask
from flask import request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
app = Flask(__name__)

cors = CORS(app, resources={"/predctLearningStyle": {"origins": "*"}})


@app.route('/predctLearningStyle', methods=['GET', 'POST'])
def predictLearningStyle():
    if request.method == 'POST':
        print(request)
        pickled_model = pickle.load(open('model.pkl', 'rb'))
        input_message = []
        json_data = request.get_json()
        for key in json_data:
            message = [json_data[key]]
            input_message.append(message)
        dataframe = pd.read_csv('Questionnaire LS.csv')
        df = pd.DataFrame(input_message, columns=["Test_Message"])
        bow_transformer = CountVectorizer(
            analyzer=text_process).fit(dataframe['Response'])
        messages_bow = bow_transformer.transform(df['Test_Message'])
        tfidf_transformer = TfidfTransformer().fit(messages_bow)
        messages_tfidf = tfidf_transformer.transform(messages_bow)
        result = pickled_model.predict(messages_tfidf)
        resultarr = np.array(result)
        visual = 0
        Kinaesthetic = 0
        read = 0
        aural = 0
        learningType = "None"
        for x in resultarr:
            if x == 'Visual':
                visual += 1
            elif x == 'Kinaesthetic':
                Kinaesthetic += 1
            elif x == 'Aural':
                aural += 1
            elif x == 'Read/Write':
                read += 1
        if (visual > Kinaesthetic) and (visual > read) and (visual > aural):
            learningType = "VISUAL"
        elif (Kinaesthetic > read) and (Kinaesthetic > aural):
            learningType = "KINAESTHETIC"
        elif (read > aural):
            learningType = "READ/WRITE"
        else:
            learningType = "AURAL"
        print(visual)
        print(read)
        print(aural)
        print(Kinaesthetic)
        data = {
            "LearningType": learningType
        }
        print(learningType)
        return data


@app.route('/test')
def test():
    return "Hello World!!!"


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=6000, debug=True)
    app.run(debug=True)
