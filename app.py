from flask import Flask, render_template, request, send_from_directory      #module of python
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import requests
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


face_detection = cv2.CascadeClassifier('static/haarcascade_files/haar_cascade_face_detection.xml')

emotion_model_path = 'static/expression_1.model'
model=   tf.keras.models.load_model(emotion_model_path, compile=False)

labels = ["Neutral","Happy","Sad","Surprise","Angry"]
def detect_emotion(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detection.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        for x, y, w, h in detected_faces:
    
            cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
            cv2.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)

            
            face = gray[y+5:y+h-5, x+20:x+w-20]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0

            
            predictions = model.predict(np.array([face.reshape((48, 48, 1))])).argmax()
            emotion = labels[predictions]

            
            movie_recommendations = recommend_movies(emotion)
            return emotion, movie_recommendations

    return None, None


#@app.route('/recommend_movies/<title>')
def recommend_movies(emotion):
    api_key = '9f7d7fba833ab36a9e0ed88e2ee2c7ca' 
    base_url = 'https://api.themoviedb.org/3/discover/movie'

    genre_mapping = {
        "Sad": 18,       # Drama
        "Angry": 28,     # Action
        "Neutral": 53,   # Thriller
        "Happy": 35,     # Comedy
        "Surprise": 9648  # Mystery
    }

    genre_id = genre_mapping.get(emotion, 18)  

    params = {
        'api_key': api_key,
        'language': 'en-US',
        'sort_by': 'popularity.desc',
        'include_adult': 'false',
        'include_video': 'false',
        'with_genres': genre_id
    }

    response = requests.get(base_url, params=params)
    data = response.json()                  #data->dictionary 
    #print(data)
 
    movies = []
    for movie in data.get('results', []):     #result->key in data 
        title = movie.get('title', '')
        rating = movie.get('vote_average', 0.0)
        poster_path = movie.get('poster_path', '')
        id = movie.get('id', '')
        
        movies.append ({
            'title': title,
            'rating': rating,
            'poster_path': f"https://image.tmdb.org/t/p/w500/{poster_path}",
            'id':id
        })
        

    if not movies:
        print(f"No movie recommendations for {emotion} emotion.")

    return movies



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'myFile' not in request.files:
        return "No file part"

    file = request.files['myFile']

    if file.filename == '':
        return "No selected file"

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    
    detected_emotion, movie_recommendations = detect_emotion(filename)

    if detected_emotion:
        
        return render_template('result.html', image_file=file, detected_emotion=detected_emotion, movie_recommendations=movie_recommendations)
    else:
        return "Error detecting emotion"



# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
