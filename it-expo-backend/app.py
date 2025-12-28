from scipy.spatial import distance
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from deepface import DeepFace
from flask_cors import CORS
from deepface.commons import functions
import sqlite3
import uuid


def create_database():
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings(
            user_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


def store_embeddings(embedding):
    # Generate a UUID for the user
    user_id = str(uuid.uuid4())
    
    # Convert embedding array to a string
    embedding_bytes = embedding.tobytes()
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()

    # Insert embedding into the database
    c.execute('''
        INSERT INTO embeddings VALUES (?, ?)
    ''', (user_id, embedding_bytes))

    conn.commit()
    conn.close()

    # Return the user_id so it can be used in the application
    return user_id

def get_closest_embedding(embedding):
    # Connect to your SQLite database
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()

    # Retrieve all user_id and embedding rows from the database
    cursor.execute('SELECT user_id, embedding FROM embeddings')
    rows = cursor.fetchall()

    closest_user_id = None
    min_distance = float('inf')

    # Iterate over the retrieved rows and calculate the Euclidean distance
    for row in rows:
        db_user_id = row[0]
        db_embedding = np.frombuffer(row[1], dtype=np.float32)

        # Calculate the Euclidean distance between the embeddings
        dist = distance.euclidean(embedding, db_embedding)

        # Update the closest embedding if a closer match is found
        if dist < min_distance:
            min_distance = dist
            closest_user_id = db_user_id

    # Close the database connection
    conn.close()

    return closest_user_id, min_distance


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/api/signup', methods=['POST'])
def handle_signup():
    data = request.get_json()
    image = data['image']
    image = base64.b64decode(image.split(',')[1])
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, flags=1)

    try:
        # Try to detect face
        faces = DeepFace.extract_faces(image, detector_backend = 'opencv')
        if faces is not None and len(faces) > 0:  
            model = DeepFace.build_model('VGG-Face')

            face_img = cv2.resize(faces[0]['face'], (224, 224))
            face_img = np.expand_dims(face_img, axis=0)  # Model expects a batch of images as input
            face_img = face_img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

            embedding = model.predict(face_img)[0]
            user_id = store_embeddings(embedding)  
            return jsonify({'status': 'ok', 'user_id':user_id})
        elif len(faces)>1:
            return jsonify({'status': 'Only one person in frame'})
        else:
            return jsonify({'status': 'No face detected'})
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'status': str(e).split('.')[0]})



@app.route('/api/login', methods=['POST'])
def handle_login():
    data = request.get_json()
    image = data['image']
    image = base64.b64decode(image.split(',')[1])
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, flags=1)

    try:
        # Try to detect face
        faces = DeepFace.extract_faces(image, detector_backend = 'opencv')
        if faces is not None and len(faces) > 0:  
            # If face detected, find embeddings
            model = DeepFace.build_model('VGG-Face')

            # Resize and normalize the face image
            face_img = cv2.resize(faces[0]['face'], (224, 224))
            face_img = np.expand_dims(face_img, axis=0)
            face_img = face_img.astype('float32') / 255.0

            embedding = model.predict(face_img)[0]

            closest_user_id, min_distance = get_closest_embedding(embedding)

            threshold = 0.4  # This value can be adjusted based on your application

            if min_distance < threshold:
                # If the minimum distance is below the threshold, the user is considered recognised
                return jsonify({'status': 'ok', 'user_id': closest_user_id})
            elif len(faces)>1:
                return jsonify({'status': 'Only one person in frame'})
            else:   
                # If the minimum distance is above the threshold, the user is not recognised
                return jsonify({'status': 'User not recognised'})
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'status': str(e).split('.')[0]})


@socketio.on('frame')
def handle_frame(data):
    image = data['image']
    image = base64.b64decode(image.split(',')[1])
    image = np.frombuffer(image, dtype=np.uint8) 
    image = cv2.imdecode(image, flags=1)
    # Convert to grayscale for dlib

    # Use DeepFace to predict the emotion
    age="Young wild and free"
    gender="Classified information"
    dominant_emotion = "No faces detected"
  
    try:
        faces = DeepFace.extract_faces(image, detector_backend = 'opencv', enforce_detection=False)
        if faces[0]['confidence']<3:    
            raise Exception("No faces detected") 
        if len(faces)>1:
            dominant_emotion = "Many detected"
            raise Exception("Many detected") 
            
        result = DeepFace.analyze(image, actions=['emotion','age','gender'], enforce_detection=True)
        dominant_emotion=result[0]['dominant_emotion']
        diff=3
        print(result[0]['gender']['Man'])
        print(result[0]['gender']['Woman'])
        gender= "Female" if  (result[0]['gender']['Woman']*1.2 > 5) else "Male"
        if gender == "Female":
            diff=10
        age=result[0]['age']-diff

    except Exception as e:
        print(f'Error: {e}')

    socketio.emit('data', {
    'emotion': dominant_emotion,
    'age': age,
    'gender':gender,
    })
    
@app.route('/api/image', methods=['POST'])
def handle_image():
    data = request.get_json()
    age = "Young "
    gender = "Classified"
    dominant_emotion = "No faces detected"

    try:
        image = data['image']
        image = base64.b64decode(image.split(',')[1])
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, flags=1)
        faces = DeepFace.extract_faces(image, detector_backend = 'opencv', enforce_detection=False)
        if faces[0]['confidence']<3:    
            raise Exception("No faces detected") 
        if len(faces)>1:
            dominant_emotion = "Many detected"
            raise Exception("Many detected") 
            
        result = DeepFace.analyze(image, actions=['emotion','age','gender'], enforce_detection=True)
        dominant_emotion=result[0]['dominant_emotion']
        diff=3
        print(result[0]['gender']['Man'])
        print(result[0]['gender']['Woman'])
        gender= "Female" if  (result[0]['gender']['Woman']*1.2 > 5) else "Male"
        if gender == "Female":
            diff=10
        age=result[0]['age']-diff


    except Exception as e:
        print(f'Error: {e}')

    return {
        'emotion': dominant_emotion,
        'age': age,
        'gender': gender,
    }, 200
    
if __name__ == '__main__':
    create_database()
    print("Running on port: http://localhost:8000")
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)
