import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import './Image.css';
import LoadingSpinner from './LoadingSpinner';
import Title from './Title.js';


const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: 'user',
};

const Image = () => {
  const webcamRef = useRef(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isTakingPhoto, setIsTakingPhoto] = useState(false);
  const [emotion, setEmotion] = useState(null);
  const [age, setAge] = useState(null);
  const [gender, setGender] = useState(null);
  const [loading, setLoading] = useState(false);


  const handleDetectClick = () => {
    setLoading(true);
    axios.post('http://localhost:8000/api/image', { image: selectedImage })
    .then(response => {
        setLoading(false);
        setEmotion(response.data.emotion);
        setAge(response.data.age);
        setGender(response.data.gender);
      })
      .catch(error => {
        setLoading(false);
        console.error('Error:', error);
      });
  };
  const handleImageUpload = (event) => {
    setIsTakingPhoto(false);
    const reader = new FileReader();
    reader.onloadend = () => {
      setSelectedImage(reader.result);
    };
    reader.readAsDataURL(event.target.files[0]);
  };

  const handleTakePhotoClick = () => {
    setIsTakingPhoto(true);
    setSelectedImage(null);
  };

  const handleCaptureClick = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setSelectedImage(imageSrc);
    setIsTakingPhoto(false);
  };

  return (
    <div className='image-container'>
        <Title title={"Detect with Image"} set={false}/>
      <div className='image'>
        <div className='image-preview'>
          {selectedImage ? (
            <img src={selectedImage} alt="Selected" />
            ) : (null)}
        </div>
        {isTakingPhoto ? (
          <div className='webcam'>
            <Webcam
              audio={false}
              height={450}
              screenshotFormat="image/jpeg"
              width={800}
              videoConstraints={videoConstraints}
              ref={webcamRef}
              />
          </div>
        ) : null}
          
      {!isTakingPhoto ? (
      <div className='image-buttons'>
        <label className='image-button' htmlFor='upload-button'>
            Upload photo
            <input 
              id='upload-button' 
              type='file' 
              accept='image/*' 
              onChange={handleImageUpload} 
              style={{display: 'none'}} // hide the input
            />
          </label>
          <button className='image-button' onClick={handleTakePhotoClick}>Take Photo</button>
        </div>):(
        <div className='image-buttons'>
          <button className='image-button' onClick={handleCaptureClick}>Capture</button>
        </div>)
          }
      </div>
      <div className="emotion">
        <h2>Emotion</h2>
        <p>{emotion}</p>
        <h2>Age</h2>
        <p>{age}</p>
        <h2>Gender</h2>
        <p>{gender}</p>
        <button className='button' onClick={handleDetectClick}>Detect Emotion</button>
      </div>
      {loading && <LoadingSpinner />}
    </div>
  );
};

export default Image;
