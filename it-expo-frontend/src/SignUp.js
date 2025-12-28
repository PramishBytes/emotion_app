import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './SignUp.css';
import Modal from './Modal.js';
import LoadingSpinner from './LoadingSpinner';
import Title from './Title.js';


const videoConstraints = {
  width: 1200,
  height: 720,
  facingMode: 'user',
};

const SignUp = () => {
  const webcamRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const navigate = useNavigate();
  const [isTakingPhoto, setIsTakingPhoto] = useState(false);
  const [signupState, setsignupState] = useState('Sign up');
  const [modalText, setModalText] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [signedUp, setSignedUp] = useState(false);
  const [loading, setLoading] = useState(false);


  const closeModal = () => {
    setShowModal(false);
  }
  
  const navigatePath = () => {
    (signedUp ? navigate('/login'):closeModal());
  }
  const handleCaptureClick = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImage(imageSrc);
    setsignupState('Take again')
    setIsTakingPhoto(false);
  };


  const handleTakePhotoClick = () => {
    setIsTakingPhoto(true);
    setCapturedImage(null);
  };

  const handleSignUp = async () => {
    try {
      setLoading(true);
      const response = await axios.post('http://localhost:8000/api/signup', { image: capturedImage });
      if(response.data.status === 'ok') {
        setModalText(`User Created! User ID: ${response.data.user_id}`);
        setShowModal(true);
        setSignedUp(true);
      } else {
        setLoading(false);
        setModalText(`Error: ${response.data.status}`);
        setShowModal(true);
      }
    } catch(error) {
      setLoading(false);
      setModalText(`Error: ${TypeError}`);
      setShowModal(true);
      console.error(TypeError);
    }
  }

return (
    <div className='signup'>
      <Title title={"Signup"}/>
      <div className='signup-container'>
        <div className='capture'>
            {isTakingPhoto ? (
              <Webcam
              audio={false}
              height={450}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={800}
              videoConstraints={videoConstraints}
              />
            ) : null}
        </div>
        {capturedImage && (
            <div className='preview'>
            <img src={capturedImage} alt="preview" />
            </div>
        )}

        {!isTakingPhoto ? (
          <div className='signup-buttons'>
              <button className='image-button' onClick={handleTakePhotoClick}>{signupState}</button>
              {capturedImage? (<button className='image-button' onClick={handleSignUp}>Sign Up</button>):(null)}
            </div>):(
              <div className='signup-buttons'>
                <button className='image-button' onClick={handleCaptureClick}>Capture</button>
            </div>)
        }
        </div>
      {loading && <LoadingSpinner />}
      {showModal && (
        <Modal closeModal={closeModal} modalText={modalText} navigatePath={navigatePath}/>
      )}
    </div>
  );
}

export default SignUp;
