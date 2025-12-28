import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Webcam from "react-webcam";
import axios from 'axios';
import './Login.css';
import Modal from './Modal.js';
import Title from './Title.js';
import LoadingSpinner from './LoadingSpinner';

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: 'user',
};

const Login = ({ isLoggedIn, setIsLoggedIn }) => {
  const webcamRef = useRef(null);
  const [modalText, setModalText] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const closeModal = () => {
    setShowModal(false);
  }
  const navigatePath = () => {
    (isLoggedIn ? navigate('/'): setShowModal(false));
  }

  const capture = React.useCallback(
    () => {
      const imageSrc = webcamRef.current.getScreenshot();
      setLoading(true);
      axios.post('http://localhost:8000/api/login', { image: imageSrc })
        .then(response => {
          setLoading(false);
          if (response.data.status === 'ok') {
            setModalText(`User verified! User ID: ${response.data.user_id}`);
            setShowModal(true);
            setIsLoggedIn(true);
          } else {
            setModalText(`Error: ${response.data.status}`);
            setShowModal(true);
          }
        })
        .catch(err => {
          setLoading(false);
          setModalText(`Error: ${err}`);
          setShowModal(true);
          console.error(err);
        });
    },
    [webcamRef, setIsLoggedIn]
  );

  return (
    <div className='login-container'>
      <Title title={"Login"}/>
      <div className='login-webcam'>
        <Webcam 
          audio={false}
          height={500}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={800}
          videoConstraints={videoConstraints}
        />
      </div>
      <div className='button-class'>
      <button className={`${loading ? 'disabled' : 'image-button'}`} onClick={capture} disabled={loading}>Login</button>

      </div>
      {loading && <LoadingSpinner />}
      {showModal && (
        <Modal closeModal={closeModal} modalText={modalText} navigatePath={navigatePath}/>
      )}
    </div>
  );
};

export default Login;
