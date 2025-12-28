import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import io from 'socket.io-client';
import './WebCam.css';
import Title from './Title.js';

const videoConstraints = {
  width: 1200,
  height: 720,
  facingMode: 'user',
};

const WebCam = () => {
  const webcamRef = useRef(null);
  const socketRef = useRef();
  const [emotion, setEmotion] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');

  useEffect(() => {
    socketRef.current = io.connect('http://localhost:8000');

    socketRef.current.on('data', data => {
      setEmotion(data.emotion);
      setGender(data.gender);
      setAge(data.age);
    });

    return () => {
      socketRef.current.disconnect();
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (webcamRef.current) {
        const image = webcamRef.current.getScreenshot();
        if (image) {
          socketRef.current.emit('frame', { image });
        }
      }
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="webcam-container">
      <Title title={"Detect with Video"} set={false}/>
      <div className="webcam">
          <Webcam
            audio={false}
            height={450}
            width={800}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
          />
      </div>
      <div className="emotion">
        <h2>Emotion</h2>
        <p>{emotion}</p>
        <br></br>
        <h2>Age</h2>
        <p>{age}</p>
        <br></br>
        <h2>Gender</h2>
        <p>{gender}</p>
      </div>
    </div>
  );
};

export default WebCam;