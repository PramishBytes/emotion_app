import React from 'react';
import './Modal.css';

const Modal = ({ closeModal, modalText, navigatePath }) => {
  return(
    <div className="modal-app">
    <div className="modal-content">
      <span className="close" onClick={closeModal}>&times;</span>
      <p>{modalText}</p>
      <button className='ok-button' onClick={navigatePath}>OK</button>
    </div>
  </div>
  );
};

export default Modal;
