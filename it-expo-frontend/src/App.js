import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import WebCam from './WebCam';
import Navbar from './Navbar';
import Image from './Image';
import Dashboard from './Dashboard';
import SignUp from './SignUp';
import Login from './Login';
import Logout from './Logout';
import Guide from './Guide';
import { ProtectedRoute } from './ProtectedRoute'; 
import './App.css';
import Vote from './Vote';


const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  
  return (
    <div className='main'>
      <Vote/>
    <Router>
      <Navbar isLoggedIn ={isLoggedIn}/>
        <Routes>
          <Route path="/" element={<ProtectedRoute isLoggedIn={isLoggedIn}><Dashboard /></ProtectedRoute>} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/guide" element={<Guide />} />
          <Route path="/login" element={<Login isLoggedIn={isLoggedIn} setIsLoggedIn={setIsLoggedIn} />} />
          <Route path="/detect-with-video" element={<ProtectedRoute isLoggedIn={isLoggedIn}><WebCam /></ProtectedRoute>} />
          <Route path="/detect-with-photo" element={<ProtectedRoute isLoggedIn={isLoggedIn}><Image /></ProtectedRoute>} />
          <Route path="/logout" element={<Logout setIsLoggedIn={setIsLoggedIn} />} />
        </Routes>
    </Router>
    </div>
  );
};

export default App;
