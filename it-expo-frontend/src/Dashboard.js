import React from 'react'
import { Link } from 'react-router-dom';
import Title from './Title.js';
import './Dashboard.css'

const Dashboard = () => {
  return (
      <div className="buttons">
        <Title title={"Dashboard"} set={false}/>
        <Link to="/detect-with-video" className="button">Detect with Video</Link>
        <Link to="/detect-with-photo" className="button">Detect with Photo</Link>
      </div>
  )
}

export default Dashboard