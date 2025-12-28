import React from 'react'
import { Helmet } from "react-helmet";


const Title = ({title, set=true}) => {
  return (
    <Helmet>
      <title>{(set ? "Emotion Detector - "+title : title)}</title>
    </Helmet>
  )
}

export default Title