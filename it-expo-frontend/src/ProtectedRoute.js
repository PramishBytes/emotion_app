import React from 'react';
import { Navigate } from 'react-router-dom';

export const ProtectedRoute = ({ children, isLoggedIn }) => {
  return isLoggedIn ? children : <Navigate to="/login" replace />;
};
