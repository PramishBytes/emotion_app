import { Link } from 'react-router-dom';
import './Navbar.css'; // Assume you have a CSS file for styling the Navbar

const Navbar = ({ isLoggedIn }) => {
  return (
    <nav>
      {!isLoggedIn && (
        <>
          <Link to="/login">Login</Link>
          <Link to="/signup">Sign Up</Link>
          <Link to="/guide">Guide</Link>
        </>
      )}
      {isLoggedIn && (
        <>
          <Link to="/">Dashboard</Link>
          <Link to="/detect-with-video">Video</Link>
          <Link to="/detect-with-photo">Image</Link>
          <Link to="/guide">Guide</Link>
          <Link to="/logout">Logout</Link>
        </>
      )}
    </nav>
  );
};

export default Navbar;
