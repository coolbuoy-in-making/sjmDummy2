import { Routes, Route } from 'react-router-dom';
import Home from '../pages/Home';
import FindWork from '../pages/FindWork';
import FindTalent from '../pages/FindTalent';
import WhyUpwork from '../components/WhyUpwork/WhyUpwork';
import Enterprise from '../components/Enterprise/Enterprise';
import Login from '../pages/Login';
import Signup from '../pages/Signup';
import Profile from '../pages/Profile';

const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/find-work" element={<FindWork />} />
      <Route path="/find-talent" element={<FindTalent />} />
      <Route path="/why-upwork" element={<WhyUpwork />} />
      <Route path="/enterprise" element={<Enterprise />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/profile/:id" element={<Profile />} />
    </Routes>
  );
};

export default AppRoutes;
