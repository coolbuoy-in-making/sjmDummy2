import Navbar from './components/Navbar/Navbar';
import Footer from './components/shared/Footer';
import AppRoutes from './routes';
import './App.css';

function App() {
  return (
    <div className="app">
      <Navbar />
      <main className="main-content">
        <AppRoutes />
      </main>
      <Footer />
    </div>
  );
}

export default App;
