import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import { UserProvider } from './contexts/UserContext';
import { theme } from './styles/theme';
import Navbar from './components/Navbar/Navbar';
import Footer from './components/shared/Footer';
import AppRoutes from './routes';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <UserProvider>
          <div className="app">
            <Navbar />
            <main className="main-content">
              <AppRoutes />
            </main>
            <Footer />
          </div>
        </UserProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;