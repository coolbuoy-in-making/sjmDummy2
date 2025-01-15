import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import Card from '../shared/Card';
import { auth } from '../../utils/api';

const Form = styled.form`
  max-width: 400px;
  margin: 40px auto;
  padding: 32px;
`;

const Input = styled.input`
  width: 100%;
  padding: 12px;
  margin-bottom: 16px;
  border: 1px solid ${props => props.theme.colors.lightGray};
  border-radius: 8px;
  font-size: 16px;

  &:focus {
    border-color: ${props => props.theme.colors.primary};
    outline: none;
  }
`;

const ErrorMessage = styled.p`
  color: ${props => props.theme.colors.error};
  margin-bottom: 16px;
  font-size: 14px;
`;

const LoginForm = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const result = await auth.login(formData);
    
    if (result.success) {
      navigate('/find-work');
    } else {
      setError(result.error);
    }
    
    setLoading(false);
  };

  return (
    <Card>
      <Form onSubmit={handleSubmit}>
        <h2>Log in to Upwork</h2>
        {error && <ErrorMessage>{error}</ErrorMessage>}
        <Input 
          type="email" 
          placeholder="Email or Username"
          value={formData.email}
          onChange={e => setFormData({...formData, email: e.target.value})}
          required
        />
        <Input 
          type="password"
          placeholder="Password"
          value={formData.password}
          onChange={e => setFormData({...formData, password: e.target.value})}
          required
        />
        <button 
          className="primary-button" 
          style={{width: '100%'}}
          disabled={loading}
        >
          {loading ? 'Logging in...' : 'Log In'}
        </button>
        <p style={{marginTop: '16px', textAlign: 'center'}}>
          Don't have an account? <Link to="/signup">Sign Up</Link>
        </p>
      </Form>
    </Card>
  );
};

export default LoginForm;