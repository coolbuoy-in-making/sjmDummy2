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

const Select = styled.select`
  width: 100%;
  padding: 12px;
  margin-bottom: 16px;
  border: 1px solid ${props => props.theme.colors.lightGray};
  border-radius: 8px;
  font-size: 16px;
  background: white;
`;

const ErrorMessage = styled.p`
  color: ${props => props.theme.colors.error};
  margin-bottom: 16px;
  font-size: 14px;
`;

const SignupForm = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    userType: 'client'
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const result = await auth.register(formData);
    
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
        <h2>Create an Account</h2>
        {error && <ErrorMessage>{error}</ErrorMessage>}
        <Input 
          type="text" 
          placeholder="Full Name"
          value={formData.name}
          onChange={e => setFormData({...formData, name: e.target.value})}
          required
        />
        <Input 
          type="email" 
          placeholder="Email"
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
        <Select
          value={formData.userType}
          onChange={e => setFormData({...formData, userType: e.target.value})}
        >
          <option value="client">I want to hire</option>
          <option value="freelancer">I want to work</option>
        </Select>
        <button 
          className="primary-button" 
          style={{width: '100%'}}
          disabled={loading}
        >
          {loading ? 'Creating Account...' : 'Create Account'}
        </button>
        <p style={{marginTop: '16px', textAlign: 'center'}}>
          Already have an account? <Link to="/login">Log In</Link>
        </p>
      </Form>
    </Card>
  );
};

export default SignupForm;