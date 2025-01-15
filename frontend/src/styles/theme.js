export const theme = {
  colors: {
    primary: '#14a800',
    secondary: '#1f57c3',
    dark: '#001e00',
    gray: '#5e6d55',
    lightGray: '#e4ebe4',
    background: '#f7faf7',
    white: '#ffffff',
    error: '#e34646',
    success: '#13544e'
  },
  breakpoints: {
    xs: '320px',
    sm: '576px',
    md: '768px',
    lg: '992px',
    xl: '1200px'
  },
  typography: {
    h1: {
      fontSize: '80px',
      lineHeight: '88px',
      fontWeight: 600,
      '@media (max-width: 992px)': {
        fontSize: '56px',
        lineHeight: '64px'
      },
      '@media (max-width: 768px)': {
        fontSize: '40px',
        lineHeight: '48px'
      }
    }
  },
  shadows: {
    small: '0 2px 4px rgba(0,0,0,0.05)',
    medium: '0 4px 12px rgba(0,0,0,0.1)',
    large: '0 8px 24px rgba(0,0,0,0.15)'
  }
};