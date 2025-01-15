import { createGlobalStyle } from 'styled-components';

export const GlobalStyles = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html {
    font-size: 16px;
    
    @media (max-width: ${props => props.theme.breakpoints.md}) {
      font-size: 14px;
    }
  }

  body {
    font-family: 'Neue Montreal', Inter, system-ui, sans-serif;
    color: ${props => props.theme.colors.dark};
    background: #ffffff;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }

  .container {
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 24px;

    @media (max-width: ${props => props.theme.breakpoints.sm}) {
      padding: 0 16px;
    }
  }

  .section {
    padding: 80px 0;
    
    @media (max-width: ${props => props.theme.breakpoints.md}) {
      padding: 40px 0;
    }
  }

  h1, h2, h3, h4, h5, h6 {
    margin: 0;
    font-weight: 600;
  }

  h1 {
    font-size: 3rem;
    line-height: 1.2;
    
    @media (max-width: ${props => props.theme.breakpoints.md}) {
      font-size: 2.5rem;
    }
  }

  h2 {
    font-size: 2.5rem;
    line-height: 1.3;
    
    @media (max-width: ${props => props.theme.breakpoints.md}) {
      font-size: 2rem;
    }
  }

  img {
    max-width: 100%;
    height: auto;
  }

  button {
    cursor: pointer;
    font-family: inherit;
  }

  .primary-button {
    background: ${props => props.theme.colors.primary};
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 24px;
    font-weight: 500;
    font-size: 1rem;
    transition: all 0.2s;
    
    &:hover {
      background: ${props => props.theme.colors.success};
    }

    @media (max-width: ${props => props.theme.breakpoints.sm}) {
      width: 100%;
      text-align: center;
    }
  }

  .secondary-button {
    background: transparent;
    color: ${props => props.theme.colors.dark};
    border: 1px solid currentColor;
    padding: 12px 24px;
    border-radius: 24px;
    font-weight: 500;
    font-size: 1rem;
    transition: all 0.2s;
    
    &:hover {
      background: ${props => props.theme.colors.lightGray};
    }

    @media (max-width: ${props => props.theme.breakpoints.sm}) {
      width: 100%;
      text-align: center;
    }
  }

  .grid {
    display: grid;
    gap: 24px;
    
    &.cols-2 {
      grid-template-columns: repeat(2, 1fr);
      
      @media (max-width: ${props => props.theme.breakpoints.md}) {
        grid-template-columns: 1fr;
      }
    }
    
    &.cols-3 {
      grid-template-columns: repeat(3, 1fr);
      
      @media (max-width: ${props => props.theme.breakpoints.lg}) {
        grid-template-columns: repeat(2, 1fr);
      }
      
      @media (max-width: ${props => props.theme.breakpoints.sm}) {
        grid-template-columns: 1fr;
      }
    }
  }
`;