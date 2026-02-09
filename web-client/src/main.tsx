import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

const rootElement = document.querySelector('#root');
if (!rootElement) throw new Error('root element is missing');

createRoot(rootElement).render(<App />);
