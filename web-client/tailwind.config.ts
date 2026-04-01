import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#e2e8f0',
        panel: '#0f172a',
        surface: '#1e293b',
        accent: '#38bdf8',
        muted: '#334155',
      },
      fontFamily: {
        sans: ['Outfit', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'monospace'],
      },
      boxShadow: {
        panel: '0 2px 16px rgba(0, 0, 0, 0.35)',
      },
      backgroundImage: {
        page: 'linear-gradient(145deg, #020617 0%, #0a1128 50%, #020617 100%)',
      },
    },
  },
  plugins: [],
} satisfies Config;
