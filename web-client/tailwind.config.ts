import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#e7ecf5',
        panel: '#0c1322',
        surface: '#16203a',
        accent: '#38bdf8',
        muted: '#2b3854',
        void: '#080e1a',
      },
      fontFamily: {
        sans: ['Outfit', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'monospace'],
      },
      boxShadow: {
        panel: '0 8px 28px rgba(2, 6, 23, 0.55)',
      },
      backgroundImage: {
        page: 'radial-gradient(1200px 800px at 70% -10%, #0d1830 0%, #060b16 55%, #04070f 100%)',
      },
    },
  },
  plugins: [],
} satisfies Config;
