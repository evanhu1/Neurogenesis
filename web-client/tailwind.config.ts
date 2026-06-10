import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#262b33',
        panel: '#fdfcf8',
        surface: '#f0ecdf',
        accent: '#15803d',
        muted: '#ded7c6',
        void: '#f6f3ea',
        line: 'rgba(62, 55, 38, 0.13)',
      },
      fontFamily: {
        sans: ['Outfit', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'monospace'],
      },
      boxShadow: {
        panel: '0 8px 24px rgba(94, 82, 50, 0.12)',
      },
      backgroundImage: {
        page: 'linear-gradient(165deg, #eee8d9 0%, #f5f1e6 50%, #e8ebd9 100%)',
        water: 'linear-gradient(180deg, #cce0ea 0%, #b4d4e2 55%, #aacddc 100%)',
      },
    },
  },
  plugins: [],
} satisfies Config;
