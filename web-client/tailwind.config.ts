import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#1b2638',
        panel: '#ffffff',
        accent: '#1167b1',
        gridA: '#d7dde8',
        gridB: '#e3e8f0',
      },
      fontFamily: {
        sans: ['Space Grotesk', 'Avenir Next', 'Segoe UI', 'sans-serif'],
        mono: ['IBM Plex Mono', 'SF Mono', 'monospace'],
      },
      boxShadow: {
        panel: '0 14px 30px rgba(15, 33, 63, 0.12)',
      },
      backgroundImage: {
        page:
          'radial-gradient(circle at 14% -12%, #dce8fb 0, #f1f5fb 35%, #dfe8f7 100%)',
      },
    },
  },
  plugins: [],
} satisfies Config;
