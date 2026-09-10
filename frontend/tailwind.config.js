/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef7f6", 100: "#d4ebe9", 200: "#a9d7d3", 300: "#74bcb6",
          400: "#469b96", 500: "#2c807b", 600: "#226663", 700: "#1e5250",
          800: "#1b4241", 900: "#193837",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
      },
    },
  },
  plugins: [],
};
