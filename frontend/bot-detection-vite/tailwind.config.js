/** @type {import('tailwindcss').Config} */
export default {
    content: [
      "./index.html",
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
      extend: {
        colors: {
          // Add any custom colors if needed
        },
        backdropBlur: {
          xs: '2px',
        },
      },
    },
    plugins: [],
  }