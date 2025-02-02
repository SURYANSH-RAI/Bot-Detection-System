#  Bot Profile Detection on Social Media


# Bot Detection System

A modern web application built with React and Vite for detecting bot-like behavior in social media data. The system provides both machine learning and rule-based approaches for analysis, with an intuitive user interface for data input and visualization.

## Features

- Upload and analyze CSV files containing social media data
- Process social media post links and profile links
- Two analysis methods:
  - Machine Learning-based detection
  - Rule-based pattern detection
- Real-time analysis with detailed results
- Modern, responsive UI with dark mode
- Interactive data visualization
- Backend API integration ready

## Prerequisites

Before you begin, ensure you have installed:
- Node.js (version 18.18.0 or higher)
- npm (version 8.0.0 or higher)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SURYANSH-RAI/Bot-Detection-System
cd Bot-Detection-System
```

2. Install dependencies:
```bash
npm install
```

This will install all required dependencies including:
- React (v18.3.1)
- React DOM (v18.3.1)
- Vite (v6.0.5)
- Tailwind CSS (v4.0.3)
- Lucide React (v0.474.0)
- Papa Parse (v5.5.2)

## Development Dependencies

The project uses several development dependencies:
- ESLint and related plugins
- TypeScript types for React
- Autoprefixer
- PostCSS
- Various Babel plugins

These are all included in the package.json and will be installed with the npm install command.

## Running the Application

1. Start the development server:
```bash
npm run dev
```

2. Build for production:
```bash
npm run build
```

3. Preview production build:
```bash
npm run preview
```

## Backend API Integration

The application is configured to connect to a backend server running on `http://localhost:8000`. The backend should provide the following endpoints:

- `/api/ml-analysis` - For machine learning-based analysis
- `/api/rule-based-analysis` - For rule-based analysis

Both endpoints expect POST requests with JSON data containing:
- csvData: Parsed CSV data
- socialLinks: Array of social media post links
- profileLinks: Array of profile links

## Styling

The project uses Tailwind CSS for styling with a custom configuration. The main styling files are:
- `index.css` - Main CSS file with Tailwind imports
- `tailwind.config.js` - Tailwind configuration
- `App.css` - Additional custom styles


## Configuration Files

The project includes several configuration files:
- `vite.config.js` - Vite bundler configuration
- `eslint.config.js` - ESLint configuration
- `tailwind.config.js` - Tailwind CSS configuration
- `package.json` - Project metadata and dependencies

## Environment Variables

No environment variables are required for basic functionality, but you may want to add:
- `VITE_API_URL` - Backend API URL (defaults to http://localhost:8000)


## How to use:
1. Initialize the project.
2. Choose what type of input has to be given:-
   - Link of social media profile
   - Link of specific social media post
   - Upload a set of inputs of social media profile or post in csv format
3. Select the analysis mode:-
   - ML model based
   - Rule Based
4. Click on Analyze
5. wait for results to appear. (Result will be shown just below the analyze button)

## Demo Video link available on google drive:
https://drive.google.com/drive/folders/1kTz6yQ6BAZMxV1cPD5muN6rzj9GLHyaB?usp=sharing

##Source Code:-
https://github.com/SURYANSH-RAI/Bot-Detection

## License
This project is licensed under the MIT License - see the LICENSE file for details.
