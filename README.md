# 🥗 Aafiya AI - Your AI-Powered Wellness Companion

Aafiya AI is a comprehensive nutrition coaching application that provides personalized nutrition guidance, meal analysis, and wellness support. "Aafiya" means "health" and "wellness" in Arabic.

## ✨ Features

### 🤖 AI Nutrition Experts
- Multiple AI personas with different nutrition specialties, Personalized nutrition guidance based on user's health goals and dietary needs

- Friendly Nutritionist — A warm, approachable guide who offers general healthy eating advice in a conversational tone.

- Strict Coach — A disciplined advisor focused on structured meal plans and performance-driven nutrition.

- Fun Chef — A creative culinary expert who suggests playful, flavor-packed recipes to keep you excited about healthy food.

- Mindful Coach — A wellness-focused guide who emphasizes mindful eating habits and long-term lifestyle balance.

### 📸 Food Image Analysis
- Upload food images for instant nutrition analysis
- Approximate calorie, protein, carbs, and fat calculations
- Visual meal assessment and recommendations

### 🧮 Nutrition Calculator
- Real-time nutrition data calculation
- Integration with Edamam API for accurate nutrition information
- Comprehensive food database with 35+ common foods

### 📝 Meal Logging
- Automatic meal tracking based on user input
- Detailed nutrition history with timestamps
- AI-powered meal recommendations and advice

### 🤖 Interactive AI Avatar
- Salma - your interactive AI nutritionist
- HeyGen avatar integration for speaking responses
- Visual and voice-based nutrition coaching

### 🍩 CraveSmart
- Transform unhealthy cravings into healthy alternatives
- Smart substitution recommendations
- Craving pattern analysis

### 📄 Document Upload (RAG)
- Upload nutrition documents, InBody results, medical reports
- PDF, TXT, and CSV file support
- AI queries based on your personal health data

### 👤 User Profile Management
- Comprehensive health profile setup
- Allergy and dietary restriction tracking
- BMI calculation and health metrics
- Personalized nutrition goals

### 🎨 Dual Theme Support
- Light mode (Ivory theme)
- Dark mode with proper contrast
- Seamless theme switching

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Streamlit
- Google Gemini API key
- Edamam API credentials 


### Installation

1. Clone the repository:
```bash
git clone https://github.com/Nourah-Alotaibi/KC.git
cd KC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```
GOOGLE_API_KEY=your_google_gemini_api_key_here
EDAMAM_APP_ID=your_edamam_app_id_here
EDAMAM_APP_KEY=your_edamam_app_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

## 📋 Requirements

- streamlit>=1.28.0
- google-generativeai>=0.3.0
- Pillow>=10.0.0
- PyPDF2>=3.0.1
- requests>=2.31.0
- python-dotenv>=1.0.0
- pandas>=2.0.0
- transformers>=4.36.0
- torch>=2.1.0
- textblob>=0.17.1
- edge-tts>=6.1.0
- gtts>=2.3.0

## 🎯 Usage

1. **Complete Your Profile**: Fill in your health information in the sidebar
2. **Choose AI Expert**: Select your preferred nutrition AI persona
3. **Ask Questions**: Type nutrition questions or upload food images
4. **Upload Documents**: Add your health documents for personalized advice
5. **Track Meals**: Monitor your nutrition intake with automatic logging
6. **Use CraveSmart**: Transform cravings into healthy alternatives

## 🔧 Configuration

### API Keys
- **Google Gemini**: Required for AI responses and image analysis
- **Edamam**: Optional for enhanced nutrition data accuracy
- **HeyGen**: Optional for avatar functionality

### Features Toggle
- Food Image Analysis
- Nutrition Calculator  
- Meal Logging
- AI Avatar Assistant

## 🛡️ Safety Features

- Content toxicity detection
- Nutrition safety validation
- Medical disclaimer prompts
- Allergy awareness alerts

## 🎨 Customization

- Theme switching (Light/Dark mode)
- AI personality selection
- Response creativity levels
- Custom nutrition templates

## 📊 Nutrition Data Sources

1. **Edamam API**: Professional nutrition database
2. **Local Database**: 35+ common foods with accurate nutrition data
3. **Smart Estimation**: AI-powered nutrition approximation

## 🔒 Privacy

- All data stored locally in session state
- No personal information transmitted to external services
- Environment variables for secure API key management

## 🤝 Contributing

This is a personal nutrition coaching application. For suggestions or improvements, please open an issue.

## 📄 License

This project is for personal and educational use.

## 🩺 Disclaimer

Aafiya AI provides general nutrition information and should not replace professional medical advice. Always consult with healthcare professionals for specific medical conditions or dietary needs.

---
