"""
🥗 Aafiya AI - Your Personalized Nutrition & Wellness Coach

A comprehensive nutrition coaching application powered by Google Gemini API,
featuring text generation, image analysis, function calling, and document integration.
Aafiya means "health" and "wellness" in Arabic.

Features:
- Multi-persona AI nutrition experts  
- Food image analysis with Gemini Pro Vision
- Interactive avatar assistant (Salma)
- CraveSmart: Transform cravings into healthy alternatives
- User profile management with allergy tracking
- Meal logging and recipe saving
- Dual theme mode (Ivory/Dark)
- Real-time AI responses

Created by: Nourah Alotaibi
Platform: Streamlit + Google Gemini API + HeyGen Avatar
"""

import streamlit as st
import google.generativeai as genai
import os
import json
import time
import requests
from PIL import Image
import io
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import re
import PyPDF2
import asyncio
import edge_tts
from nutrition_rag import (
    nutrition_document_uploader, 
    enhance_prompt_with_rag, 
    display_rag_info
)

# Sentiment analysis and toxicity detection
try:
    from transformers import pipeline
    # Initialize toxicity detection pipeline
    toxicity_classifier = pipeline(
        "text-classification", 
        model="unitary/toxic-bert-base-uncased",
        return_all_scores=True
    )
    sentiment_analyzer = pipeline("sentiment-analysis")
    SENTIMENT_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    # Handle any model download or import errors gracefully
    toxicity_classifier = None
    sentiment_analyzer = None
    SENTIMENT_AVAILABLE = False

# Text-to-Speech functionality enabled
TTS_AVAILABLE = True

# Avatar functionality with HeyGen (replaces Edge TTS)
HEYGEN_AVAILABLE = True  # HeyGen embed always available via JavaScript

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Aafiya AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global dropdown styling for white mode - loads on every page
st.markdown("""
<style>
/* --- Generic dropdown container --- */
div[data-baseweb="select"] > div {background:#FFFFFF!important;
                                   border:1px solid #D6E8DD!important;
                                   border-radius:6px!important;
                                   color:#125C4A!important;}

/* --- Open listbox --- */
div[role="listbox"] {background:#FFFFFF!important;}

/* --- Options --- */
div[data-baseweb="option"] {color:#125C4A!important;}
div[data-baseweb="option"]:hover,
div[data-baseweb="option"][aria-selected="true"]  {
    background:#F1FAF6!important;
    color:#0E4839!important;
    font-weight:600!important;
}

/* --- Navbar (streamlit-option-menu) --- */
ul[class*="nav"], .navbar-nav .dropdown-menu {
    background:#FFFFFF!important;
    border:1px solid #D6E8DD!important;
}
ul[class*="nav"] li a {color:#125C4A!important;}
ul[class*="nav"] li a:hover {background:#F1FAF6!important;}
</style>
""", unsafe_allow_html=True)

# Dynamic CSS based on theme mode
def get_theme_css(white_mode=False):
    if white_mode:
        return """
        <style>
            /* Ivory white background */
            .stApp {
                background-color: #FFFFF0 !important;  /* Ivory white */
                color: #0f5a5e !important;             /* Dark teal text */
            }
            
            .main .block-container {
                background: #FFFFF0;
                color: #0f5a5e;
                border-radius: 12px;
                padding: 2rem;
            }
            
            /* Global font color */
            body, p, label, span, h1, h2, h3, h4, h5, h6 {
                color: #0f5a5e !important;
            }
            
            /* Sidebar background */
            .css-1d391kg, .stSidebar, .css-1d391kg > div, section[data-testid="stSidebar"] {
                background-color: #B8E8D0 !important;  /* Green-leaning mint navbar background */
                color: #0f5a5e;
            }
            
            .css-1d391kg .element-container {
                color: #0f5a5e;
            }
            
            section[data-testid="stSidebar"] > div {
                background-color: #B8E8D0 !important;
            }
            
            /* Profile Section - Transparent with 4 shades darker than ivory background */
            .stExpander[data-testid="expander"] {
                background: #F5F5DC !important;  /* 4 shades darker than ivory (#FFFFF0) */
                border-radius: 8px !important;
                border: 1px solid #90EE90 !important;
            }
            
            .stExpander .streamlit-expanderHeader {
                background: #F5F5DC !important;  /* 4 shades darker than ivory */
            }
            
            /* Input fields (text, number, dropdowns, etc.) - Darker ivory, NO BORDERS */
            input, select, textarea {
                background-color: #F5F5DC !important;  /* Darker ivory background (beige) */
                color: #0f5a5e !important;             /* Text dark teal */
                border: none !important;               /* NO BORDERS */
                outline: none !important;             /* NO OUTLINES */
                box-shadow: none !important;          /* NO SHADOWS */
                border-radius: 8px;
            }
            
            /* Fix black Streamlit containers (like dropdown wrappers) */
            div[data-baseweb="select"],
            div[data-baseweb="input"] {
                background-color: #F5F5DC !important;  /* Darker ivory background */
                color: #0f5a5e !important;
                border: none !important;               /* NO BORDERS */
                outline: none !important;             /* NO OUTLINES */
                box-shadow: none !important;          /* NO SHADOWS */
                border-radius: 8px !important;
            }
            
            /* Specific targeting for all form elements */
            .stSelectbox > div > div {
                background-color: #F5F5DC !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stNumberInput > div > div {
                background-color: #F5F5DC !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stTextInput > div > div {
                background-color: #F5F5DC !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stTextArea > div > div {
                background-color: #F5F5DC !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* Number input +/- buttons - DARK GREEN text/icons */
            button[aria-label="Increment"], button[aria-label="Decrement"] {
                background-color: #F5F5DC !important;  /* Darker ivory background */
                color: #0f5a5e !important;  /* Dark green text for +/- buttons */
                border: none !important;               /* NO BORDERS */
                outline: none !important;             /* NO OUTLINES */
                box-shadow: none !important;          /* NO SHADOWS */
            }
            
            /* Number input +/- button icons */
            button[aria-label="Increment"] svg, button[aria-label="Decrement"] svg {
                fill: #0f5a5e !important;  /* Dark green icons */
                color: #0f5a5e !important;
            }
            
            /* Multiple choice fields - darker ivory background with green text */
            .stMultiSelect > div > div {
                background-color: #F5F5DC !important;  /* Darker ivory background */
                border: none !important;               /* NO BORDERS */
                outline: none !important;             /* NO OUTLINES */
                box-shadow: none !important;          /* NO SHADOWS */
                border-radius: 8px !important;
            }
            
            .stMultiSelect > div > div > div {
                background-color: #F5F5DC !important;  /* Darker ivory background */
                color: #228B22 !important;  /* Forest green text */
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stMultiSelect [data-baseweb="tag"] {
                background-color: #e8f5e8 !important;  /* Light green background for selected items */
                color: #228B22 !important;
                border: none !important;               /* NO BORDERS */
            }
            
            .stMultiSelect [data-baseweb="tag"] span {
                color: #228B22 !important;
            }
            
            /* Fix multiple choice input text color */
            .stMultiSelect input {
                background-color: #F5F5DC !important;
                color: #228B22 !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stMultiSelect div[role="listbox"] {
                background-color: #F5F5DC !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stMultiSelect div[role="option"] {
                background-color: #F5F5DC !important;
                color: #228B22 !important;
                border: none !important;
                outline: none !important;
            }
            
            /* Dropdown caret & icons - DARK GREEN for input elements */
            .stSelectbox svg, .stNumberInput svg, .stTextInput svg {
                fill: #0f5a5e !important;  /* Dark green dropdown arrows and input icons */
            }
            
            /* General SVG icons (non-input) keep dark */
            svg:not(.stSelectbox svg):not(.stNumberInput svg):not(.stTextInput svg):not(.stFileUploader svg):not(.stCheckbox svg):not(.stRadio svg) {
                fill: #0f5a5e !important;
            }
            
            /* Additional input element icons */
            div[data-baseweb="select"] svg,
            div[data-baseweb="input"] svg {
                fill: #0f5a5e !important;  /* Dark green icons in input containers */
            }
            
            /* Slider styling */
            .stSlider > div > div > div > div {
                background-color: #FFFFF0 !important;  /* Ivory background */
            }
            
            .stSlider svg {
                fill: #0f5a5e !important;  /* Dark green slider icons */
            }
            
            /* Select dropdown arrow */
            select {
                background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='%230f5a5e' viewBox='0 0 16 16'%3e%3cpath d='m7.247 4.86-4.796 5.481c-.566.647-.106 1.659.753 1.659h9.592a1 1 0 0 0 .753-1.659l-4.796-5.48a1 1 0 0 0-1.506 0z'/%3e%3c/svg%3e") !important;
            }
            
            /* File uploader - title background color */
            .stFileUploader > div {
                background-color: #B8E8D0 !important;  /* Title background color */
                border: none !important;               /* NO BORDERS */
                border-radius: 8px !important;
            }
            
            .stFileUploader label {
                color: #0f5a5e !important;
            }
            
            /* File uploader browse button - title background color */
            .stFileUploader button {
                background-color: #B8E8D0 !important;  /* Title background color */
                color: #0f5a5e !important;  /* Dark green text for browse button */
                border: none !important;               /* NO BORDERS */
            }
            
            /* File uploader browse button hover */
            .stFileUploader button:hover {
                background-color: #A8DCC0 !important;  /* Slightly darker on hover */
                color: #0f5a5e !important;
            }
            
            /* File uploader icons */
            .stFileUploader svg {
                fill: #0f5a5e !important;  /* Dark green file upload icons */
            }
            
            /* Checkbox styling - DARK GREEN checkmarks */
            .stCheckbox > label > div > div {
                background-color: #FFFFF0 !important;  /* Ivory background */
                border: none !important;               /* NO BORDERS */
            }
            
            .stCheckbox input:checked + div > div {
                background-color: #FFFFF0 !important;  /* Ivory background */
                border: none !important;               /* NO BORDERS */
            }
            
            /* Checkbox checkmark icon */
            .stCheckbox svg {
                fill: #0f5a5e !important;  /* Dark green checkmark */
                color: #0f5a5e !important;
            }
            
            /* Radio button styling */
            .stRadio > div > label > div > div {
                background-color: #FFFFF0 !important;  /* Ivory background */
                border: none !important;               /* NO BORDERS */
            }
            
            .stRadio input:checked + div > div {
                background-color: #FFFFF0 !important;  /* Ivory background */
                border: none !important;               /* NO BORDERS */
            }
            
            .stRadio svg {
                fill: #0f5a5e !important;  /* Dark green radio button dot */
            }
            
            /* Salma section - title background color with dark green text */
            .heygen-info {
                background: #B8E8D0 !important;  /* Same as title background */
                color: #0f5a5e !important;       /* Dark green text */
                border: none !important;         /* No borders */
            }
            
            /* White Mode - All Text Elements */
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
                color: #0f5a5e !important;
            }
            
            .stApp p, .stApp div, .stApp span, .stApp label {
                color: #0f5a5e !important;
            }
            
            .stApp .stMarkdown, .stApp .stText {
                color: #0f5a5e !important;
            }
            
            /* White Mode - Input Elements */
            .stApp .stTextInput label, .stApp .stTextArea label, .stApp .stSelectbox label, 
            .stApp .stNumberInput label, .stApp .stSlider label, .stApp .stCheckbox label,
            .stApp .stRadio label, .stApp .stMultiSelect label {
                color: #0f5a5e !important;
            }
            
            .stApp .stTextInput > div > div > input,
            .stApp .stTextArea > div > div > textarea,
            .stApp .stSelectbox > div > div > select,
            .stApp .stNumberInput > div > div > input {
                background-color: #f2ede6 !important;
                color: #0f5a5e !important;
                border: 1px solid #a7c4bd !important;
            }
            
            /* White Mode - Dropdown Options */
            .stApp .stSelectbox option {
                color: #0f5a5e !important;
            }
            
            /* White Mode - Metric Labels */
            .stApp .metric-container > div {
                color: #0f5a5e !important;
            }
            
            /* White Mode - Expander Headers */
            .stApp .streamlit-expanderHeader {
                color: #0f5a5e !important;
            }
            
            /* White Mode - Tab Labels */
            .stApp .stTabs [data-baseweb="tab-list"] button {
                color: #0f5a5e !important;
            }
            
            /* Comprehensive Input Field Styling - IVORY BACKGROUNDS */
            .stSelectbox select, .stNumberInput input, .stTextInput input, .stTextArea textarea {
                background-color: #FFFFF0 !important;
                color: #0f5a5e !important;
                border: none !important;               /* NO BORDERS */
            }
            
            /* AI Personality and Response Length dropdowns */
            .stSelectbox > div > div > div {
                background-color: #FFFFF0 !important;
                color: #0f5a5e !important;
            }
            
            /* Dropdown options */
            .stSelectbox option {
                background-color: #F5F5DC !important;
                color: #0f5a5e !important;
                border: none !important;
                outline: none !important;
            }
            
            /* Multiple choice additional styling */
            .stMultiSelect input::placeholder {
                color: #228B22 !important;
            }
            
            .stMultiSelect div[data-baseweb="popover"] {
                background-color: #F5F5DC !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* COMPREHENSIVE INPUT FIELD OVERRIDE - NO BORDERS, DARKER IVORY BACKGROUNDS */
            input[type="text"], input[type="number"], input[type="email"], input[type="password"], 
            input[type="file"], input[type="search"], input[type="url"], input[type="tel"],
            select, textarea, .stTextInput input, .stNumberInput input, .stSelectbox select,
            .stTextArea textarea, .stMultiSelect input {
                background-color: #F5F5DC !important;
                color: #0f5a5e !important;
                border: 0px solid transparent !important;
                outline: 0px solid transparent !important;
                box-shadow: none !important;
                -webkit-appearance: none !important;
                -moz-appearance: none !important;
                appearance: none !important;
            }
            
            /* Override Streamlit input containers - COMPREHENSIVE BORDER REMOVAL */
            .stTextInput, .stTextInput > div, .stTextInput > div > div, .stTextInput > div > div > input,
            .stNumberInput, .stNumberInput > div, .stNumberInput > div > div, .stNumberInput > div > div > input,
            .stSelectbox, .stSelectbox > div, .stSelectbox > div > div, .stSelectbox > div > div > select,
            .stTextArea, .stTextArea > div, .stTextArea > div > div, .stTextArea > div > div > textarea,
            .stMultiSelect, .stMultiSelect > div, .stMultiSelect > div > div, .stMultiSelect > div > div > div {
                background-color: #F5F5DC !important;
                border: 0px solid transparent !important;
                outline: 0px solid transparent !important;
                box-shadow: none !important;
                -webkit-box-shadow: none !important;
                -moz-box-shadow: none !important;
            }
            
            /* Override deeply nested input elements */
            div[data-testid="textInput"], div[data-testid="numberInput"], 
            div[data-testid="selectbox"], div[data-testid="textArea"] {
                background-color: #F5F5DC !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* Fix number input controls - age, weight, height, goal duration */
            .stNumberInput input[type="number"] {
                background-color: #F5F5DC !important;
                color: #0f5a5e !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stNumberInput > div > div > input {
                background-color: #F5F5DC !important;
                color: #0f5a5e !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* Fix file uploader input areas - title background color */
            .stFileUploader input[type="file"] {
                background-color: #B8E8D0 !important;
                color: #0f5a5e !important;
                border: none !important;
            }
            
            .stFileUploader > div > div {
                background-color: #B8E8D0 !important;
                border: none !important;
                outline: none !important;
            }
            
            /* Fix file drop zone */
            .stFileUploader [data-testid="fileDropzone"] {
                background-color: #B8E8D0 !important;
                border: none !important;
                outline: none !important;
            }
            
            /* Fix file uploader text input field */
            .stFileUploader input[type="text"] {
                background-color: #B8E8D0 !important;
                color: #0f5a5e !important;
                border: none !important;
            }
            
            /* Fix file uploader wrapper */
            .stFileUploader div[data-testid="fileUploaderDropzone"] {
                background-color: #B8E8D0 !important;
                border: none !important;
                outline: none !important;
            }
            
            /* Fix file uploader inner elements */
            .stFileUploader div[data-testid="fileUploaderDropzone"] > div {
                background-color: #B8E8D0 !important;
                border: none !important;
            }
            
            /* Fix file uploader file input specifically */
            .stFileUploader input[type="file"]::-webkit-file-upload-button {
                background-color: #B8E8D0 !important;
                color: #0f5a5e !important;
                border: none !important;
            }
            
            /* Fix file uploader choose files button */
            .stFileUploader button[kind="secondary"] {
                background-color: #B8E8D0 !important;
                color: #0f5a5e !important;
                border: none !important;
            }
            
            /* Remove ALL borders from input containers - COMPREHENSIVE */
            div[data-baseweb="input"], div[data-baseweb="input"]:focus, div[data-baseweb="input"]:hover,
            div[data-baseweb="select"], div[data-baseweb="select"]:focus, div[data-baseweb="select"]:hover,
            .stTextInput, .stTextInput:focus-within, .stTextInput:hover,
            .stNumberInput, .stNumberInput:focus-within, .stNumberInput:hover,
            .stSelectbox, .stSelectbox:focus-within, .stSelectbox:hover,
            .stTextArea, .stTextArea:focus-within, .stTextArea:hover,
            .stMultiSelect, .stMultiSelect:focus-within, .stMultiSelect:hover,
            .stFileUploader, .stFileUploader:focus-within, .stFileUploader:hover {
                border: 0px solid transparent !important;
                outline: 0px solid transparent !important;
                box-shadow: none !important;
                -webkit-box-shadow: none !important;
                -moz-box-shadow: none !important;
            }
            
            /* Remove focus borders - COMPREHENSIVE */
            input, input:focus, input:hover, input:active,
            select, select:focus, select:hover, select:active,
            textarea, textarea:focus, textarea:hover, textarea:active {
                border: 0px solid transparent !important;
                outline: 0px solid transparent !important;
                box-shadow: none !important;
                -webkit-box-shadow: none !important;
                -moz-box-shadow: none !important;
                -webkit-appearance: none !important;
                -moz-appearance: none !important;
                appearance: none !important;
            }
            
            /* Open Salma button - same as title background */
            a[href*="heygen"] {
                background: #B8E8D0 !important;
                color: #0f5a5e !important;
                border: none !important;
            }
            
            /* Salma info box - pastel opal green background */
            .avatar-status {
                background: #C8E6C9 !important;  /* Pastel opal green */
                color: #0f5a5e !important;       /* Dark green text */
                border: none !important;         /* No borders */
            }
            
            .avatar-status strong {
                color: #0f5a5e !important;       /* Dark green for strong text */
            }
            
            .avatar-status span {
                color: #0f5a5e !important;       /* Dark green for span text */
            }
            
            /* Knowledge Base box - 2 points lighter */
            .knowledge-base-box {
                background: #B8DDB8 !important;  /* 2 points lighter green background */
                color: #0f5a5e !important;       /* Dark green text */
                border: none !important;         /* No borders */
                border-radius: 8px !important;
                padding: 1.5rem !important;
                margin: 1rem 0 !important;
            }
            
            .knowledge-base-box h3 {
                color: #0f5a5e !important;       /* Dark green heading */
                margin-bottom: 0.5rem !important;
            }
            
            .knowledge-base-box p {
                color: #0f5a5e !important;       /* Dark green paragraph text */
                margin: 0 !important;
            }
            
            /* Ivory Mode - Cards and Components */
            .nutrition-card {
                background: #FFFFFB !important;  /* Light ivory for cards */
                border-left: 4px solid #8B4513 !important;  /* Saddle brown accent */
                color: #0f5a5e !important;
            }
            
            .calorie-display {
                background: #FFFFFB !important;
                border: 2px solid #8B4513 !important;
                color: #0f5a5e !important;
            }
            
            /* Beige Mode - Alert Boxes */
            .stAlert[data-baseweb="notification"][kind="info"] {
                background-color: #e6f3ff !important;  /* Light blue background */
                color: #0f5a5e !important;
                border: 2px solid #2c3e50 !important;
            }
            
            .stAlert[data-baseweb="notification"][kind="success"] {
                background-color: #e8f5e8 !important;  /* Light green background */
                color: #228B22 !important;
                border: 2px solid #228B22 !important;
            }
            
            .stAlert[data-baseweb="notification"][kind="warning"] {
                background-color: #fff3cd !important;  /* Light yellow background */
                color: #856404 !important;
                border: 2px solid #856404 !important;
            }
            
            .stAlert[data-baseweb="notification"][kind="error"] {
                background-color: #f8d7da !important;  /* Light red background */
                color: #721c24 !important;
                border: 2px solid #721c24 !important;
            }
            
            /* Beige Mode - Buttons (keep original gradient but with subtle shadow) */
            .stButton > button {
                box-shadow: 0 4px 8px rgba(44, 62, 80, 0.2) !important;
            }
            
            /* Beige Mode - Better contrast for links */
            a {
                color: #8B4513 !important;  /* Saddle brown for links */
            }
            
            a:hover {
                color: #5D2F0A !important;  /* Darker brown on hover */
            }
            
            /* Pastel Green Banner Text Override */
            .hero-banner h1, .hero-banner h2 {
                color: #1a4d1a !important;  /* Bolder, darker green text on pastel green background */
            }
            
            .hero-banner p {
                color: #1a4d1a !important;  /* Bolder, darker green text for subtitle */
                opacity: 1 !important;
            }
            
            /* All buttons - title background color */
            .stButton > button {
                background: #B8E8D0 !important;  /* Same as title background */
                color: #0f5a5e !important;       /* Dark green text */
                border: none !important;         /* No borders */
                box-shadow: none !important;     /* No shadows */
            }
            
            .stButton > button:hover {
                background: #A8DCC0 !important;  /* Slightly darker on hover */
                color: #0f5a5e !important;       /* Dark green text */
            }
            
            /* Specific styling for allergy add button - mint green - IVORY MODE ONLY */
            button[key="add_allergy_btn"], 
            .stButton > button[key="add_allergy_btn"],
            div[data-testid="stButton"] button[key="add_allergy_btn"] {
                font-size: 0.8rem !important;    /* 3 points smaller */
                padding: 0.25rem 0.5rem !important; /* Smaller padding */
                height: auto !important;
                min-height: 32px !important;     /* Smaller height */
                margin-top: 14px !important;     /* Position 4 points down */
                background: #B8E8D0 !important;  /* Mint green */
                color: #0f5a5e !important;       /* Dark teal text */
                border: none !important;
                border-radius: 6px !important;
            }
            
            button[key="add_allergy_btn"]:hover,
            .stButton > button[key="add_allergy_btn"]:hover,
            div[data-testid="stButton"] button[key="add_allergy_btn"]:hover,
            * button[key="add_allergy_btn"]:hover {
                background: #A8DCC0 !important;  /* Darker mint green on hover */
                color: #0f5a5e !important;       /* Dark teal text */
            }
            
            /* FORCE Add button styling - PASTEL GREEN - ULTIMATE OVERRIDE */
            html body div div div div button[key="add_allergy_btn"],
            html div button[key="add_allergy_btn"],
            * button[key="add_allergy_btn"] {
                background-color: #90EE90 !important;  /* Pastel green */
                background: #90EE90 !important;
                color: #006400 !important;            /* Darker green text */
                margin-top: 14px !important;
                font-size: 0.8rem !important;
                padding: 0.25rem 0.5rem !important;
                border: none !important;
                border-radius: 6px !important;
                height: auto !important;
                min-height: 32px !important;
            }
            
            /* ULTIMATE BORDER REMOVAL - Target every possible input element */
            * [class*="Input"], * [class*="Select"], * [class*="TextArea"], * [class*="MultiSelect"],
            * [data-testid*="Input"], * [data-testid*="Select"], * [data-testid*="TextArea"],
            * [role="textbox"], * [role="combobox"], * [role="listbox"], * [role="option"],
            input, select, textarea, button[type="button"] {
                border: 0px solid transparent !important;
                border-width: 0px !important;
                border-style: none !important;
                outline: 0px solid transparent !important;
                outline-width: 0px !important;
                outline-style: none !important;
                box-shadow: none !important;
                -webkit-box-shadow: none !important;
                -moz-box-shadow: none !important;
            }
            
            /* Remove borders on focus, hover, active states */
            * [class*="Input"]:focus, * [class*="Input"]:hover, * [class*="Input"]:active,
            * [class*="Select"]:focus, * [class*="Select"]:hover, * [class*="Select"]:active,
            * [class*="TextArea"]:focus, * [class*="TextArea"]:hover, * [class*="TextArea"]:active,
            input:focus, input:hover, input:active,
            select:focus, select:hover, select:active,
            textarea:focus, textarea:hover, textarea:active {
                border: 0px solid transparent !important;
                outline: 0px solid transparent !important;
                box-shadow: none !important;
            }
            
            /* Remove all borders from inputs and textareas - IVORY MODE ONLY */
            input, textarea, select {
                border: none !important;
                outline: none !important;
                background-color: inherit !important;
                box-shadow: none !important;
            }
            
            /* Streamlit input containers - IVORY MODE ONLY */
            .stTextInput > div > div, .stTextArea > div > div, .stNumberInput > div > div,
            .stSelectbox > div > div, .stMultiSelect > div > div {
                border: none !important;
                outline: none !important;
                background-color: inherit !important;
                box-shadow: none !important;
            }
            
            /* All input types - IVORY MODE ONLY */
            input[type="text"], input[type="number"], input[type="email"], input[type="password"],
            input[type="search"], input[type="tel"], input[type="url"], textarea, select {
                border: none !important;
                outline: none !important;
                background-color: inherit !important;
                box-shadow: none !important;
            }
            
            /* Placeholder text color - IVORY MODE ONLY */
            input::placeholder, textarea::placeholder {
                color: #006400 !important;
            }
            
            /* COMPREHENSIVE FILE INPUT AND NUMBER INPUT FIXES - IVORY MODE ONLY */
            
            /* File input browse button - all possible selectors */
            input[type="file"], input[type="file"]::-webkit-file-upload-button {
                background-color: #F5F5DC !important;
                color: #0f5a5e !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* File input container and wrapper */
            .stFileUploader, .stFileUploader > div, .stFileUploader > div > div,
            .stFileUploader section, .stFileUploader section > div {
                background-color: #F5F5DC !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* File input text and labels */
            .stFileUploader label, .stFileUploader span, .stFileUploader p {
                color: #0f5a5e !important;
            }
            
            /* Number input +/- buttons - comprehensive targeting */
            .stNumberInput button, .stNumberInput button[role="button"],
            button[data-testid="baseButton-secondary"], button[data-testid="baseButton-minimal"],
            .stNumberInput div[role="button"], .stNumberInput [class*="step"] {
                background-color: #F5F5DC !important;
                color: #0f5a5e !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* Number input +/- button icons and SVGs */
            .stNumberInput svg, .stNumberInput path,
            button[aria-label*="increment"] svg, button[aria-label*="decrement"] svg,
            button[aria-label*="Increment"] svg, button[aria-label*="Decrement"] svg {
                fill: #0f5a5e !important;
                color: #0f5a5e !important;
                stroke: #0f5a5e !important;
            }
            
            /* Additional file upload selectors */
            [data-testid="fileUploader"], [data-testid="fileDropzone"],
            [data-testid="fileUploaderDropzone"], [data-testid="fileUploaderInput"] {
                background-color: #F5F5DC !important;
                color: #0f5a5e !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            /* Transparent background for personality section - IVORY MODE ONLY */
            .personality-box {
                background-color: transparent !important;
                padding: 0rem !important;
                border-radius: 0px !important;
                margin: 0rem 0 !important;
                border: none !important;
            }
            
            /* Craving input widget borders - IVORY MODE */
            textarea[key="craving_main_input"], textarea[key="craving_main_input"] + div,
            .stTextArea[data-testid*="craving"] > div > div, 
            .stTextArea:has(textarea[key="craving_main_input"]) > div > div {
                border: 2px solid #A8D5A8 !important;
                border-radius: 8px !important;
            }
            
            /* DROPDOWN MENU STYLING - IVORY MODE ONLY */
            .stSelectbox > div > div > div[role="listbox"] {
                background-color: #FFFFF0 !important;  /* Ivory background */
                border: 1px solid #B8E8D0 !important;
                border-radius: 8px !important;
            }
            
            .stSelectbox > div > div > div[role="listbox"] div[role="option"] {
                background-color: #FFFFF0 !important;  /* Ivory background for options */
                color: #0f5a5e !important;             /* Dark teal text */
            }
            
            .stSelectbox > div > div > div[role="listbox"] div[role="option"]:hover {
                background-color: #B8E8D0 !important;  /* Mint green on hover */
                color: #0f5a5e !important;
            }
            
            /* Number input dropdown */
            .stNumberInput > div > div > div[role="listbox"] {
                background-color: #FFFFF0 !important;
                border: 1px solid #B8E8D0 !important;
                border-radius: 8px !important;
            }
            
            .stNumberInput > div > div > div[role="listbox"] div[role="option"] {
                background-color: #FFFFF0 !important;
                color: #0f5a5e !important;
            }
            
            .stNumberInput > div > div > div[role="listbox"] div[role="option"]:hover {
                background-color: #B8E8D0 !important;
                color: #0f5a5e !important;
            }
            
            /* CraveSmart button beige - IVORY MODE ONLY */
            section[data-testid="stSidebar"] button[kind="secondary"] {
                background: #F5F5DC !important;  /* Beige */
                color: #0f5a5e !important;       /* Dark green text */
                border: 2px solid #D3D3D3 !important;  /* Border */
                border-radius: 8px !important;
            }
            
            section[data-testid="stSidebar"] button[kind="secondary"]:hover {
                background: #EEEED2 !important;  /* Darker beige on hover */
                color: #0f5a5e !important;
                border: 2px solid #C8C8BE !important;
            }
            
            /* About persona expander white box - IVORY MODE ONLY (exclude profile expander) */
            .stExpander:has([data-testid="stExpanderToggleIcon"]):not(:has-text("Complete Your Profile")) {
                background-color: white !important;
                border-radius: 8px !important;
                padding: 0.5rem !important;
                margin: 0.25rem 0 !important;
                border: 1px solid #e0e0e0 !important;
            }
            
            /* Profile section - 4 shades darker with MINT GREEN border - IVORY MODE ONLY */
            .stExpander[data-testid="expander"] {
                background: #E6E6DC !important;  /* 4 shades darker than ivory (#FFFFF0) */
                border-radius: 8px !important;
                border: 2px solid #B8E8D0 !important;  /* Mint green border */
            }
            
            .stExpander .streamlit-expanderHeader {
                background: #E6E6DC !important;  /* 4 shades darker than ivory */
            }
            
            /* Force profile expander to be 4 shades darker with MINT GREEN border - higher specificity */
            .stExpander:has(div:contains("Complete Your Profile")) {
                background: #E6E6DC !important;  /* 4 shades darker than ivory */
                border: 2px solid #B8E8D0 !important;  /* Mint green border */
            }
            
            .stExpander:has(div:contains("Complete Your Profile")) .streamlit-expanderHeader {
                background: #E6E6DC !important;  /* 4 shades darker than ivory */
            }
            
            /* Allergy add button mint green - IVORY MODE ONLY */
            button[key="add_allergy_btn"] {
                background-color: #B8E8D0 !important;  /* Mint green */
                color: #0f5a5e !important;             /* Dark teal text */
                border: none !important;
                border-radius: 6px !important;
                font-size: 0.8rem !important;
                padding: 0.25rem 0.5rem !important;
                height: auto !important;
                min-height: 32px !important;
                margin-top: 14px !important;           /* 4 points down */
            }
            
            button[key="add_allergy_btn"]:hover {
                background-color: #A8DCC0 !important;  /* Darker mint green on hover */
                color: #0f5a5e !important;
            }
            
            /* CraveSmart Transform My Craving button - 5 shades darker - IVORY MODE ONLY */
            button[data-testid="baseButton-primary"]:contains("Transform My Craving!"),
            .stButton > button[data-testid="baseButton-primary"] {
                background: linear-gradient(135deg, #8FD3B0, #7FB896) !important;
                color: #0f5a5e !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
            }
            
            button[data-testid="baseButton-primary"]:contains("Transform My Craving!"):hover,
            .stButton > button[data-testid="baseButton-primary"]:hover {
                background: linear-gradient(135deg, #7FB896, #6FA07C) !important;
                color: #0f5a5e !important;
            }
            
        """
    else:
        return """
        <style>
            /* Dark Mode - Main App Background */
            .stApp {
                background: #062b2c;
            }
            
            .main .block-container {
                background: #062b2c;
                color: #e2fef9;
                border-radius: 12px;
                padding: 2rem;
            }
            
            /* Dark Mode - Sidebar Styling */
            .css-1d391kg, .stSidebar, .css-1d391kg > div, section[data-testid="stSidebar"] {
                background-color: #051f20 !important;
                color: #b9dfd9;
            }
            
            .css-1d391kg .element-container {
                color: #b9dfd9;
            }
            
            section[data-testid="stSidebar"] > div {
                background-color: #051f20 !important;
            }
            
            /* Placeholder text color - DARK MODE ONLY */
            input::placeholder, textarea::placeholder {
                color: #b0b0b0 !important;
            }
            
            /* Specific styling for allergy add button - DARK MODE */
            button[key="add_allergy_btn"] {
                margin-top: 10px !important;     /* Position 2 points more down */
            }
            
            /* Input fields - DARK MODE */
            div[data-testid="textInput"] > div > div > input,
            div[data-testid="numberInput"] > div > div > input,
            div[data-testid="textArea"] > div > div > textarea,
            .stTextInput input, .stNumberInput input, .stTextArea textarea {
                background-color: #051f20 !important;
                color: #e2fef9 !important;
                border: 1px solid #2d5a5c !important;
                border-radius: 6px !important;
            }
            
            /* Selectbox and Multiselect - DARK MODE */
            .stSelectbox > div > div > div,
            .stMultiSelect > div > div > div,
            div[data-testid="selectbox"] > div,
            div[data-testid="multiselect"] > div {
                background-color: #051f20 !important;
                border: 1px solid #2d5a5c !important;
                border-radius: 6px !important;
            }
            
            .stSelectbox > div > div > div > div,
            .stMultiSelect > div > div > div > div,
            div[data-testid="selectbox"] > div > div,
            div[data-testid="multiselect"] > div > div {
                background-color: #051f20 !important;
                color: #e2fef9 !important;
            }
            
            /* Selectbox dropdown options - DARK MODE */
            .stSelectbox ul,
            .stMultiSelect ul,
            div[data-testid="selectbox"] ul,
            div[data-testid="multiselect"] ul {
                background-color: #051f20 !important;
                border: 1px solid #2d5a5c !important;
            }
            
            .stSelectbox li,
            .stMultiSelect li,
            div[data-testid="selectbox"] li,
            div[data-testid="multiselect"] li {
                background-color: #051f20 !important;
                color: #e2fef9 !important;
            }
            
            .stSelectbox li:hover,
            .stMultiSelect li:hover,
            div[data-testid="selectbox"] li:hover,
            div[data-testid="multiselect"] li:hover {
                background-color: #2d5a5c !important;
            }
            
            /* Craving input widget borders - DARK MODE */
            textarea[key="craving_main_input"], textarea[key="craving_main_input"] + div,
            .stTextArea[data-testid*="craving"] > div > div, 
            .stTextArea:has(textarea[key="craving_main_input"]) > div > div {
                border: 2px solid #91f2c4 !important;
                border-radius: 8px !important;
            }
        """

# Dynamic CSS based on theme
white_mode = st.session_state.get('white_mode', False)
if white_mode:
    input_bg = "#FFFFF0"  # Ivory background for white mode
else:
    input_bg = "rgba(17, 47, 48, 0.9)"     # Muted teal for dark mode
input_color = "#0f5a5e" if white_mode else "#e2fef9"
text_color = "#0f5a5e" if white_mode else "#b9dfd9"

# Build CSS without f-string issues
css_content = get_theme_css(white_mode) + """
    
    /* Hero Banner */
    .hero-banner {
        background: """ + ("#B8E8D0" if white_mode else "linear-gradient(to right, #a8f6c2, #007a87)") + """;
        padding: 0.35rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: """ + ("none" if white_mode else "0 8px 25px rgba(0, 0, 0, 0.4), 0 4px 10px rgba(0, 0, 0, 0.2)") + """;
        width: calc(100% - 7px);
        margin-left: auto;
        margin-right: auto;
        border: """ + ("none" if white_mode else "none") + """;
    }
    
    .hero-banner h1, .hero-banner h2 {
        color: """ + ("#1a4d1a" if white_mode else "#ffffff") + """;
        font-weight: bold;
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        text-shadow: none;
    }
    
    .hero-banner p {
        color: """ + ("#1a4d1a" if white_mode else "#ffffff") + """;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0 0 1.5rem 0;
        margin-top: -0.3rem;
        opacity: """ + ("1" if white_mode else "0.8") + """;
    }
    
    /* Nutrition Image */
    .nutrition-image {
        width: 100%;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4), 0 4px 12px rgba(0, 0, 0, 0.2);
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Cards */
    .nutrition-card {
        background: #112f30;
        border-left: 4px solid #91f2c4;
        color: #b9dfd9;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .calorie-display {
        background: #112f30;
        border: 2px solid #91f2c4;
        color: #e2fef9;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    
    /* Avatar Styling */
    .heygen-info {
        background: #23435b;
        color: #e2fef9;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .avatar-status {
        background: #112f30;
        border-left: 4px solid #91f2c4;
        color: #b9dfd9;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .avatar-status strong {
        color: #e2fef9;
    }
    
    /* Buttons */
    .stButton > button {
        background: """ + ("#B8E8D0" if white_mode else "linear-gradient(to right, #91f2c4, #0f5a5e)") + """;
        color: """ + ("#0f5a5e" if white_mode else "#ffffff") + """;
        border: """ + ("none" if white_mode else "none") + """;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 900;
        font-size: 1.65rem;
        transition: all 0.3s ease;
        box-shadow: """ + ("none" if white_mode else "0 4px 15px rgba(0, 0, 0, 0.3), 0 2px 8px rgba(0, 0, 0, 0.15)") + """;
    }
    
    .stButton > button:hover {
        background: """ + ("#A8DCC0" if white_mode else "linear-gradient(to right, #a8f6c2, #007a87)") + """;
        color: """ + ("#0f5a5e" if white_mode else "#ffffff") + """;
        transform: translateY(-2px);
        box-shadow: """ + ("none" if white_mode else "0 6px 20px rgba(0, 0, 0, 0.4), 0 3px 12px rgba(0, 0, 0, 0.2)") + """;
    }
    
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input,
    .stMultiSelect > div > div > div {
        background-color: """ + input_bg + """;
        border: """ + ("none" if white_mode else "1px solid #91f2c4") + """;
        color: """ + input_color + """;
        border-radius: 8px;
        padding: 0.5rem;
        font-size: 1rem;
        min-height: 40px;
    }
    
    .stTextArea > div > div > textarea {
        min-height: 80px;
        resize: vertical;
    }
    
    /* Match input fields to textarea style */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: """ + input_bg + """ !important;
        color: """ + input_color + """ !important;
        border: """ + ("none" if white_mode else "1px solid #91f2c4") + """ !important;
        border-radius: 8px !important;
    }

    input, select, textarea {
        color: """ + input_color + """ !important;
        background-color: """ + input_bg + """ !important;
    }
    
    /* Text Colors */
    .stMarkdown, .stText {
        color: """ + text_color + """;
    }
    
    h1, h2, h3 {
        color: """ + ("#0f5a5e" if white_mode else "#e2fef9") + """;
    }
    
    /* Full-screen link styling */
    .fullscreen-link {
        display: inline-block;
        padding: 10px 20px;
        background: """ + ("#B8E8D0" if white_mode else "linear-gradient(to right, #91f2c4, #0f5a5e)") + """;
        color: """ + ("#0f5a5e" if white_mode else "white") + """;
        text-decoration: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: """ + ("none" if white_mode else "none") + """;
    }
    
    .fullscreen-link:hover {
        background: """ + ("#A8DCC0" if white_mode else "linear-gradient(to right, #a8f6c2, #007a87)") + """;
        color: """ + ("#0f5a5e" if white_mode else "white") + """;
        transform: translateY(-2px);
    }
    
    /* Hide HeyGen branding */
    #heygen-streaming-embed [class*="powered"],
    #heygen-streaming-embed [class*="heygen"],
    #heygen-streaming-embed [class*="logo"],
    #heygen-streaming-embed [id*="powered"],
    #heygen-streaming-embed [id*="heygen"],
    #heygen-streaming-embed [id*="logo"],
    iframe[src*="heygen"] [class*="powered"],
    iframe[src*="heygen"] [class*="logo"],
    iframe[src*="heygen"] [id*="powered"],
    iframe[src*="heygen"] [id*="logo"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
    }
    
    /* Salma embedded avatar styling */
    .salma-avatar-container {
        background: linear-gradient(135deg, #23435b 0%, #112f30 100%);
        border: 2px solid #91f2c4;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(145, 242, 196, 0.2);
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .salma-avatar-container:hover {
        box-shadow: 0 8px 24px rgba(145, 242, 196, 0.3);
        border-color: #a8f6c2;
    }
    
    .salma-controls {
        background: rgba(17, 47, 48, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 8px;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    
    /* Full-screen link styling */
    .fullscreen-link {
        display: inline-block;
        padding: 8px 16px;
        background: linear-gradient(45deg, #91f2c4, #0f5a5e);
        color: white !important;
        text-decoration: none !important;
        border-radius: 6px;
        font-size: 0.9em;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(145, 242, 196, 0.3);
    }
    
    .fullscreen-link:hover {
        background: linear-gradient(45deg, #a8f6c2, #007a87);
        box-shadow: 0 4px 8px rgba(145, 242, 196, 0.4);
        transform: translateY(-1px);
        color: white !important;
    }
    
    /* Button container styling */
    .avatar-button-container {
        background: rgba(17, 47, 48, 0.8);
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
        border: 1px solid rgba(145, 242, 196, 0.2);
    }
    
    /* Warning and Info Boxes */
    .stAlert[data-baseweb="notification"] {
        border-radius: 8px;
    }
    
    .stAlert[data-baseweb="notification"][kind="error"] {
        background-color: #7f3d3d;
        color: #ffffff;
        border: 1px solid #a05252;
    }
    
    .stAlert[data-baseweb="notification"][kind="info"] {
        background-color: #23435b;
        color: #e2fef9;
        border: 1px solid #2e5470;
    }
    
    .stAlert[data-baseweb="notification"][kind="warning"] {
        background-color: #7f3d3d;
        color: #ffffff;
        border: 1px solid #a05252;
    }
    
    .stAlert[data-baseweb="notification"][kind="success"] {
        background-color: #112f30;
        color: #91f2c4;
        border: 1px solid #91f2c4;
    }
    
    /* Profile Icon Styling */
    .profile-icon {
        color: #aa4acb;
        font-size: 1.2em;
    }
    
    /* White Mode Adjustments */
    .white-mode-text {
        color: #0f5a5e;
    }
    
    .white-mode-card {
        background: rgba(240, 249, 248, 0.8);
        border: 1px solid #a8f6c2;
        color: #0f5a5e;
    }
    
    .white-mode-info {
        background: rgba(35, 67, 91, 0.1);
        color: #0f5a5e;
        border: 1px solid #a8f6c2;
    }
    
    .white-mode-warning {
        background: rgba(127, 61, 61, 0.1);
        color: #0f5a5e;
        border: 1px solid #7f3d3d;
    }
</style>
"""

st.markdown(css_content, unsafe_allow_html=True)

st.markdown("""
<!-- HeyGen Interactive Streaming Avatar Integration -->
<script>
!function(window){
    const host="https://labs.heygen.com",
    url=host+"/guest/streaming-embed?share=eyJxdWFsaXR5IjoiaGlnaCIsImF2YXRhck5hbWUiOiJBbGVzc2FuZHJhX0NoYWlyX1NpdHRpbmdf%0D%0AcHVibGljIiwicHJldmlld0ltZyI6Imh0dHBzOi8vZmlsZXMyLmhleWdlbi5haS9hdmF0YXIvdjMv%0D%0AODllMDdiODI2ZjFjNGNiMWE1NTQ5MjAxY2RkOGY0ZDZfNTUzMDAvcHJldmlld190YXJnZXQud2Vi%0D%0AcCIsIm5lZWRSZW1vdmVCYWNrZ3JvdW5kIjpmYWxzZSwia25vd2xlZGdlQmFzZUlkIjoiZTQ0MzAw%0D%0AYWY5YWJjNGRlNmJlMjk4MzI5MzVlOTUzZjIiLCJ1c2VybmFtZSI6IjYwOGYyODY0MWE3ODRjZDk5%0D%0ANzZiZjMwNDQ4OGNhNTcxIn0%3D&inIFrame=1",
    clientWidth=document.body.clientWidth,
    wrapDiv=document.createElement("div");
    
    wrapDiv.id="heygen-streaming-embed";
    const container=document.createElement("div");
    container.id="heygen-streaming-container";
    
    const stylesheet=document.createElement("style");
    stylesheet.innerHTML=`
      #heygen-streaming-embed {
        z-index: 9999;
        position: fixed;
        left: 40px;
        bottom: 40px;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        border: 2px solid #4CAF50;
        box-shadow: 0px 8px 24px 0px rgba(76, 175, 80, 0.3);
        transition: all linear 0.1s;
        overflow: hidden;
        opacity: 0;
        visibility: hidden;
      }
      #heygen-streaming-embed.show {
        opacity: 1;
        visibility: visible;
      }
      #heygen-streaming-embed.expand {
        ${clientWidth<540?"height: 266px; width: 96%; left: 50%; transform: translateX(-50%);":"height: 366px; width: calc(366px * 16 / 9);"}
        border: 2px solid #4CAF50;
        border-radius: 12px;
        box-shadow: 0px 12px 32px 0px rgba(76, 175, 80, 0.4);
      }
      #heygen-streaming-container {
        width: 100%;
        height: 100%;
      }
      #heygen-streaming-container iframe {
        width: 100%;
        height: 100%;
        border: 0;
      }
    `;
    
    const iframe=document.createElement("iframe");
    iframe.allowFullscreen=false;
    iframe.title="Salma - AI Nutritionist";
    iframe.role="dialog";
    iframe.allow="microphone";
    iframe.src=url;
    
    let visible=false,initial=false;
    
    // Global functions for Streamlit integration
    window.heygenStreamingAPI = {
        isVisible: () => visible,
        isInitialized: () => initial,
        show: () => {
            if (initial) {
                visible = true;
                wrapDiv.classList.add("expand");
            }
        },
        hide: () => {
            visible = false;
            wrapDiv.classList.remove("expand");
        },
        sendMessage: (message) => {
            // Send message to HeyGen avatar
            if (iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage({
                    type: 'streaming-embed-send',
                    message: message
                }, host);
            }
        }
    };
    
    window.addEventListener("message",(e=>{
        if(e.origin===host && e.data && e.data.type && "streaming-embed"===e.data.type) {
            if("init"===e.data.action) {
                initial=true;
                wrapDiv.classList.toggle("show",initial);
                console.log("HeyGen Avatar (Salma) initialized");
            } else if("show"===e.data.action) {
                visible=true;
                wrapDiv.classList.toggle("expand",visible);
                console.log("HeyGen Avatar (Salma) expanded");
            } else if("hide"===e.data.action) {
                visible=false;
                wrapDiv.classList.toggle("expand",visible);
                console.log("HeyGen Avatar (Salma) minimized");
            }
        }
    }));
    
    container.appendChild(iframe);
    wrapDiv.appendChild(stylesheet);
    wrapDiv.appendChild(container);
    document.body.appendChild(wrapDiv);
    
    console.log("HeyGen Streaming Avatar (Salma) integration loaded");
}(globalThis);
</script>
""", unsafe_allow_html=True)

def configure_gemini():
    """Configure Gemini API with proper error handling"""
    try:
        api_key = os.getenv('GOOGLE_API_KEY') or st.secrets.get('GOOGLE_API_KEY')
        if not api_key:
            st.error("🔑 Please set your GOOGLE_API_KEY in environment variables or Streamlit secrets")
            st.info("Get your API key from: https://makersuite.google.com/app/apikey")
            st.stop()
        
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini API: {str(e)}")
        return False

def initialize_session_state():
    """Initialize session state variables for Aafiya AI"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'nutrition_data' not in st.session_state:
        st.session_state.nutrition_data = []
    if 'meal_log' not in st.session_state:
        st.session_state.meal_log = []
    if 'user_recipe_list' not in st.session_state:
        st.session_state.user_recipe_list = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'age': None,
            'gender': None,
            'weight': None,
            'height': None,
            'activity_level': None,
            'goal': None,
            'allergies': '',
            'health_issues': '',
            'goal_duration': 4,
            'unpreferred_foods': [],
            'profile_complete': False
        }
    if 'white_mode' not in st.session_state:
        st.session_state.white_mode = False
    if 'clear_image' not in st.session_state:
        st.session_state.clear_image = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"

def get_nutrition_prompt_templates():
    """Return nutrition-focused prompt templates"""
    return {
        "Meal Analysis": "Analyze this meal and provide nutritional insights: {user_input}. Include calories, macros, and health rating.",
        "Healthy Alternatives": "Suggest 3 healthier alternatives for: {user_input}. Focus on nutrition and taste.",
        "Diet Plan": "Create a daily meal plan for someone who wants to {user_input}. Include breakfast, lunch, dinner, and snacks.",
        "Calorie Counter": "Calculate the approximate calories and macros for: {user_input}. Break down by ingredient.",
        "Nutrition Education": "Explain the nutritional benefits and concerns about: {user_input}. Keep it simple and actionable.",
        "Recipe Modification": "Make this recipe healthier while keeping it tasty: {user_input}. Suggest ingredient swaps."
    }

def get_nutrition_ai_personas():
    """Return nutrition expert AI personas"""
    return {
        "Friendly Nutritionist": {
            "system_instruction": "You are a warm, encouraging registered dietitian who provides evidence-based nutrition advice. Always be supportive and focus on sustainable healthy habits. Use emojis and friendly language.",
            "temperature": 0.6,
            "style": "friendly"
        },
        "Strict Coach": {
            "system_instruction": "You are a disciplined nutrition coach focused on results and accountability. Be direct, honest, and focused on optimal health outcomes. Challenge users to make better choices.",
            "temperature": 0.3,
            "style": "strict"
        },
        "Fun Chef": {
            "system_instruction": "Think like a gourmet chef. Recommend creative, visually appealing, and culturally inspired meals. Include spices, plating tips, and elegant ingredients while keeping things nutritious.",
            "temperature": 0.8,
            "style": "playful"
        },
        "Mindful Coach": {
            "system_instruction": "Offer advice that combines nutrition with mindfulness and emotional well-being. Address nutrient deficiencies gently. Prioritize self-kindness, balance, and stress-free planning.",
            "temperature": 0.5,
            "style": "holistic"
        }
    }

def get_edamam_nutrition(food_item: str, quantity: str = "1 serving") -> Dict[str, Any]:
    """
    Get nutrition data from Edamam API
    """
    try:
        # Edamam API credentials from environment variables
        app_id = os.getenv("EDAMAM_APP_ID")
        app_key = os.getenv("EDAMAM_APP_KEY")
        
        if not app_id or not app_key:
            print(f"Missing Edamam API credentials: app_id={app_id}, app_key={app_key}")
            return None
        
        # Construct the query
        query = f"{quantity} {food_item}"
        print(f"Querying Edamam API for: {query}")
        
        # Edamam Nutrition Analysis API endpoint
        url = "https://api.edamam.com/api/nutrition-data"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "ingr": query
        }
        
        response = requests.get(url, params=params, timeout=10)
        print(f"Edamam API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Edamam API response data: {data}")
            
            # Check if we got valid nutrition data
            if data.get("calories", 0) > 0:
                result = {
                    "food": food_item,
                    "quantity": quantity,
                    "calories": round(data.get("calories", 0)),
                    "protein": round(data.get("totalNutrients", {}).get("PROCNT", {}).get("quantity", 0), 1),
                    "carbs": round(data.get("totalNutrients", {}).get("CHOCDF", {}).get("quantity", 0), 1),
                    "fat": round(data.get("totalNutrients", {}).get("FAT", {}).get("quantity", 0), 1),
                    "fiber": round(data.get("totalNutrients", {}).get("FIBTG", {}).get("quantity", 0), 1),
                    "sugar": round(data.get("totalNutrients", {}).get("SUGAR", {}).get("quantity", 0), 1),
                    "sodium": round(data.get("totalNutrients", {}).get("NA", {}).get("quantity", 0), 1),
                    "success": True,
                    "source": "Edamam API"
                }
                print(f"Edamam API result: {result}")
                return result
        else:
            print(f"Edamam API error: {response.status_code} - {response.text}")
        
        return None
        
    except Exception as e:
        print(f"Edamam API error: {e}")
        return None

def calculate_nutrition(food_item: str, quantity: str = "1 serving") -> Dict[str, Any]:
    """
    Calculate nutrition using Edamam API first, fallback to local database
    """
    # Try Edamam API first
    edamam_result = get_edamam_nutrition(food_item, quantity)
    if edamam_result:
        return edamam_result
    
    # Fallback to local database - expanded with more accurate nutrition data
    food_db = {
        "egg": {"calories": 70, "protein": 6, "carbs": 0.5, "fat": 5},
        "toast": {"calories": 80, "protein": 3, "carbs": 15, "fat": 1},
        "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3},
        "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fat": 0.4},
        "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
        "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
        "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
        "salmon": {"calories": 208, "protein": 22, "carbs": 0, "fat": 12},
        "broccoli": {"calories": 25, "protein": 3, "carbs": 5, "fat": 0.3},
        "bread": {"calories": 75, "protein": 2.5, "carbs": 14, "fat": 1},
        "pasta": {"calories": 220, "protein": 8, "carbs": 44, "fat": 1.5},
        "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fat": 10},
        "salad": {"calories": 20, "protein": 1.5, "carbs": 4, "fat": 0.2},
        "beef": {"calories": 250, "protein": 26, "carbs": 0, "fat": 17},
        "pork": {"calories": 242, "protein": 27, "carbs": 0, "fat": 14},
        "fish": {"calories": 206, "protein": 22, "carbs": 0, "fat": 12},
        "tuna": {"calories": 154, "protein": 25, "carbs": 0, "fat": 5},
        "turkey": {"calories": 135, "protein": 25, "carbs": 0, "fat": 3.2},
        "cheese": {"calories": 113, "protein": 7, "carbs": 1, "fat": 9},
        "milk": {"calories": 42, "protein": 3.4, "carbs": 5, "fat": 1},
        "yogurt": {"calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4},
        "oatmeal": {"calories": 68, "protein": 2.4, "carbs": 12, "fat": 1.4},
        "cereal": {"calories": 379, "protein": 8, "carbs": 84, "fat": 1.5},
        "orange": {"calories": 47, "protein": 0.9, "carbs": 12, "fat": 0.1},
        "grape": {"calories": 69, "protein": 0.6, "carbs": 16, "fat": 0.4},
        "carrot": {"calories": 25, "protein": 0.5, "carbs": 6, "fat": 0.1},
        "potato": {"calories": 77, "protein": 2, "carbs": 17, "fat": 0.1},
        "tomato": {"calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
        "lettuce": {"calories": 5, "protein": 0.5, "carbs": 1, "fat": 0.1},
        "spinach": {"calories": 7, "protein": 0.9, "carbs": 1.1, "fat": 0.1},
        "burger": {"calories": 354, "protein": 17, "carbs": 31, "fat": 17},
        "sandwich": {"calories": 300, "protein": 15, "carbs": 35, "fat": 12},
        "soup": {"calories": 85, "protein": 4, "carbs": 12, "fat": 2.5},
        "steak": {"calories": 271, "protein": 26, "carbs": 0, "fat": 19},
        "avocado": {"calories": 234, "protein": 2.9, "carbs": 12, "fat": 21}
    }
    
    # Simple parsing
    food_lower = food_item.lower()
    multiplier = 1
    
    # Extract quantity
    if "2" in quantity or "two" in food_lower:
        multiplier = 2
    elif "3" in quantity or "three" in food_lower:
        multiplier = 3
    elif "half" in food_lower or "0.5" in quantity:
        multiplier = 0.5
    
    # Find matching food
    for food, nutrition in food_db.items():
        if food in food_lower:
            return {
                "food": food_item,
                "quantity": quantity,
                "calories": round(nutrition["calories"] * multiplier),
                "protein": round(nutrition["protein"] * multiplier, 1),
                "carbs": round(nutrition["carbs"] * multiplier, 1),
                "fat": round(nutrition["fat"] * multiplier, 1),
                "success": True,
                "source": "Local Database"
            }
    
    # Smart estimation
    estimated_nutrition = {"calories": 150, "protein": 8.0, "carbs": 20.0, "fat": 5.0}
    
    # Adjust estimates based on food type
    if any(word in food_lower for word in ['meat', 'chicken', 'beef', 'pork', 'fish']):
        estimated_nutrition = {"calories": 200, "protein": 25.0, "carbs": 2.0, "fat": 8.0}
    elif any(word in food_lower for word in ['fruit', 'apple', 'orange', 'berry']):
        estimated_nutrition = {"calories": 80, "protein": 1.0, "carbs": 20.0, "fat": 0.5}
    elif any(word in food_lower for word in ['vegetable', 'broccoli', 'carrot', 'spinach']):
        estimated_nutrition = {"calories": 30, "protein": 2.0, "carbs": 6.0, "fat": 0.3}
    elif any(word in food_lower for word in ['bread', 'pasta', 'rice', 'grain']):
        estimated_nutrition = {"calories": 180, "protein": 6.0, "carbs": 35.0, "fat": 2.0}
    
    return {
        "food": food_item,
        "quantity": quantity,
        "calories": round(estimated_nutrition["calories"] * multiplier),
        "protein": round(estimated_nutrition["protein"] * multiplier, 1),
        "carbs": round(estimated_nutrition["carbs"] * multiplier, 1),
        "fat": round(estimated_nutrition["fat"] * multiplier, 1),
        "success": False,
        "source": "Estimated",
        "note": "Estimated values based on food category"
    }

def check_nutrition_safety(text: str) -> tuple[bool, List[str]]:
    """Check for unsafe nutrition advice or harmful content with enhanced pattern matching and AI sentiment analysis"""
    unsafe_patterns = [
        # Extreme dieting
        "extreme diet", "crash diet", "starvation", "no food", "skip meals", "fast for days",
        "eat nothing", "stop eating", "starve yourself", "severe calorie restriction",
        
        # Harmful substances
        "dangerous", "harmful", "toxic", "poison", "laxatives", "diet pills", "weight loss pills",
        "appetite suppressants", "fat burners", "detox tea", "cleanse pills",
        
        # Eating disorders
        "eating disorder", "anorexia", "bulimia", "binge eating", "purging", "vomiting",
        "pro ana", "pro mia", "thinspo", "skinny goals",
        
        # Dangerous behaviors
        "self harm", "hurt yourself", "punish yourself", "exercise until exhaustion",
        "workout punishment", "food punishment", "guilt eating",
        
        # Medical misinformation
        "cure diabetes", "cure cancer", "magic weight loss", "miracle diet",
        "lose 20 pounds in a week", "instant results", "no exercise needed",
        
        # Inappropriate content
        "sexual", "explicit", "violence", "hate", "discrimination", "racist",
        "suicide", "death", "kill", "murder"
    ]
    
    flagged = []
    text_lower = text.lower()
    
    # Pattern matching
    for pattern in unsafe_patterns:
        if pattern in text_lower:
            flagged.append(pattern)
    
    # Advanced AI-based safety checks
    if SENTIMENT_AVAILABLE and len(text.strip()) > 10:
        try:
            # Toxicity detection
            toxicity_results = toxicity_classifier(text)
            if toxicity_results and len(toxicity_results[0]) > 0:
                toxic_score = toxicity_results[0][0]['score'] if toxicity_results[0][0]['label'] == 'TOXIC' else toxicity_results[0][1]['score']
                if toxic_score > 0.7:  # High toxicity threshold
                    flagged.append("AI-detected toxic content")
            
            # Negative sentiment detection for eating disorder patterns
            sentiment_result = sentiment_analyzer(text)
            if sentiment_result[0]['label'] == 'NEGATIVE' and sentiment_result[0]['score'] > 0.9:
                # Check if highly negative content contains food/body related terms
                body_terms = ["body", "weight", "fat", "skinny", "food", "eat", "diet", "calories"]
                if any(term in text_lower for term in body_terms):
                    flagged.append("AI-detected harmful body/food negativity")
                    
        except Exception as e:
            # Fallback to pattern matching if AI models fail
            pass
    
    return len(flagged) == 0, flagged

def text_to_speech(text: str, voice: str = 'en-US-AriaNeural') -> bytes:
    """Convert text to speech using Edge TTS and return audio bytes"""
    if not TTS_AVAILABLE:
        return None
    
    try:
        # Clean text for TTS
        clean_text = re.sub(r'\*\*.*?\*\*', '', text)  # Remove bold markdown
        clean_text = re.sub(r'\*.*?\*', '', clean_text)  # Remove italic markdown
        clean_text = re.sub(r'[#•\-]', '', clean_text)  # Remove special characters
        clean_text = clean_text.replace('\n', ' ').strip()
        
        # Limit text length
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        if len(clean_text.strip()) < 5:
            return None
        
        # Run async TTS generation
        async def generate_speech():
            communicate = edge_tts.Communicate(clean_text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_bytes = loop.run_until_complete(generate_speech())
        loop.close()
        
        return audio_bytes if audio_bytes else None
            
    except Exception as e:
        return None

def generate_nutrition_response(
    prompt: str,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.6,
    max_tokens: int = 1024,
    system_instruction: str = None,
    image_data: bytes = None,
    stream: bool = False
):
    """Generate nutrition-focused responses using Gemini"""
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        # Add nutrition context to all prompts
        nutrition_context = "\n\nRemember: Always prioritize health and safety. If unsure about medical conditions, recommend consulting healthcare professionals."
        enhanced_prompt = prompt + nutrition_context
        
        if image_data:
            model = genai.GenerativeModel(model_name)
            image = Image.open(io.BytesIO(image_data))
            
            if stream:
                response = model.generate_content(
                    [enhanced_prompt, image],
                    generation_config=generation_config,
                    stream=True
                )
            else:
                response = model.generate_content(
                    [enhanced_prompt, image],
                    generation_config=generation_config
                )
        else:
            model_config = {"model_name": model_name}
            if system_instruction:
                model_config["system_instruction"] = system_instruction
            
            model = genai.GenerativeModel(**model_config)
            
            if stream:
                response = model.generate_content(
                    enhanced_prompt,
                    generation_config=generation_config,
                    stream=True
                )
            else:
                response = model.generate_content(
                    enhanced_prompt,
                    generation_config=generation_config
                )
        
        return response
        
    except Exception as e:
        st.error(f"🚨 Aafiya AI Error: {str(e)}")
        return None

def check_profile_complete() -> bool:
    """Check if user profile is complete with all required fields"""
    profile = st.session_state.user_profile
    return (
        profile.get('age') is not None and
        profile.get('gender') is not None and
        profile.get('weight') is not None and
        profile.get('height') is not None and
        profile.get('activity_level') is not None and
        profile.get('goal') is not None and
        profile.get('age') >= 14 and
        profile.get('weight') > 0 and
        profile.get('height') > 0
    )

def calculate_bmi(weight: float, height: int) -> float:
    """Calculate BMI from weight (kg) and height (cm)"""
    height_m = height / 100  # Convert cm to meters
    return weight / (height_m ** 2)

def generate_healthy_food_image(craving_text: str) -> str:
    """
    Generate a healthy food alternative image using Pollinations API
    Returns: URL of the generated image
    """
    try:
        # Create a prompt for healthy alternatives
        healthy_prompt = f"A beautiful, colorful, healthy plate of food that satisfies someone craving {craving_text}. Fresh ingredients, vibrant colors, nutritious alternatives, appetizing presentation, food photography style, well-lit, professional quality"
        
        # URL encode the prompt
        import urllib.parse
        encoded_prompt = urllib.parse.quote(healthy_prompt)
        
        # Pollinations API URL
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&seed=42"
        
        return image_url
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def get_color_nutrition_tags(craving_text: str) -> List[Dict[str, str]]:
    """
    Generate color-coded nutrition tags based on craving
    Returns: List of color tags with nutritional benefits
    """
    # Common healthy food colors and their benefits
    color_tags = []
    
    craving_lower = craving_text.lower()
    
    # Add relevant color tags based on craving
    if any(word in craving_lower for word in ['sweet', 'sugar', 'dessert', 'candy']):
        color_tags.extend([
            {"color": "🟣", "food": "Blueberries", "benefit": "Antioxidants & natural sweetness"},
            {"color": "🟠", "food": "Sweet Potato", "benefit": "Complex carbs & beta carotene"},
            {"color": "🔴", "food": "Strawberries", "benefit": "Vitamin C & fiber"}
        ])
    
    if any(word in craving_lower for word in ['fries', 'chips', 'salty', 'crispy']):
        color_tags.extend([
            {"color": "🟠", "food": "Roasted Carrots", "benefit": "Beta carotene & fiber"},
            {"color": "🟡", "food": "Air-fried Squash", "benefit": "Vitamins A & C"},
            {"color": "🟢", "food": "Kale Chips", "benefit": "Iron & vitamin K"}
        ])
    
    if any(word in craving_lower for word in ['tired', 'energy', 'fatigue']):
        color_tags.extend([
            {"color": "🟫", "food": "Almonds", "benefit": "Healthy fats & protein"},
            {"color": "🟢", "food": "Spinach", "benefit": "Iron & B vitamins"},
            {"color": "🟡", "food": "Bananas", "benefit": "Potassium & natural energy"}
        ])
    
    if any(word in craving_lower for word in ['anti-inflammatory', 'inflammation', 'joint']):
        color_tags.extend([
            {"color": "🟠", "food": "Turmeric", "benefit": "Curcumin anti-inflammatory"},
            {"color": "🟢", "food": "Leafy Greens", "benefit": "Antioxidants & omega-3s"},
            {"color": "🔴", "food": "Berries", "benefit": "Anthocyanins & vitamin C"}
        ])
    
    # Default colorful tags if no specific match
    if not color_tags:
        color_tags = [
            {"color": "🟢", "food": "Leafy Greens", "benefit": "Vitamins & minerals"},
            {"color": "🟠", "food": "Orange Vegetables", "benefit": "Beta carotene & fiber"},
            {"color": "🔴", "food": "Red Fruits", "benefit": "Antioxidants & vitamin C"},
            {"color": "🟣", "food": "Purple Foods", "benefit": "Anthocyanins & brain health"}
        ]
    
    return color_tags[:4]  # Return max 4 tags

def extract_nutrition_data(text: str) -> Dict[str, Any]:
    """Extract structured nutrition data from AI response - NO NULL VALUES"""
    # Enhanced regex patterns for better extraction
    calories_match = re.search(r'[~]?(\d+)\s*(?:cal|kcal|calories)', text.lower())
    protein_match = re.search(r'[~]?(\d+(?:\.\d+)?)\s*g?\s*(?:protein|pro)', text.lower())
    carbs_match = re.search(r'[~]?(\d+(?:\.\d+)?)\s*g?\s*(?:carb|carbohydrate|carbs)', text.lower())
    fat_match = re.search(r'[~]?(\d+(?:\.\d+)?)\s*g?\s*(?:fat|fats)', text.lower())
    
    # Also try to extract from structured format like "Calories: 450"
    if not calories_match:
        calories_match = re.search(r'calories[:\s]+[~]?(\d+)', text.lower())
    if not protein_match:
        protein_match = re.search(r'protein[:\s]+[~]?(\d+(?:\.\d+)?)', text.lower())
    if not carbs_match:
        carbs_match = re.search(r'carbs?[:\s]+[~]?(\d+(?:\.\d+)?)', text.lower())
    if not fat_match:
        fat_match = re.search(r'fat[:\s]+[~]?(\d+(?:\.\d+)?)', text.lower())
    
    # Always return estimated values - NO NULLS
    extracted_data = {
        "calories": int(calories_match.group(1)) if calories_match else 200,
        "protein": float(protein_match.group(1)) if protein_match else 10.0,
        "carbs": float(carbs_match.group(1)) if carbs_match else 25.0,
        "fat": float(fat_match.group(1)) if fat_match else 8.0,
        "source": "Extracted from AI response"
    }
    
    return extracted_data

def filter_recipe_for_user_safety(recipe_text, user_allergies_str, unpreferred_foods_list):
    """
    🎯 Filter recipe based on user allergies and preferences
    Returns: (is_safe, filtered_content, warnings)
    """
    warnings = []
    is_safe = True
    
    # Convert allergies string to list
    if user_allergies_str:
        user_allergies = [allergy.strip().lower() for allergy in user_allergies_str.split(',')]
    else:
        user_allergies = []
    
    # Convert unpreferred foods to lowercase
    unpreferred_foods = [food.strip().lower() for food in unpreferred_foods_list] if unpreferred_foods_list else []
    
    # Check for allergens in recipe text
    recipe_lower = recipe_text.lower()
    detected_allergens = []
    
    for allergy in user_allergies:
        if allergy in recipe_lower:
            detected_allergens.append(allergy)
            is_safe = False
    
    # Check for unpreferred foods
    detected_unpreferred = []
    for food in unpreferred_foods:
        if food in recipe_lower:
            detected_unpreferred.append(food)
    
    # Generate warnings
    if detected_allergens:
        warnings.append(f"⚠️ **ALLERGY WARNING**: This recipe contains: {', '.join(detected_allergens)}")
    
    if detected_unpreferred:
        warnings.append(f"ℹ️ **Note**: This recipe contains foods you prefer to avoid: {', '.join(detected_unpreferred)}")
    
    # Create filtered content with warnings
    filtered_content = recipe_text
    if warnings:
        warning_text = "\n\n" + "\n".join(warnings)
        if not is_safe:
            warning_text += "\n\n🚨 **Please consult with a healthcare provider before consuming foods you're allergic to.**"
        filtered_content = warning_text + "\n\n" + recipe_text
    
    return is_safe, filtered_content, warnings

def save_user_recipe_list(recipe_content, recipe_title="Custom Recipe", nutrition_data=None):
    """Save user's filtered recipe to session state"""
    if 'user_recipe_list' not in st.session_state:
        st.session_state.user_recipe_list = []
    
    # Create recipe entry
    recipe_entry = {
        'title': recipe_title,
        'content': recipe_content,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'allergies': st.session_state.user_profile.get('allergies', ''),
        'unpreferred_foods': st.session_state.user_profile.get('unpreferred_foods', []),
        'nutrition': nutrition_data if nutrition_data else None
    }
    
    # Add to list (keep last 10 recipes)
    st.session_state.user_recipe_list.insert(0, recipe_entry)
    if len(st.session_state.user_recipe_list) > 10:
        st.session_state.user_recipe_list = st.session_state.user_recipe_list[:10]
    
    return len(st.session_state.user_recipe_list)

def process_nutrition_request(prompt_to_use, uploaded_image_data, selected_persona, persona_config, temperature, max_tokens, enable_nutrition_calculator, enable_streaming, enable_meal_logging, safety_level, request_type, enable_tts=False, selected_voice="Avatar", selected_voice_name="Salma (Professional Nutritionist)", avatar_style="Professional Nutritionist", avatar_response_length="Detailed", enable_avatar=True):
    """Process a nutrition request and display results"""
    
    # Enhanced safety check with AI analysis
    is_safe, flagged_terms = check_nutrition_safety(prompt_to_use)
    if not is_safe:
        st.error(f"🚨 Content flagged for safety: {', '.join(flagged_terms)}")
        st.info("🛡️ This content was flagged by our advanced safety system to prevent potentially harmful nutrition advice.")
        if SENTIMENT_AVAILABLE:
            st.info("✨ Enhanced protection enabled with AI-powered content analysis")
        
        # FORCE SAVE CHAT HISTORY EVEN FOR FLAGGED CONTENT
        flagged_chat_entry = {
            "prompt": prompt_to_use,
            "response": "Content flagged for safety and blocked",
            "persona": selected_persona,
            "temperature": temperature,
            "function_result": None,
            "nutrition_data": None,
            "timestamp": time.time(),
            "has_image": uploaded_image_data is not None,
            "request_type": request_type
        }
        st.session_state.chat_history.append(flagged_chat_entry)
        st.success(f"💾 Chat saved! Total conversations: {len(st.session_state.chat_history)}")
        return
    
    # Add user context to prompt
    profile = st.session_state.user_profile
    
    # Enhance prompt with RAG if documents are available
    documents = st.session_state.get('nutrition_documents', [])
    if documents:
        display_rag_info(documents, prompt_to_use)
        enhanced_prompt = enhance_prompt_with_rag(prompt_to_use, documents)
    else:
        enhanced_prompt = prompt_to_use
    
    # Get user allergies and dietary preferences
    user_allergies = profile.get('allergies', '')
    unpreferred_foods = profile.get('unpreferred_foods', [])
    health_issues = profile.get('health_issues', '')
    
    context_prompt = f"""
    User Profile Context:
    - Age: {profile['age']} years
    - Gender: {profile['gender']}
    - Weight: {profile['weight']} kg
    - Height: {profile['height']} cm
    - Activity Level: {profile['activity_level']}
    - Goal: {profile['goal']}
    - Allergies: {user_allergies if user_allergies else 'None'}
    - Unpreferred Foods: {', '.join(unpreferred_foods) if unpreferred_foods else 'None'}
    - Health Issues: {health_issues if health_issues else 'None'}
    
    Enhanced User Question: {enhanced_prompt}
    
    🎯 CRITICAL SAFETY INSTRUCTIONS:
    You are a certified nutritionist. When providing recipes, meal plans, or food recommendations:
    
    1. ALWAYS avoid foods the user is allergic to - this is a safety requirement
    2. Try to minimize or avoid foods they dislike (unpreferred foods)
    3. Consider their health goals and any medical conditions
    4. If asked for a recipe that contains allergens, suggest safe alternatives
    5. Always mention if a recipe has been modified for allergies/preferences
    
    📊 NUTRITION ANALYSIS INSTRUCTIONS:
    When analyzing food images, ALWAYS provide approximate nutrition values including:
    - Approximate calories (e.g., "~450 calories")
    - Approximate protein (e.g., "~25g protein") 
    - Approximate carbs (e.g., "~55g carbs")
    - Approximate fat (e.g., "~12g fat")
    
    Use your nutritional knowledge to estimate values based on typical ingredients and portion sizes visible in the image.
    
    Provide advice with empathy and accuracy, considering the user's complete profile and any provided document context.
    """
    
    # Stage 3: Function Calling for Nutrition Calculator
    function_result = None
    if enable_nutrition_calculator:
        # Always calculate nutrition for image analysis, or when user asks about nutrition facts
        should_calculate = (request_type == "image") or any(keyword in prompt_to_use.lower() for keyword in ['calorie', 'calories', 'nutrition', 'macro', 'protein', 'carb', 'fat', 'ate', 'eating', 'food'])
        
        if should_calculate:
            # Try to extract food items for calculation - enhanced food detection
            food_keywords = ['egg', 'toast', 'apple', 'banana', 'chicken', 'rice', 'salmon', 'broccoli', 'bread', 'pasta', 'pizza', 'salad',
                           'beef', 'pork', 'fish', 'tuna', 'turkey', 'cheese', 'milk', 'yogurt', 'oatmeal', 'cereal', 'orange', 'grape',
                           'carrot', 'potato', 'tomato', 'lettuce', 'spinach', 'burger', 'sandwich', 'soup', 'steak', 'avocado']
            
            # Look for food items in the prompt
            detected_foods = []
            for food in food_keywords:
                if food in prompt_to_use.lower():
                    detected_foods.append(food)
            
            # If no specific food found, try to extract from the entire prompt
            if not detected_foods:
                if request_type == "image":
                    # For image analysis, use a generic description
                    detected_foods = ["pasta dish with sauce"]
                else:
                    # Use the entire prompt as food item for Edamam API
                    detected_foods = [prompt_to_use.strip()[:100]]  # Limit prompt length
            
            # Calculate nutrition for detected foods
            if detected_foods:
                st.info("🧮 Aafiya is calculating nutrition data...")
                # Use the first detected food or the whole prompt
                food_item = detected_foods[0]
                
                # Clean up the food item string
                food_item = food_item.replace("what are the nutrition facts for", "").replace("calories in", "").replace("nutrition info for", "").strip()
                
                function_result = calculate_nutrition(food_item, "1 serving")
    
    try:
        # Generate response
        model_name = "gemini-1.5-flash"
        image_data = uploaded_image_data.read() if uploaded_image_data else None
        
        if enable_streaming:
            response_container = st.empty()
            response_text = ""
            
            with st.spinner("Aafiya is generating nutrition advice..."):
                response = generate_nutrition_response(
                    context_prompt,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=persona_config['system_instruction'],
                    image_data=image_data,
                    stream=True
                )
                
                if response:
                    for chunk in response:
                        if hasattr(chunk, 'text'):
                            response_text += chunk.text
                            response_container.markdown(response_text)
                    
                    # 🎯 Filter recipe for user safety
                    user_allergies = st.session_state.user_profile.get('allergies', '')
                    unpreferred_foods = st.session_state.user_profile.get('unpreferred_foods', [])
                    
                    # Apply filtering if response contains recipe content
                    if any(keyword in response_text.lower() for keyword in ['recipe', 'ingredients', 'meal plan', 'breakfast', 'lunch', 'dinner']):
                        is_safe, filtered_response, warnings = filter_recipe_for_user_safety(
                            response_text, user_allergies, unpreferred_foods
                        )
                        
                        # Update the displayed response with filtered content
                        response_container.markdown(filtered_response)
                        
                        # Show filtering status
                        if warnings:
                            st.divider()
                            st.subheader("🎯 Recipe Safety Check")
                            for warning in warnings:
                                if "ALLERGY WARNING" in warning:
                                    st.error(warning)
                                else:
                                    st.info(warning)
                        
                        # Recipe saving disabled - focusing on meal logging instead
                        # if is_safe or st.session_state.get('save_unsafe_recipes', False):
                        #     recipe_count = save_user_recipe_list(
                        #         filtered_response, 
                        #         f"Recipe from {time.strftime('%Y-%m-%d %H:%M')}",
                        #         function_result
                        #     )
                        #     st.success(f"✅ Recipe saved to your list! ({recipe_count} total recipes)")
                        
                        # Store filtered response for Salma integration
                        st.session_state.latest_ai_response = filtered_response
                        final_response = filtered_response
                    else:
                        # Store regular response for Salma integration
                        st.session_state.latest_ai_response = response_text
                        final_response = response_text
                    
                    
                    # Add Interactive Streaming Avatar Response
                    if enable_avatar and len(response_text.strip()) > 10:
                        st.subheader("🎥 Chat with Salma")
                        
                        col1, col2, col3 = st.columns([2, 2, 2])
                        
                        with col1:
                            if st.button("💬 Send to Salma", key=f"send_to_avatar_stream_{request_type}", type="primary", use_container_width=True):
                                if HEYGEN_AVAILABLE:
                                    with st.spinner("📤 Sending message to Salma..."):
                                        # Store the response for potential sending to avatar
                                        st.session_state[f"latest_response_{request_type}"] = response_text
                                        
                                        # JavaScript to send message to both avatars (floating + embedded)
                                        st.markdown(f"""
                                        <script>
                                        const message = `Here's the nutrition advice: {response_text[:500].replace('"', "'")}`;
                                        
                                        // Send to floating avatar
                                        if (window.heygenStreamingAPI) {{
                                            window.heygenStreamingAPI.sendMessage(message);
                                            window.heygenStreamingAPI.show();
                                        }}
                                        
                                        // Send to embedded avatar in right panel
                                        const embeddedIframe = document.querySelector('iframe[title="Salma - AI Nutritionist"]');
                                        if (embeddedIframe && embeddedIframe.contentWindow) {{
                                            embeddedIframe.contentWindow.postMessage({{
                                                type: 'send-message',
                                                message: message
                                            }}, '*');
                                        }}
                                        </script>
                                        """, unsafe_allow_html=True)
                                        
                                        st.success("✅ Message sent to Salma! She should respond shortly.")
                                else:
                                    st.error("❌ HeyGen Avatar not available")
                        
                        with col2:
                            if st.button("📹 Open Chat", key=f"open_avatar_stream_{request_type}", type="secondary", use_container_width=True):
                                st.markdown("""
                                <script>
                                if (window.heygenStreamingAPI) {
                                    window.heygenStreamingAPI.show();
                                }
                                </script>
                                """, unsafe_allow_html=True)
                                st.success("Salma's chat opened!")
                        
                        with col3:
                            st.info(f"🎭 {selected_voice_name}")
                        
                        # Display streaming avatar instructions
                        st.markdown("""
                        <div class="avatar-status" style="color: #000000;">
                            <strong style="color: #000000;">💬 How to Chat with Salma:</strong><br>
                            <span style="color: #000000;">1. <strong style="color: #000000;">Send to Salma:</strong> Sends this nutrition advice to her directly</span><br>
                            <span style="color: #000000;">2. <strong style="color: #000000;">Open Chat:</strong> Expands Salma's video chat interface</span><br>
                            <span style="color: #000000;">3. <strong style="color: #000000;">Voice Chat:</strong> Speak directly to Salma using your microphone</span><br>
                            <span style="color: #000000;">4. <strong style="color: #000000;">Follow-up:</strong> Ask Salma additional nutrition questions live</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif enable_avatar and not HEYGEN_AVAILABLE:
                        st.subheader("🤖 Avatar Response")
                        st.warning("🔇 Interactive avatar unavailable")
                        st.info("💡 Avatar integration requires HeyGen streaming embed")
                    elif not enable_avatar:
                        st.info("🤖 Interactive avatar disabled - Enable in sidebar for avatar responses")
        else:
            with st.spinner("🤖 Aafiya is analyzing your nutrition question..."):
                response = generate_nutrition_response(
                    context_prompt,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=persona_config['system_instruction'],
                    image_data=image_data
                )
                
                if response:
                    response_text = response.text
                    st.markdown(response_text)
                    
                    # Store latest response for Salma integration
                    st.session_state.latest_ai_response = response_text
                    
                    # Add Interactive Streaming Avatar Response
                    if enable_avatar and len(response_text.strip()) > 10:
                        st.subheader("🎥 Chat with Salma")
                        
                        # Avatar integration message
                        st.info("💬 Chat with Salma in the right panel for interactive nutrition guidance!")
                    
                    elif enable_avatar and not HEYGEN_AVAILABLE:
                        st.subheader("🤖 Avatar Response")
                        st.warning("🔇 Interactive avatar unavailable")
                        st.info("💡 Avatar integration requires HeyGen streaming embed")
                    elif not enable_avatar:
                        st.info("🤖 Interactive avatar disabled - Enable in sidebar for avatar responses")
                    
                else:
                    response_text = "Sorry, I couldn't generate a response. Please try again."
                    st.error(response_text)
        
        # Recipe filtering and saving for non-streaming responses
        if response_text and any(keyword in response_text.lower() for keyword in ['recipe', 'ingredients', 'meal plan', 'breakfast', 'lunch', 'dinner']):
            is_safe = filter_recipe_for_user_safety(response_text)
            
            # Recipe saving disabled - focusing on meal logging instead
            # if is_safe or st.session_state.get('save_unsafe_recipes', False):
            #     recipe_count = save_user_recipe_list(
            #         recipe_title=f"Recipe from {selected_persona}",
            #         recipe_content=response_text,
            #         nutrition_data=function_result or extracted_nutrition
            #     )
            #     
            #     if recipe_count > 0:
            #         st.success(f"🍽️ Recipe saved to your collection! Total: {recipe_count} recipes")
            # else:
            #     st.warning("⚠️ Recipe contains ingredients that may not be suitable for your dietary restrictions. Recipe not saved automatically.")
            #     if st.button("Save Recipe Anyway", key="save_unsafe_recipe"):
            #         st.session_state.save_unsafe_recipes = True
            #         st.rerun()
        
        # Display Function Calling Results
        if function_result:
            st.subheader("📊 Approximate Nutrition Values")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Calories (approx)", f"~{function_result['calories']}")
            with col2:
                st.metric("Protein (approx)", f"~{function_result['protein']}g")
            with col3:
                st.metric("Carbs (approx)", f"~{function_result['carbs']}g")
            with col4:
                st.metric("Fat (approx)", f"~{function_result['fat']}g")
            
            # Show additional nutrients if available (from Edamam API)
            if 'fiber' in function_result or 'sugar' in function_result or 'sodium' in function_result:
                st.write("**Additional Approximate Nutrients:**")
                add_col1, add_col2, add_col3 = st.columns(3)
                with add_col1:
                    if 'fiber' in function_result:
                        st.metric("Fiber (approx)", f"~{function_result['fiber']}g")
                with add_col2:
                    if 'sugar' in function_result:
                        st.metric("Sugar (approx)", f"~{function_result['sugar']}g")
                with add_col3:
                    if 'sodium' in function_result:
                        st.metric("Sodium (approx)", f"~{function_result['sodium']}mg")
            
            # Show approximate disclaimer
            st.info("📊 These are approximate nutrition values based on food analysis. Actual values may vary based on preparation method, portion size, and specific ingredients.")
            
            with st.expander("🔍 Detailed Nutrition Data"):
                st.json(function_result)
            
            # Add to nutrition tracking
            st.session_state.nutrition_data.append(function_result)
        
        # Extract structured data from response (always complete values)
        extracted_nutrition = {}
        if 'response_text' in locals() and response_text:
            extracted_nutrition = extract_nutrition_data(response_text)
            if extracted_nutrition and any(extracted_nutrition.get(key, 0) != 0 for key in ['calories', 'protein', 'carbs', 'fat']):
                st.info(f"🔍 Nutrition extracted from AI response: {extracted_nutrition.get('calories', 0)} cal, {extracted_nutrition.get('protein', 0)}g protein")
        
        # Save to chat history - ALWAYS save the conversation
        final_response = response_text if 'response_text' in locals() else "No response generated"
        chat_entry = {
            "prompt": prompt_to_use,
            "response": final_response,
            "persona": selected_persona,
            "temperature": temperature,
            "function_result": function_result,
            "nutrition_data": extracted_nutrition,
            "timestamp": time.time(),
            "has_image": uploaded_image_data is not None,
            "request_type": request_type
        }
        st.session_state.chat_history.append(chat_entry)
        
        # Debug info for chat saving
        st.success(f"💾 Chat saved! Total conversations: {len(st.session_state.chat_history)}")
        
        # Add to meal log if relevant - expanded food keywords and nutrition-based detection
        food_keywords = ['ate', 'meal', 'breakfast', 'lunch', 'dinner', 'snack', 'food', 'eating', 'consumed', 'had', 'calories', 'nutrition', 'protein', 'carbs', 'fat', 'drink', 'beverage', 'recipe', 'ingredient', 'cooking', 'cooked', 'prepared', 'portion', 'serving']
        
        has_food_keywords = any(food in prompt_to_use.lower() for food in food_keywords)
        has_nutrition_data = function_result or extracted_nutrition
        
        if enable_meal_logging and (uploaded_image_data or has_food_keywords or has_nutrition_data):
            meal_entry = {
                "food": prompt_to_use[:100] + "..." if len(prompt_to_use) > 100 else prompt_to_use,
                "time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "nutrition": function_result if function_result else extracted_nutrition,
                "ai_advice": final_response[:200] + "..." if len(final_response) > 200 else final_response
            }
            st.session_state.meal_log.append(meal_entry)
            st.success(f"🍽️ Meal logged! Total meals tracked: {len(st.session_state.meal_log)}")
        else:
            if enable_meal_logging:
                # Debug message when meal logging is enabled but conditions not met
                st.info("ℹ️ Meal logging enabled but no food-related keywords detected. Try mentioning food, meals, or nutrition.")
        
        # Display token usage if available
        if 'response' in locals() and hasattr(response, 'usage_metadata'):
            st.success(f"📊 Tokens used: {response.usage_metadata.total_token_count}")
        else:
            # Estimate token usage
            estimated_tokens = len(prompt_to_use.split()) + len(response_text.split()) if 'response_text' in locals() else 0
            st.info(f"📊 Estimated tokens: ~{estimated_tokens}")
        
        # Clear image after processing if it was an image request
        if uploaded_image_data and request_type == "image":
            st.session_state.clear_image = True
            
    except Exception as e:
        # FORCE SAVE CHAT HISTORY EVEN WHEN EXCEPTIONS OCCUR (silently)
        error_chat_entry = {
            "prompt": prompt_to_use,
            "response": "Sorry, there was an issue processing your request. Please try again.",
            "persona": selected_persona,
            "temperature": temperature,
            "function_result": None,
            "nutrition_data": None,
            "timestamp": time.time(),
            "has_image": uploaded_image_data is not None,
            "request_type": request_type
        }
        st.session_state.chat_history.append(error_chat_entry)
        st.success(f"💾 Chat saved! Total conversations: {len(st.session_state.chat_history)}")

def cravesmart_page():
    """CraveSmart - Transform Your Cravings page"""
    
    # Apply theme CSS
    white_mode = st.session_state.get('white_mode', False)
    st.markdown(get_theme_css(white_mode), unsafe_allow_html=True)
    
    # Header with back button
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back", key="back_to_main"):
            st.session_state.current_page = "main"
            st.rerun()
    
    with col2:
        st.title("🍩 CraveSmart - Transform Your Cravings")
    
    st.markdown("""
    **🧠 Smart Craving Transformation**: Turn your cravings into colorful, nutritious alternatives!
    
    Type what you're craving, and get a visual plate of healthier options that satisfy both your taste buds and nutritional needs.
    """)
    
    # Main craving input
    st.subheader("📝 What are you craving?")
    
    # Example prompts
    st.markdown("**💡 Try these examples:**")
    example_buttons = st.columns(4)
    
    with example_buttons[0]:
        if st.button("🍟 I'm craving fries", key="example_fries"):
            st.session_state.craving_input = "fries"
    
    with example_buttons[1]:
        if st.button("🍰 Something sweet", key="example_sweet"):
            st.session_state.craving_input = "something sweet and dessert-like"
    
    with example_buttons[2]:
        if st.button("🍫 Chocolate craving", key="example_chocolate"):
            st.session_state.craving_input = "chocolate"
    
    with example_buttons[3]:
        if st.button("😴 I feel tired", key="example_tired"):
            st.session_state.craving_input = "I feel tired and want something energizing"
    
    # Craving input
    craving_text = st.text_area(
        "Describe your craving, mood, or health goal:",
        value=st.session_state.get('craving_input', ''),
        height=100,
        placeholder="E.g., 'I'm craving fries' or 'I want a colorful anti-inflammatory breakfast' or 'I feel tired and want something sweet'",
        key="craving_main_input"
    )
    
    # Generate button
    if st.button("🎨 Transform My Craving!", type="primary", use_container_width=True):
        if craving_text.strip():
            with st.spinner("🎨 Creating your healthy visual alternative..."):
                # Generate healthy food image
                image_url = generate_healthy_food_image(craving_text)
                
                if image_url:
                    # Display the generated image
                    st.subheader("🖼️ Your Healthy Alternative")
                    st.image(image_url, caption=f"Healthy alternative for: {craving_text}", width=512)
                    
                    # Generate and display color nutrition tags
                    color_tags = get_color_nutrition_tags(craving_text)
                    
                    st.subheader("🎨 Color Plate Nutrition Power")
                    st.markdown("**Why color diversity = nutritional power:**")
                    
                    # Display color tags in a nice grid
                    tag_cols = st.columns(2)
                    for i, tag in enumerate(color_tags):
                        with tag_cols[i % 2]:
                            st.markdown(f"""
                            **{tag['color']} {tag['food']}**  
                            {tag['benefit']}
                            """)
                    
                    # Nutritional tip
                    st.info("💡 **Pro Tip**: The more colors on your plate, the more diverse nutrients you're getting! Each color represents different vitamins, minerals, and antioxidants.")
                    
                    # Save to favorites option
                    if st.button("💾 Save to My Healthy Alternatives", key="save_alternative"):
                        if 'saved_alternatives' not in st.session_state:
                            st.session_state.saved_alternatives = []
                        
                        alternative_entry = {
                            'craving': craving_text,
                            'image_url': image_url,
                            'color_tags': color_tags,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        st.session_state.saved_alternatives.append(alternative_entry)
                        st.success("✅ Saved to your healthy alternatives!")
        else:
            st.warning("⚠️ Please describe what you're craving first!")
    
    # Show saved alternatives
    if 'saved_alternatives' in st.session_state and st.session_state.saved_alternatives:
        st.divider()
        st.subheader("💾 Your Saved Healthy Alternatives")
        
        for i, alternative in enumerate(reversed(st.session_state.saved_alternatives[-5:])):  # Show last 5
            with st.expander(f"🍽️ {alternative['craving'][:50]}... - {alternative['timestamp']}"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(alternative['image_url'], width=200)
                
                with col2:
                    st.markdown("**🎨 Nutrition Colors:**")
                    for tag in alternative['color_tags']:
                        st.markdown(f"**{tag['color']} {tag['food']}**: {tag['benefit']}")
    
    # Credit line at the end of CraveSmart page
    st.divider()
    st.markdown(
        "<div style='text-align: center; margin-top: 2rem; padding: 1rem; color: #666; font-size: 0.9em;'>"
        "Created by Nourah Alotaibi"
        "</div>",
        unsafe_allow_html=True
    )

def main():
    """Main Aafiya AI application"""
    
    if not configure_gemini():
        return
    
    initialize_session_state()
    
    # TTS cleanup removed
    
    # Handle page routing
    current_page = st.session_state.get('current_page', 'main')
    
    if current_page == 'cravesmart':
        cravesmart_page()
        return
    
    # Clear image if flag is set
    if st.session_state.clear_image:
        st.session_state.clear_image = False
        st.rerun()
    
    # Hero Banner with integrated title and image
    st.markdown("""
    <div class="hero-banner" style="margin-bottom: 2.3rem;">
        <h1>🥗 Aafiya AI: Your AI-Powered Wellness Companion</h1>
        <p>Personalized nutrition guidance at your fingertips</p>
        <div style="text-align: center; margin-top: 0.5rem;">
            <img src="https://www.paho.org/sites/default/files/styles/top_hero/public/2023-08/nutrition-topic-banner-1500x750_0.jpg?h=807b420e&itok=D9rlaEq-" 
                 class="nutrition-image" 
                 style="width: 95%; height: 160px; object-fit: cover; border-radius: 12px;" 
                 alt="Nutrition Banner" />
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # TTS variables removed
    
    # Initialize avatar setting before sidebar and main content
    # This ensures enable_avatar is accessible throughout the app
    enable_avatar = True  # Default value, will be overridden by sidebar checkbox
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Aafiya Settings")
        
        # AI Persona Selection with conditional white box
        if st.session_state.get('white_mode', False):
            st.markdown('<div class="personality-box">', unsafe_allow_html=True)
        
        st.subheader("👨‍⚕️ Choose Your Nutrition Expert")
        personas = get_nutrition_ai_personas()
        selected_persona = st.selectbox(
            "AI Personality",
            list(personas.keys()),
            help="Different experts have different approaches to nutrition advice"
        )
        persona_config = personas[selected_persona]
        
        # Display persona info
        with st.expander(f"About {selected_persona}"):
            st.write(persona_config['system_instruction'])
        
        if st.session_state.get('white_mode', False):
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Parameters
        st.subheader("🔧 AI Parameters")
        temperature = st.slider(
            "Creativity Level",
            0.0, 1.0, 
            persona_config['temperature'], 0.1,
            help="Higher = more creative responses, Lower = more focused advice"
        )
        
        max_tokens = st.number_input(
            "Response Length (tokens)",
            50, 2048, 1024, 50,
            help="Maximum length of AI responses"
        )
        
        # Theme Settings
        st.subheader("🎨 Theme Settings")
        white_mode = st.checkbox("☀️ Ivory Mode", value=st.session_state.white_mode, help="Switch to light theme with comfortable ivory white background")
        if white_mode != st.session_state.white_mode:
            st.session_state.white_mode = white_mode
            st.rerun()
        
        # Advanced Features
        st.subheader("🚀 Advanced Features")
        enable_image_analysis = st.checkbox("📸 Food Image Analysis", value=True)
        enable_nutrition_calculator = st.checkbox("🧮 Nutrition Calculator", value=True)
        enable_streaming = False  # Removed real-time responses widget
        enable_meal_logging = st.checkbox("📝 Meal Logging", value=True)
        
        # TTS feature removed
        
        # CraveSmart Feature
        st.subheader("🍩 CraveSmart")
        if st.button("🍩 CraveSmart - Transform Your Cravings!", use_container_width=True, type="secondary"):
            st.session_state.current_page = "cravesmart"
            st.rerun()
        
        # Interactive Avatar with HeyGen
        st.subheader("🤖 AI Avatar Assistant")
        
        # Always show the enable checkbox for avatar
        enable_avatar = st.checkbox("Enable Salma (AI Avatar)", value=enable_avatar, help="Enable Salma, your interactive AI nutritionist that can speak responses")
        
        if HEYGEN_AVAILABLE:
            if enable_avatar:
                # Avatar is enabled
                
                # Set default avatar settings
                avatar_style = "Professional Nutritionist"
                avatar_response_length = "Detailed"
                
            else:
                avatar_style = "Professional Nutritionist"
                avatar_response_length = "Detailed"
        else:
            enable_avatar = False
            avatar_style = "Professional Nutritionist"
            avatar_response_length = "Detailed"
            st.error("🔇 HeyGen Avatar not available")
        
        # Set TTS variables for compatibility (now using avatar instead)
        enable_tts = enable_avatar
        selected_voice = "Avatar"
        selected_voice_name = f"Salma ({avatar_style})"
        
        # Safety Settings (removed user configuration)
        safety_level = "Standard"  # Fixed safety level
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stage 1: Text-Based Nutrition Advice
        
        # Prompt Templates
        templates = get_nutrition_prompt_templates()
        selected_template = st.selectbox(
            "Choose a nutrition topic or ask custom question:",
            ["Custom Question"] + list(templates.keys()),
            help="Pre-made templates for common nutrition questions"
        )
        
        # Text input area
        if selected_template != "Custom Question":
            # Show different template display for Diet Plan
            if selected_template == "Diet Plan":
                st.info(f"📝 Template: {templates[selected_template].split(':')[0].replace('{user_input}', '[your goal]')}")
            else:
                st.info(f"📝 Template: {templates[selected_template].split(':')[0].replace('{user_input}', '[your input]')}")
                
            text_user_input = st.text_area(
                "Describe your meal or ask for diet advice:",
                height=100,
                placeholder="E.g., grilled chicken with quinoa and vegetables",
                key="text_input"
            )
            if text_user_input:
                text_prompt = templates[selected_template].format(user_input=text_user_input)
                
                # Show full template with user input for Diet Plan
                if selected_template == "Diet Plan":
                    st.info(f"📝 Template: Create a daily meal plan to {text_user_input} (e.g., build muscle, lose weight, maintain health). Include breakfast, lunch, dinner, and snacks.")
            else:
                text_prompt = ""
        else:
            text_prompt = st.text_area(
                "Describe your meal or ask for diet advice:",
                height=120,
                placeholder="E.g., What's a healthy breakfast for weight loss?",
                key="custom_text_input"
            )
        
        # Text advice button
        text_advice_button = st.button(
            "Get Advice",
            type="primary",
            use_container_width=True,
            key="text_button"
        )
        
        # Output window for text-based advice
        text_output_container = st.container()
        if text_advice_button and not text_prompt:
            with text_output_container:
                st.warning("⚠️ Please enter your nutrition question in the text area above.")
        elif text_advice_button and text_prompt:
            # Remove divider spacing
            pass
        
        # Stage 2: Image Analysis Section
        if enable_image_analysis:
            st.subheader("📸 Food Image Analysis")
            
            # Image analysis template selection
            image_templates = get_nutrition_prompt_templates()
            selected_image_template = st.selectbox(
                "Choose a nutrition topic or ask custom question:",
                ["Custom Question"] + list(image_templates.keys()),
                help="Pre-made templates for common nutrition questions",
                key="image_template_selector"
            )
            
            # Image prompt input with template support
            if selected_image_template != "Custom Question":
                # Show different template display for Diet Plan
                if selected_image_template == "Diet Plan":
                    st.info(f"📝 Template: {image_templates[selected_image_template].split(':')[0].replace('{user_input}', '[your goal]')}")
                else:
                    st.info(f"📝 Template: {image_templates[selected_image_template].split(':')[0].replace('{user_input}', '[your input]')}")
                    
                image_user_input = st.text_area(
                    "Describe what you want Aafiya to analyze about your meal photo:",
                    height=80,
                    placeholder="E.g., grilled chicken with quinoa and vegetables",
                    key="image_prompt_input"
                )
                if image_user_input:
                    image_prompt = image_templates[selected_image_template].format(user_input=image_user_input)
                    
                    # Show full template with user input for Diet Plan
                    if selected_image_template == "Diet Plan":
                        st.info(f"📝 Template: Create a daily meal plan to {image_user_input} (e.g., build muscle, lose weight, maintain health). Include breakfast, lunch, dinner, and snacks.")
                else:
                    image_prompt = ""
            else:
                image_prompt = st.text_area(
                    "Describe what you want Aafiya to analyze about your meal photo:",
                    height=80,
                    placeholder="E.g., Analyze this meal's nutrition and suggest improvements",
                    key="image_custom_prompt_input"
                )
            
            # Image uploader
            uploaded_image = st.file_uploader(
                "Upload a photo of your meal for analysis",
                type=['png', 'jpg', 'jpeg'],
                help="Take a photo of your meal for detailed nutritional analysis",
                key="image_uploader"
            )
            
            if uploaded_image:
                st.image(uploaded_image, caption="Your meal photo", use_container_width=True)
            
            # Image analysis button
            image_analysis_button = st.button(
                "📸 Analyze Food Image",
                type="secondary",
                use_container_width=True,
                key="image_button",
                disabled=not uploaded_image
            )
        else:
            image_analysis_button = False
            uploaded_image = None
            image_prompt = ""
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        # User Profile - Required Setup
        st.subheader("👤 Your Profile")
        
        # Check if profile is complete
        profile_complete = check_profile_complete()
        
        if not profile_complete:
            st.error("⚠️ Please complete your profile before using Aafiya AI")
        
        with st.expander("Complete Your Profile", expanded=not profile_complete):
            # Force profile expander styling in ivory mode
            if st.session_state.get('white_mode', False):
                st.markdown("""
                <style>
                div[data-testid="stExpander"] {
                    background: #FFFFF0 !important;
                    border: 1px solid #B8E8D0 !important;
                    border-radius: 8px !important;
                }
                div[data-testid="stExpander"] summary {
                    background: #FFFFF0 !important;
                }
                </style>
                """, unsafe_allow_html=True)
            # Age input
            age_value = st.session_state.user_profile['age'] if st.session_state.user_profile['age'] is not None else 0
            st.session_state.user_profile['age'] = st.number_input(
                "Age *", 
                min_value=0, 
                max_value=100, 
                value=age_value,
                help="Required for personalized nutrition recommendations. Enter your age (14+)",
                key="age_input"
            )
            
            # Gender input
            gender_options = ["Male", "Female"]
            current_gender = st.session_state.user_profile.get('gender')
            
            if current_gender in gender_options:
                gender_index = gender_options.index(current_gender) + 1  # +1 because of "Select Gender" at index 0
            else:
                gender_index = 0
            
            selected_gender = st.selectbox(
                "Gender *",
                ["Select Gender"] + gender_options,
                index=gender_index,
                help="Required for accurate calorie and nutrition calculations",
                key="gender_select"
            )
            
            if selected_gender != "Select Gender":
                st.session_state.user_profile['gender'] = selected_gender
            else:
                st.session_state.user_profile['gender'] = None
            
            # Weight input
            weight_value = st.session_state.user_profile['weight'] if st.session_state.user_profile['weight'] is not None else 0.0
            st.session_state.user_profile['weight'] = st.number_input(
                "Weight (kg) *", 
                min_value=0.0, 
                max_value=200.0, 
                value=weight_value,
                step=0.1,
                help="Current body weight in kilograms. Enter your weight",
                key="weight_input"
            )
            
            # Height input
            height_value = st.session_state.user_profile['height'] if st.session_state.user_profile['height'] is not None else 0
            st.session_state.user_profile['height'] = st.number_input(
                "Height (cm) *", 
                min_value=0, 
                max_value=220, 
                value=height_value,
                help="Height in centimeters. Enter your height",
                key="height_input"
            )
            
            # Activity Level
            activity_options = ["inactive", "light", "moderate", "active", "very active"]
            activity_labels = ["Inactive", "Light", "Moderate", "Active", "Very Active"]
            activity_index = 0 if st.session_state.user_profile['activity_level'] is None else activity_options.index(st.session_state.user_profile['activity_level'])
            selected_activity = st.selectbox(
                "Activity Level *",
                activity_labels,
                index=activity_index,
                help="How active are you throughout the week?",
                key="activity_select"
            )
            st.session_state.user_profile['activity_level'] = activity_options[activity_labels.index(selected_activity)]
            
            # Goal
            goal_options = ["lose weight", "maintain", "gain weight", "build muscle"]
            goal_labels = ["Lose Weight", "Maintain", "Gain Weight", "Build Muscle"]
            goal_index = 1 if st.session_state.user_profile['goal'] is None else goal_options.index(st.session_state.user_profile['goal'])
            selected_goal = st.selectbox(
                "Primary Goal *",
                goal_labels,
                index=goal_index,
                help="What is your main health/fitness goal?",
                key="goal_select"
            )
            st.session_state.user_profile['goal'] = goal_options[goal_labels.index(selected_goal)]
            
            # Optional: Goal Duration (in weeks)
            st.session_state.user_profile['goal_duration'] = st.number_input(
                "Goal Duration (weeks) - Optional",
                min_value=4,
                max_value=52,
                value=st.session_state.user_profile.get('goal_duration', 4),
                help="How many weeks do you want to work towards this goal? (minimum 4 weeks)",
                key="goal_duration_input"
            )
            
            # Optional: Allergies & Intolerances
            st.markdown("🥗 **Food Allergies & Intolerances**")
            
            # Default list of common allergies and sensitivities
            allergy_list = [
                "Lactose Intolerant",
                "Gluten Sensitivity", 
                "Fructose Intolerance",
                "Eggs",
                "Fish",
                "Shellfish",
                "Nuts",
                "Peanuts",
                "Wheat",
                "Soy",
                "Sesame",
                "Honey"
            ]
            
            # Initialize session state for allergies if not exists
            if 'user_allergies' not in st.session_state:
                st.session_state.user_allergies = []
            
            # Multiselect for known allergies
            selected_allergies = st.multiselect(
                "Select your known allergies or food sensitivities:",
                options=allergy_list,
                default=st.session_state.user_allergies,
                key="allergies_multiselect"
            )
            
            # Add custom allergy functionality
            col1, col2 = st.columns([4, 1])
            with col1:
                custom_allergy = st.text_input(
                    "Add a custom allergy:",
                    placeholder="Enter custom allergy/intolerance",
                    key="custom_allergy_input"
                )
            with col2:
                st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)  # Add spacing
                add_custom = st.button(
                    "Add", 
                    key="add_allergy_btn",
                    help="Add custom allergy to your list"
                )
                
                # ULTIMATE FORCE Add button styling - IVORY MODE ONLY
                if st.session_state.get('white_mode', False):
                    st.markdown("""
                    <style>
                    /* Target ALL possible button selectors */
                    button[key="add_allergy_btn"],
                    div[data-testid="column"] button,
                    .stButton > button,
                    div.stButton button,
                    [data-testid="stButton"] button,
                    div[data-testid="stColumn"] button {
                        background: #B8E8D0 !important;
                        background-color: #B8E8D0 !important;
                        color: #0f5a5e !important;
                        margin-top: 14px !important;
                        font-size: 0.8rem !important;
                        padding: 0.25rem 0.5rem !important;
                        border: none !important;
                        border-radius: 6px !important;
                        height: auto !important;
                        min-height: 32px !important;
                    }
                    </style>
                    <script>
                    // Multiple attempts to style the button
                    function styleAddButton() {
                        // Try multiple selectors
                        const selectors = [
                            'button[key="add_allergy_btn"]',
                            'button:contains("Add")',
                            'div[data-testid="stColumn"] button',
                            '.stButton button',
                            'button'
                        ];
                        
                        selectors.forEach(selector => {
                            try {
                                const buttons = document.querySelectorAll(selector);
                                buttons.forEach(button => {
                                    if (button.textContent.includes('Add') || button.getAttribute('key') === 'add_allergy_btn') {
                                        button.style.setProperty('background', '#B8E8D0', 'important');
                                        button.style.setProperty('background-color', '#B8E8D0', 'important');
                                        button.style.setProperty('color', '#0f5a5e', 'important');
                                        button.style.setProperty('margin-top', '14px', 'important');
                                        button.style.setProperty('font-size', '0.8rem', 'important');
                                        button.style.setProperty('padding', '0.25rem 0.5rem', 'important');
                                        button.style.setProperty('border', 'none', 'important');
                                        button.style.setProperty('border-radius', '6px', 'important');
                                    }
                                });
                            } catch(e) {}
                        });
                    }
                    
                    // Run immediately and then repeatedly
                    styleAddButton();
                    setTimeout(styleAddButton, 100);
                    setTimeout(styleAddButton, 500);
                    setTimeout(styleAddButton, 1000);
                    
                    // Also run on any DOM changes
                    const observer = new MutationObserver(styleAddButton);
                    observer.observe(document.body, { childList: true, subtree: true });
                    </script>
                    """, unsafe_allow_html=True)
            
            # Handle adding custom allergy
            if add_custom and custom_allergy:
                if custom_allergy not in selected_allergies and len(selected_allergies) < len(allergy_list):
                    selected_allergies.append(custom_allergy)
                    st.success(f"✅ Added '{custom_allergy}' to your list!")
                    st.session_state.custom_allergy_input = ""  # Clear input
                elif len(selected_allergies) >= len(allergy_list):
                    st.warning(f"⚠️ Maximum {len(allergy_list)} allergies allowed")
                elif custom_allergy in selected_allergies:
                    st.info("ℹ️ This allergy is already in your list")
            
            # Note: Users can remove allergies directly from the multiselect above
            
            # Update session state and user profile
            st.session_state.user_allergies = selected_allergies
            st.session_state.user_profile['allergies'] = ', '.join(selected_allergies) if selected_allergies else ''
            
            # Display current allergies
            if selected_allergies:
                st.success(f"**Current allergies:** {', '.join(selected_allergies)}")
                st.info(f"📋 Total: {len(selected_allergies)}/{len(allergy_list)} allergies selected")
            else:
                st.info("No allergies selected")
            
            # Optional: Health Issues
            st.session_state.user_profile['health_issues'] = st.text_input(
                "Health Issues - Optional",
                value=st.session_state.user_profile.get('health_issues', ''),
                placeholder="e.g., diabetes, high blood pressure, heart disease",
                help="Any health conditions that affect your diet",
                key="health_issues_input"
            )
            
            # Optional: Unpreferred Foods (Picky Eater)  
            # Handle conversion from list to string for text input
            current_unpreferred = st.session_state.user_profile.get('unpreferred_foods', [])
            if isinstance(current_unpreferred, list):
                current_unpreferred_str = ', '.join(current_unpreferred)
            else:
                current_unpreferred_str = current_unpreferred
                
            unpreferred_input = st.text_area(
                "Are you a picky eater? (Optional)",
                value=current_unpreferred_str,
                placeholder="e.g., broccoli, spicy food, fish, mushrooms, onions",
                help="List any foods you prefer to avoid, separated by commas",
                key="unpreferred_foods_input",
                height=80
            )
            
            # Convert back to list and store
            if unpreferred_input.strip():
                st.session_state.user_profile['unpreferred_foods'] = [item.strip() for item in unpreferred_input.split(',') if item.strip()]
            else:
                st.session_state.user_profile['unpreferred_foods'] = []
            
            # Update profile complete status
            st.session_state.user_profile['profile_complete'] = check_profile_complete()
            
            if st.session_state.user_profile['profile_complete']:
                st.success("✓ Profile Complete! You can now use Aafiya AI")
                
                # Done button to confirm profile completion
                profile_done_button = st.button(
                    "✅ Done - Confirm Profile",
                    type="secondary",
                    use_container_width=True,
                    key="profile_done_button"
                )
                
                if profile_done_button:
                    st.balloons()
                    st.success("🎉 Profile confirmed! Welcome to Aafiya AI!")
                    st.info("You can now start getting personalized nutrition advice below.")
                
                # Calculate and display BMI with detailed explanation
                bmi = calculate_bmi(st.session_state.user_profile['weight'], st.session_state.user_profile['height'])
                
                # BMI Category classification
                if bmi < 18.5:
                    bmi_category = "Underweight"
                    bmi_color = "blue"
                    bmi_advice = "Consider consulting a healthcare provider for healthy weight gain strategies."
                elif 18.5 <= bmi < 25:
                    bmi_category = "Normal weight"
                    bmi_color = "green"
                    bmi_advice = "Great! Maintain your healthy weight with balanced nutrition and regular activity."
                elif 25 <= bmi < 30:
                    bmi_category = "Overweight"
                    bmi_color = "orange"
                    bmi_advice = "Consider incorporating more physical activity and portion control in your routine."
                else:
                    bmi_category = "Obese"
                    bmi_color = "red"
                    bmi_advice = "Consult with a healthcare provider for a comprehensive weight management plan."
                
                # Display BMI with detailed information
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Your BMI", f"{bmi:.1f}", help="Body Mass Index calculated from your height and weight")
                with col2:
                    st.markdown(f"**Category:** :{bmi_color}[{bmi_category}]")
                
                with st.expander("📚 Understanding Your BMI"):
                    st.markdown(f"""
                    **Your BMI Analysis:**
                    - **BMI Value:** {bmi:.1f}
                    - **Category:** {bmi_category}
                    - **Recommendation:** {bmi_advice}
                    
                    **BMI Categories (WHO Standards):**
                    - Underweight: Below 18.5
                    - Normal weight: 18.5 - 24.9
                    - Overweight: 25.0 - 29.9
                    - Obese: 30.0 and above
                    
                    **Important Note:** BMI is a screening tool and doesn't account for muscle mass, bone density, or body composition. For personalized health advice, consult with healthcare professionals.
                    
                    **Aafiya AI Integration:** Your BMI and health goals are automatically considered in all nutrition recommendations to provide personalized advice tailored to your needs.
                    """)
        
        # Enhanced Knowledge Base with uploader
        st.markdown("""
        <div class="knowledge-base-box">
            <h3>📚 Aafiya's Knowledge Base</h3>
            <p><strong>Upload nutrition documents, Please provide your InBody test results (if available), basic body measurements, medical history, current medications or allergies</strong></p>
        """, unsafe_allow_html=True)
        
        # Include uploader inside the box
        nutrition_documents = nutrition_document_uploader()
        
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
        
        # 📝 Meal Logging Section
        st.divider()
        st.subheader("📝 Your Meal Log")
        
        # Initialize meal log if not exists
        if 'meal_log' not in st.session_state:
            st.session_state.meal_log = []
        
        if st.session_state.meal_log:
            st.success(f"🍽️ You have logged {len(st.session_state.meal_log)} meals")
            
            # Display recent meals in expander
            with st.expander("View Recent Meals", expanded=False):
                for i, meal in enumerate(reversed(st.session_state.meal_log[-10:])):
                    meal_number = len(st.session_state.meal_log) - i
                    st.markdown(f"**{meal_number}. {meal['food']}**")
                    st.caption(f"Logged: {meal['time']}")
                    
                    # Show nutrition data if available
                    if meal.get('nutrition'):
                        nutrition = meal['nutrition']
                        st.write("**Approximate Nutrition:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Calories", f"~{nutrition.get('calories', 0)}")
                        with col2:
                            st.metric("Protein", f"~{nutrition.get('protein', 0)}g")
                        with col3:
                            st.metric("Carbs", f"~{nutrition.get('carbs', 0)}g")
                        with col4:
                            st.metric("Fat", f"~{nutrition.get('fat', 0)}g")
                        
                        # Show source information
                        if nutrition.get('source'):
                            st.caption(f"Data source: {nutrition['source']}")
                    
                    # Show AI advice if available
                    if meal.get('ai_advice'):
                        st.write("**AI Advice:**")
                        st.info(meal['ai_advice'])
                    
                    # Remove meal button
                    if st.button(f"🗑️ Remove Meal {meal_number}", key=f"remove_meal_{i}"):
                        # Find and remove the specific meal
                        meal_index = len(st.session_state.meal_log) - 1 - i
                        st.session_state.meal_log.pop(meal_index)
                        st.success("Meal removed!")
                        st.rerun()
                    
                    st.divider()
                
                # Clear all meals button
                if st.button("🗑️ Clear All Meals", key="clear_all_meals"):
                    st.session_state.meal_log = []
                    st.success("All meals cleared!")
                    st.rerun()
        else:
            st.info("No meals logged yet. Start a food conversation to see your meals here!")
            st.caption("🍽️ Meals are automatically logged when you discuss food, nutrition, or upload food images")
        
        # Salma Avatar Integration in Right Panel
        if enable_avatar and HEYGEN_AVAILABLE:
            st.divider()
            st.subheader("🎥 Chat with Salma")
            
            # Embed Salma's avatar directly in the right panel
            st.markdown("""
            <div class="salma-avatar-container" style="width: 100%; height: 300px;">
                <iframe 
                    src="https://labs.heygen.com/guest/streaming-embed?share=eyJxdWFsaXR5IjoiaGlnaCIsImF2YXRhck5hbWUiOiJBbGVzc2FuZHJhX0NoYWlyX1NpdHRpbmdf%0D%0AcHVibGljIiwicHJldmlld0ltZyI6Imh0dHBzOi8vZmlsZXMyLmhleWdlbi5haS9hdmF0YXIvdjMv%0D%0AODllMDdiODI2ZjFjNGNiMWE1NTQ5MjAxY2RkOGY0ZDZfNTUzMDAvcHJldmlld190YXJnZXQud2Vi%0D%0AcCIsIm5lZWRSZW1vdmVCYWNrZ3JvdW5kIjpmYWxzZSwia25vd2xlZGdlQmFzZUlkIjoiZTQ0MzAw%0D%0AYWY5YWJjNGRlNmJlMjk4MzI5MzVlOTUzZjIiLCJ1c2VybmFtZSI6IjYwOGYyODY0MWE3ODRjZDk5%0D%0ANzZiZjMwNDQ4OGNhNTcxIn0%3D&inIFrame=1"
                    width="100%" 
                    height="100%" 
                    frameborder="0" 
                    allow="microphone; camera"
                    title="Salma - AI Nutritionist"
                    style="border-radius: 12px;">
                </iframe>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="avatar-status">
                <strong>👩‍⚕️ Salma - Your AI Nutritionist</strong><br>
                <span>🎤 <strong>Voice Chat:</strong> Speak directly to Salma</span><br>
                <span>🧠 <strong>Expertise:</strong> Nutrition, meal planning, health advice</span><br>
                <span>🎭 <strong>Interactive:</strong> Play fact or myth about diet culture with Salma</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Start chat instruction
            st.info("🎤 Click on Salma above to start speaking directly to her!")
            
            # Full-screen link
            st.markdown("""
            <div style="text-align: center; margin-top: 15px;">
                <a href="https://labs.heygen.com/guest/streaming-embed?share=eyJxdWFsaXR5IjoiaGlnaCIsImF2YXRhck5hbWUiOiJBbGVzc2FuZHJhX0NoYWlyX1NpdHRpbmdf%0D%0AcHVibGljIiwicHJldmlld0ltZyI6Imh0dHBzOi8vZmlsZXMyLmhleWdlbi5haS9hdmF0YXIvdjMv%0D%0AODllMDdiODI2ZjFjNGNiMWE1NTQ5MjAxY2RkOGY0ZDZfNTUzMDAvcHJldmlld190YXJnZXQud2Vi%0D%0AcCIsIm5lZWRSZW1vdmVCYWNrZ3JvdW5kIjpmYWxzZSwia25vd2xlZGdlQmFzZUlkIjoiZTQ0MzAw%0D%0AYWY5YWJjNGRlNmJlMjk4MzI5MzVlOTUzZjIiLCJ1c2VybmFtZSI6IjYwOGYyODY0MWE3ODRjZDk5%0D%0ANzZiZjMwNDQ4OGNhNTcxIn0%3D" 
                   target="_blank" 
                   class="fullscreen-link" 
                   style="display: inline-block; padding: 10px 20px; background: linear-gradient(to right, #91f2c4, #0f5a5e); color: white; text-decoration: none; border-radius: 8px; font-weight: bold;">
                    🖥️ Open Salma in Full-Screen
                </a>
                <div style="margin-top: 5px; font-size: 0.8em; color: #666;">
                    ✨ For the best experience with Salma's video chat
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        elif enable_avatar and not HEYGEN_AVAILABLE:
            st.divider()
            st.subheader("🎥 Salma Avatar")
            st.warning("🔇 HeyGen Avatar not available in this environment")
            st.info("💡 Avatar integration requires HeyGen streaming capabilities")
        
        elif not enable_avatar:
            st.divider()
            st.info("🤖 Enable Salma in the sidebar to see your AI nutritionist here!")
        
        # Quick nutrition info
        if st.session_state.nutrition_data:
            st.subheader("📊 Today's Nutrition")
            total_calories = sum(item.get('calories', 0) for item in st.session_state.nutrition_data)
            st.markdown(f"""
            <div class="calorie-display">
                🔥 {total_calories} calories today
            </div>
            """, unsafe_allow_html=True)
        
        # Meal log
        if enable_meal_logging:
            if st.session_state.meal_log:
                st.subheader("📝 Recent Meals")
                for i, meal in enumerate(reversed(st.session_state.meal_log[-3:])):
                    with st.expander(f"Meal {len(st.session_state.meal_log) - i}"):
                        st.write(f"**Food:** {meal['food']}")
                        st.write(f"**Time:** {meal['time']}")
                        if 'nutrition' in meal and meal['nutrition']:
                            nutrition = meal['nutrition']
                            source = nutrition.get('source', 'Unknown')
                            
                            # Display nutrition info with approximate values
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Calories (approx)", f"~{nutrition.get('calories', 0)}")
                            with col2:
                                st.metric("Protein (approx)", f"~{nutrition.get('protein', 0)}g")
                            with col3:
                                st.metric("Carbs (approx)", f"~{nutrition.get('carbs', 0)}g")
                            with col4:
                                st.metric("Fat (approx)", f"~{nutrition.get('fat', 0)}g")
                            
                            # Show additional nutrients if available
                            if 'fiber' in nutrition or 'sugar' in nutrition or 'sodium' in nutrition:
                                st.write("**Additional Approximate Nutrients:**")
                                add_col1, add_col2, add_col3 = st.columns(3)
                                with add_col1:
                                    if 'fiber' in nutrition:
                                        st.metric("Fiber (approx)", f"~{nutrition['fiber']}g")
                                with add_col2:
                                    if 'sugar' in nutrition:
                                        st.metric("Sugar (approx)", f"~{nutrition['sugar']}g")
                                with add_col3:
                                    if 'sodium' in nutrition:
                                        st.metric("Sodium (approx)", f"~{nutrition['sodium']}mg")
                            
                            # Show approximate disclaimer
                            st.caption("📊 Approximate nutrition values based on food analysis")
                        
                        # Show AI advice if available
                        if 'ai_advice' in meal:
                            st.write(f"**AI Advice:** {meal['ai_advice']}")
    
    # Process requests with separate outputs
    profile_complete = check_profile_complete()
    
    # Show profile completion notice but allow chat to continue
    if not profile_complete:
        if text_advice_button or image_analysis_button:
            st.warning("⚠️ Profile incomplete - Complete your profile in the sidebar for personalized nutrition advice")
            st.info("👉 Fill in all required fields (Age, Weight, Height, Activity Level, Goal) for better recommendations")
    
    # Process requests regardless of profile completion (for chat history saving)
    # Handle text-based advice
    if text_advice_button and text_prompt:
        process_nutrition_request(
            text_prompt, 
            None, 
            selected_persona, 
            persona_config, 
            temperature, 
            max_tokens, 
            enable_nutrition_calculator, 
            enable_streaming, 
            enable_meal_logging, 
            safety_level,
            "text",
            enable_tts,
            selected_voice,
            selected_voice_name,
            avatar_style,
            avatar_response_length,
            enable_avatar
        )
    
    # Handle image analysis
    if image_analysis_button and uploaded_image and image_prompt:
        st.subheader("📸 Aafiya's Image Analysis Results")
        process_nutrition_request(
            image_prompt, 
            uploaded_image, 
            selected_persona, 
            persona_config, 
            temperature, 
            max_tokens, 
            enable_nutrition_calculator, 
            enable_streaming, 
            enable_meal_logging, 
            safety_level,
            "image",
            enable_tts,
            selected_voice,
            selected_voice_name,
            avatar_style,
            avatar_response_length,
            enable_avatar
        )
    
    # Show helpful messages
    if image_analysis_button and not uploaded_image:
        st.warning("⚠️ Please upload an image to analyze.")
    if image_analysis_button and not image_prompt:
        st.warning("⚠️ Please describe what you want Aafiya to analyze about your meal photo.")
    
    # Display Chat History - ALWAYS SHOW SECTION
    st.subheader("💬 Your Conversations with Aafiya")
    
    if st.session_state.chat_history:
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            request_type_icon = "📸" if chat.get('has_image') else "💬"
            with st.expander(
                f"{request_type_icon} Chat {len(st.session_state.chat_history) - i} - {chat['persona']}", 
                expanded=(i == 0)
            ):
                st.markdown(f"**🧑 You:** {chat['prompt']}")
                st.markdown(f"**🤖 Aafiya ({chat['persona']}):** {chat['response']}")
                
                # Show nutrition data if available
                nutrition_data = chat.get('function_result') or chat.get('nutrition_data')
                if nutrition_data:
                    st.write("**📊 Nutrition Information:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Calories", f"~{nutrition_data.get('calories', 0)}")
                    with col2:
                        st.metric("Protein", f"~{nutrition_data.get('protein', 0)}g")
                    with col3:
                        st.metric("Carbs", f"~{nutrition_data.get('carbs', 0)}g")
                    with col4:
                        st.metric("Fat", f"~{nutrition_data.get('fat', 0)}g")
                    
                    # Show source if available
                    if nutrition_data.get('source'):
                        st.caption(f"Data source: {nutrition_data['source']}")
                    
                    # Show detailed data in expander
                    with st.expander("🔍 Detailed Nutrition Data"):
                        st.json(nutrition_data)
                
                st.caption(f"🌡️ Temperature: {chat['temperature']} | ⏰ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(chat['timestamp']))}")
    else:
        st.info("💬 No conversations yet. Start chatting with Aafiya to see your chat history here!")
        st.caption("🔍 Debug: Chat history is initialized but empty")

    # Credit line at the end of the page
    st.divider()
    st.markdown(
        "<div style='text-align: center; margin-top: 2rem; padding: 1rem; color: #666; font-size: 0.9em;'>"
        "Created by Nourah Alotaibi"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()