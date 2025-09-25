import os
import warnings
import requests
import time
from datetime import datetime, timedelta
import openai
from typing import List, Dict, Any, Optional, Tuple
import logging
from html import escape
from fpdf import FPDF
import io
import hashlib
import json
from openpyxl import Workbook 
from io import StringIO

# Suppress torch warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
import PyPDF2
import fitz  # pymupdf
from PIL import Image
import base64
import re


# Add these imports at the top of your file
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import black, blue, grey
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.platypus import Image as ReportLabImage
from reportlab.lib.utils import ImageReader
import tempfile

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import plotly.io as pio

# Configuration constants
USER_CONFIGS_DIR = "user_configs"
DEFAULT_CONFIG = {
    "qdrant": {
        "host": "......",
        "port": 6333,
        "auto_connect": True
    },
    "webhook": {
        "url": ".......",
        "timeout": 60,
        "auto_test": False
    },
    "openai": {
        "api_key": "",
        "model": "text-embedding-3-small",
        "chat_model": "gpt-3.5-turbo",
        "auto_load": False
    },
    "SENTENCE_TRANSFORMER": {
        "api_key": "",
        "model": "all-mpnet-base-v2",
        "auto_load": True
    },
    "preferences": {
        "theme": "default",
        "debug_mode": False,
        "auto_save": True
    }
}

# Configuration constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CHUNK_SIZE = 2000
DEFAULT_CHUNK_SIZE = 500
OPENAI_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small", 
    "text-embedding-3-large"
]
SENTENCE_TRANSFORMER_MODELS = [
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1"
]

# User Management Constants
USERS_FILE = "users.json"
SESSIONS_FILE = "sessions.json"

# Page configuration
st.set_page_config(
    page_title="Qdrant File Reader - Multi-User",
    page_icon="🔍",
    layout="wide"
)

# Add these imports at the top of your file
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import black, blue, grey
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
import tempfile
import os
from datetime import datetime

# Configuration constants
SAVED_QUESTIONS_DIR = "saved_questions"

# Configuration for visualizations
CHART_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

PLOTLY_THEME = 'plotly_white'
CHART_HEIGHT = 400
CHART_WIDTH = 800

# Modified init_session_state function
def init_session_state():
    """Initialize session state variables with sentence transformer defaults"""
    defaults = {
        'qdrant_client': None,
        'model': None,
        'model_type': 'sentence-transformers',  # Changed default to sentence-transformers
        'collections': [],
        'user_collections': [],
        'webhook_url': ".......",
        'webhook_timeout': 60,
        'webhook_auto_test': False,
        'chat_history': [],
        'openai_api_key': "......",
        'openai_model': "text-embedding-3-small",
        'openai_chat_model': "gpt-3.5-turbo",
        'openai_auto_load': True,
        'qdrant_host': "......",
        'qdrant_port': 6333,
        'qdrant_auto_connect': True,
        'last_query': "",
        'needs_refresh': False,
        'connection_status': False,
        'model_status': False,
        'debug_mode': False,
        'auto_save_config': True,
        # Sentence transformer defaults
        'default_sentence_model': "all-mpnet-base-v2",  # Added this line
        'sentence_model_auto_load': True,  # Added this line
        # Authentication states
        'authenticated': False,
        'current_user': None,
        'login_attempt': False,
        # Question management states
        'selected_question': "",
        'show_question_manager': False,
        'show_save_question_modal': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def auto_connect_services():
    """Auto-connect to Qdrant and load default sentence transformer model"""
    
    # Auto-connect to Qdrant
    if (not st.session_state.connection_status and 
        st.session_state.get('qdrant_auto_connect', True) and
        st.session_state.authenticated):
        
        with st.spinner("🔄 Auto-connecting to Qdrant..."):
            try:
                client = QdrantManager.connect(
                    st.session_state.get('qdrant_host', '......'), 
                    st.session_state.get('qdrant_port', 6333)
                )
                if client:
                    st.session_state.qdrant_client = client
                    st.session_state.connection_status = True
                    
                    # Load collections with user filtering
                    try:
                        collections = client.get_collections().collections
                        all_collections = [col.name for col in collections]
                        st.session_state.collections = all_collections
                        
                        # Filter collections for current user
                        if st.session_state.current_user:
                            username = st.session_state.current_user["username"]
                            st.session_state.user_collections = get_user_collections(username, all_collections)
                        
                        st.success("✅ Auto-connected to Qdrant!")
                    except Exception as e:
                        st.warning(f"Connected to Qdrant but failed to load collections: {str(e)}")
                else:
                    st.warning("⚠️ Could not auto-connect to Qdrant")
            except Exception as e:
                st.warning(f"⚠️ Qdrant auto-connection failed: {str(e)}")
    
    # Auto-load sentence transformer model
    if (not st.session_state.model_status and 
        st.session_state.get('sentence_model_auto_load', True) and
        st.session_state.get('model_type') == 'sentence-transformers' and
        st.session_state.authenticated):
        
        model_name = st.session_state.get('default_sentence_model', 'all-mpnet-base-v2')
        
        with st.spinner(f"🔄 Auto-loading sentence transformer model: {model_name}..."):
            try:
                model = EmbeddingManager.load_sentence_transformer(model_name)
                st.session_state.model = model
                st.session_state.model_type = "sentence-transformers"
                st.session_state.model_status = True
                vector_size = EmbeddingManager.get_vector_size(model, "sentence-transformers")
                st.success(f"✅ Auto-loaded model: {model_name} ({vector_size}D vectors)")
            except Exception as e:
                st.warning(f"⚠️ Could not auto-load sentence transformer model: {str(e)}")

def is_plotly_json(text):
    """Check if text looks like Plotly JSON"""
    if not isinstance(text, str) or len(text) < 50:
        return False
    
    # Check for common Plotly JSON indicators
    plotly_indicators = [
        '"data":', '"layout":', '"config":',
        '"type":"bar"', '"type":"line"', '"type":"scatter"',
        '"template":', '"xaxis":', '"yaxis":',
        '"marker":', '"line":', '"fill":',
        '"barmode":', '"polar":', '"geo":'
    ]
    
    return any(indicator in text for indicator in plotly_indicators)

def process_webhook_response_enhanced(webhook_response):
    """Enhanced webhook response processing with chart detection"""
    if webhook_response.get("success"):
        response_content = webhook_response.get("response", "No response")
        
        # Create response data structure
        response_data = {"response": response_content}
        
        # Check if response is Plotly chart JSON
        if is_plotly_json(response_content):
            response_data["chart_data"] = {
                "chart_json": response_content,
                "config": {
                    "title": "Generated Visualization", 
                    "chart_type": "auto-detected",
                    "description": "Chart generated from your data"
                }
            }
            # Replace the response with a cleaner message
            response_data["response"] = "I've generated a visualization based on your request. The chart is displayed above."
        
        # Include existing chart data if available
        elif webhook_response.get("chart_data"):
            response_data["chart_data"] = webhook_response["chart_data"]
        
        # Check if response contains table data
        elif TableResponseParser.detect_table_content(response_content):
            response_data["contains_table"] = True
        
        # Convert to JSON string for storage with better error handling
        try:
            response_json = json.dumps(response_data, default=str, ensure_ascii=False)
        except Exception as e:
            st.error(f"JSON serialization error: {str(e)}")
            response_json = str(response_content)
        
        assistant_chat = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sender": "Assistant",
            "message": response_json,
            "type": "assistant",
            "metadata": {
                "success": True,
                "has_chart": bool(response_data.get("chart_data")),
                "has_table": response_data.get("contains_table", False),
                "response_time": webhook_response.get("response_time"),
                "status_code": webhook_response.get("status_code")
            }
        }
    else:
        # Error response
        assistant_chat = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sender": "Assistant", 
            "message": f"Error: {webhook_response.get('error', 'Unknown error')}",
            "type": "assistant",
            "metadata": {
                "success": False,
                "error": webhook_response.get('error')
            }
        }
    
    return assistant_chat

def get_api_key() -> Optional[str]:
    """Get OpenAI API key from secrets, environment, or user input"""
    # Try Streamlit secrets first
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        pass
    
    # Try environment variable
    env_key = os.getenv('OPENAI_API_KEY')
    if env_key:
        return env_key
    
    # Use session state (user input)
    return st.session_state.openai_api_key if st.session_state.openai_api_key else None


def render_user_management_tab():
    """Render user management tab (super admin only)"""
    if not st.session_state.current_user or st.session_state.current_user.get("role") != "super_admin":
        st.error("❌ Access denied. Super admin privileges required.")
        return
    
    st.header("👥 User Management")
    
    # Create new user
    with st.expander("➕ Create New User", expanded=False):
        with st.form("create_user_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input("Username", placeholder="Enter username")
                new_password = st.text_input("Password", type="password", placeholder="Enter password")
            with col2:
                new_role = st.selectbox("Role", ["user", "super_admin"])
                st.write("")  # Spacing
            
            if st.form_submit_button("🎯 Create User"):
                if new_username and new_password:
                    if len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters long")
                    elif UserManager.create_user(new_username, new_password, new_role):
                        st.success(f"✅ User '{new_username}' created successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Username already exists or creation failed")
                else:
                    st.error("❌ Please fill in all fields")
    
    # Display existing users
    st.subheader("📋 Existing Users")
    users = UserManager.get_all_users()
    
    if users:
        for username, user_data in users.items():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                with col1:
                    status_icon = "🟢" if user_data.get("active", True) else "🔴"
                    role_icon = "👑" if user_data.get("role") == "super_admin" else "👤"
                    st.write(f"{status_icon} {role_icon} **{username}**")
                    if user_data.get("created_at"):
                        created = user_data["created_at"][:10]  # Just the date
                        st.caption(f"Created: {created}")
                
                with col2:
                    st.write(f"**{user_data.get('role', 'user')}**")
                
                with col3:
                    status = "Active" if user_data.get("active", True) else "Inactive"
                    st.write(status)
                
                with col4:
                    if username != "admin":  # Don't allow deactivating admin
                        current_status = user_data.get("active", True)
                        new_status = not current_status
                        action = "Deactivate" if current_status else "Activate"
                        if st.button(f"{action}", key=f"toggle_{username}"):
                            if UserManager.update_user_status(username, new_status):
                                st.success(f"✅ User '{username}' {action.lower()}d")
                                st.rerun()
                
                with col5:
                    if username != "admin":  # Don't allow deleting admin
                        if st.button("🗑️ Delete", key=f"delete_{username}"):
                            if f"confirm_delete_{username}" not in st.session_state:
                                st.session_state[f"confirm_delete_{username}"] = True
                                st.rerun()
                
                # Confirmation for delete
                if st.session_state.get(f"confirm_delete_{username}", False):
                    st.warning(f"⚠️ Are you sure you want to delete user '{username}'?")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ Yes, Delete", key=f"yes_delete_{username}"):
                            if UserManager.delete_user(username):
                                st.success(f"✅ User '{username}' deleted")
                                st.session_state[f"confirm_delete_{username}"] = False
                                st.rerun()
                    with col_b:
                        if st.button("❌ Cancel", key=f"cancel_delete_{username}"):
                            st.session_state[f"confirm_delete_{username}"] = False
                            st.rerun()
                
                st.divider()
    else:
        st.info("📝 No users found")


class UserConfigManager:
    """Handles user-specific configuration storage and retrieval"""
    
    @staticmethod
    def ensure_config_dir():
        """Ensure user configs directory exists"""
        if not os.path.exists(USER_CONFIGS_DIR):
            os.makedirs(USER_CONFIGS_DIR)
    
    @staticmethod
    def get_user_config_path(username: str) -> str:
        """Get the config file path for a specific user"""
        UserConfigManager.ensure_config_dir()
        return os.path.join(USER_CONFIGS_DIR, f"{username}_config.json")
    
    @staticmethod
    def load_user_config(username: str) -> Dict[str, Any]:
        """Load user configuration, return default if not exists"""
        try:
            config_path = UserConfigManager.get_user_config_path(username)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge with default config to ensure all keys exist
                merged_config = DEFAULT_CONFIG.copy()
                for section, values in user_config.items():
                    if section in merged_config and isinstance(values, dict):
                        merged_config[section].update(values)
                    else:
                        merged_config[section] = values
                
                return merged_config
            else:
                return DEFAULT_CONFIG.copy()
        except Exception as e:
            st.error(f"❌ Error loading user config: {str(e)}")
            return DEFAULT_CONFIG.copy()
    
    @staticmethod
    def save_user_config(username: str, config: Dict[str, Any]) -> bool:
        """Save user configuration"""
        try:
            config_path = UserConfigManager.get_user_config_path(username)
            
            # Add metadata
            config["_metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"❌ Error saving user config: {str(e)}")
            return False
    
    @staticmethod
    def update_user_config_section(username: str, section: str, updates: Dict[str, Any]) -> bool:
        """Update a specific section of user config"""
        try:
            config = UserConfigManager.load_user_config(username)
            
            if section not in config:
                config[section] = {}
            
            config[section].update(updates)
            return UserConfigManager.save_user_config(username, config)
        except Exception as e:
            st.error(f"❌ Error updating config section: {str(e)}")
            return False
    
    @staticmethod
    def get_user_config_section(username: str, section: str) -> Dict[str, Any]:
        """Get a specific section of user config"""
        config = UserConfigManager.load_user_config(username)
        return config.get(section, {})
    
    @staticmethod
    def reset_user_config(username: str) -> bool:
        """Reset user config to defaults"""
        return UserConfigManager.save_user_config(username, DEFAULT_CONFIG.copy())
    
    @staticmethod
    def export_user_config(username: str) -> str:
        """Export user config as JSON string"""
        config = UserConfigManager.load_user_config(username)
        return json.dumps(config, indent=2)
    
    @staticmethod
    def import_user_config(username: str, config_json: str) -> bool:
        """Import user config from JSON string"""
        try:
            config = json.loads(config_json)
            return UserConfigManager.save_user_config(username, config)
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Error importing config: {str(e)}")
            return False
    

def load_user_session_config():
    """Load user configuration into session state"""
    if not st.session_state.current_user:
        return
    
    username = st.session_state.current_user["username"]
    user_config = UserConfigManager.load_user_config(username)
    
    # Load Qdrant config
    qdrant_config = user_config.get("qdrant", {})
    st.session_state.qdrant_host = qdrant_config.get("host", "......")
    st.session_state.qdrant_port = qdrant_config.get("port", 6333)
    st.session_state.qdrant_auto_connect = qdrant_config.get("auto_connect", True)
    
    # Load Webhook config
    webhook_config = user_config.get("webhook", {})
    st.session_state.webhook_url = webhook_config.get("url", "......")
    st.session_state.webhook_timeout = webhook_config.get("timeout", 45)
    st.session_state.webhook_auto_test = webhook_config.get("auto_test", False)
    
    # Load OpenAI config
    openai_config = user_config.get("openai", {})
    if openai_config.get("api_key"):
        st.session_state.openai_api_key = openai_config["api_key"]
    st.session_state.openai_model = openai_config.get("model", "text-embedding-3-small")
    st.session_state.openai_chat_model = openai_config.get("chat_model", "gpt-3.5-turbo")
    st.session_state.openai_auto_load = openai_config.get("auto_load", True)
    
    # Load preferences
    preferences = user_config.get("preferences", {})
    st.session_state.debug_mode = preferences.get("debug_mode", False)
    st.session_state.auto_save_config = preferences.get("auto_save", True)


# Modified render_login_page to auto-load config
def render_login_page():
    """Render the login page"""
    st.title("🔐 Login to Qdrant File Reader")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Please log in to continue")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submit_button:
                if username and password:
                    user_data = UserManager.authenticate_user(username, password)
                    if user_data:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user_data
                        st.session_state.login_attempt = False
                        
                        # Load user configuration
                        load_user_session_config()
                        
                        st.success("✅ Login successful!")
                        st.info("🔄 Loading your personal configuration...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                        st.session_state.login_attempt = True
                else:
                    st.error("❌ Please enter both username and password")
                    st.session_state.login_attempt = True
        
        # Show default credentials for demo
        with st.expander("🔧 Demo Credentials", expanded=False):
            st.markdown("""
            **Default Accounts:**
            - **Super Admin:** username: `admin`, password: `admin123`
            - **Regular User:** username: `user1`, password: `user123`
            
            Super admin can see all users' documents and manage users.
            Regular users can only see their own documents.
            Each user has their own configuration settings.
            """)

def create_user_collection_name(username: str, collection_name: str) -> str:
    """Create a user-specific collection name"""
    if st.session_state.current_user and st.session_state.current_user.get("role") == "super_admin":
        # Super admin can create collections without prefix
        return collection_name
    
    prefix = get_user_collection_prefix(username)
    if not collection_name.startswith(prefix):
        return f"{prefix}{collection_name}"
    return collection_name


class UserManager:
    """Handles user authentication and authorization"""
    
    @staticmethod
    def load_users() -> Dict[str, Dict]:
        """Load users from JSON file"""
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r') as f:
                    return json.load(f)
            else:
                # Create default users if file doesn't exist
                default_users = {
                    "admin": {
                        "password_hash": UserManager.hash_password("admin123"),
                        "role": "super_admin",
                        "created_at": datetime.now().isoformat(),
                        "active": True
                    },
                    "user1": {
                        "password_hash": UserManager.hash_password("user123"),
                        "role": "user",
                        "created_at": datetime.now().isoformat(),
                        "active": True
                    }
                }
                UserManager.save_users(default_users)
                return default_users
        except Exception as e:
            st.error(f"❌ Error loading users: {str(e)}")
            return {}
    
    @staticmethod
    def save_users(users: Dict[str, Dict]) -> bool:
        """Save users to JSON file"""
        try:
            with open(USERS_FILE, 'w') as f:
                json.dump(users, f, indent=2)
            return True
        except Exception as e:
            st.error(f"❌ Error saving users: {str(e)}")
            return False
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return UserManager.hash_password(password) == password_hash
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user data if valid"""
        users = UserManager.load_users()
        if username in users:
            user = users[username]
            if user.get("active", True) and UserManager.verify_password(password, user["password_hash"]):
                return {
                    "username": username,
                    "role": user["role"],
                    "created_at": user.get("created_at"),
                    "last_login": datetime.now().isoformat()
                }
        return None
    
    @staticmethod
    def create_user(username: str, password: str, role: str = "user") -> bool:
        """Create a new user"""
        users = UserManager.load_users()
        if username in users:
            return False  # User already exists
        
        users[username] = {
            "password_hash": UserManager.hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        return UserManager.save_users(users)
    
    @staticmethod
    def delete_user(username: str) -> bool:
        """Delete a user"""
        users = UserManager.load_users()
        if username in users and username != "admin":  # Protect admin user
            del users[username]
            return UserManager.save_users(users)
        return False
    
    @staticmethod
    def get_all_users() -> Dict[str, Dict]:
        """Get all users (for admin interface)"""
        return UserManager.load_users()
    
    @staticmethod
    def update_user_status(username: str, active: bool) -> bool:
        """Update user active status"""
        users = UserManager.load_users()
        if username in users:
            users[username]["active"] = active
            return UserManager.save_users(users)
        return False

def parse_complex_table_response(message_content):
    """Parse the specific complex table format from your webhook"""
    import pandas as pd
    import json
    
    try:
        # Extract the nested response
        if isinstance(message_content, str):
            try:
                # Parse the outer JSON
                outer_data = json.loads(message_content)
                
                # Get the response field which contains another JSON string
                response_field = outer_data.get('response', '')
                
                if response_field:
                    # Parse the inner JSON
                    inner_data = json.loads(response_field)
                    table_text = inner_data.get('output', '')
                else:
                    # Fallback to direct output field
                    table_text = outer_data.get('output', str(outer_data))
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, use the raw content
                table_text = message_content
        else:
            table_text = str(message_content)
        
        if not table_text or '|' not in table_text:
            return pd.DataFrame(), ""
        
        # Replace escaped newlines with actual newlines
        table_text = table_text.replace('\\n', '\n')
        
        # Split into lines
        lines = table_text.split('\n')
        
        # Find table lines (lines with multiple |)
        table_lines = []
        summary_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a table line
            if line.count('|') >= 4:  # At least 4 pipes for a table row
                # Skip separator lines (just dashes and pipes)
                if not (set(line.replace(' ', '')) <= set('|-')):
                    table_lines.append(line)
            else:
                # This might be summary text
                if ('table' in line.lower() or 'report' in line.lower() or 
                    'records' in line.lower() or 'details' in line.lower()):
                    summary_text += line + " "
        
        if len(table_lines) < 2:  # Need at least header + 1 data row
            return pd.DataFrame(), summary_text.strip()
        
        # Process the table lines
        processed_rows = []
        
        for line in table_lines:
            # Remove leading/trailing pipes and split
            if line.startswith('|'):
                line = line[1:]
            if line.endswith('|'):
                line = line[:-1]
            
            # Split by | and clean up
            cells = [cell.strip() for cell in line.split('|')]
            processed_rows.append(cells)
        
        if not processed_rows:
            return pd.DataFrame(), summary_text.strip()
        
        # First row should be headers
        headers = processed_rows[0]
        data_rows = processed_rows[1:]
        
        # Clean up headers (remove any empty ones)
        clean_headers = [h for h in headers if h]
        
        # Process data rows to match header count
        clean_data_rows = []
        for row in data_rows:
            # Take only as many columns as we have headers
            clean_row = row[:len(clean_headers)]
            # Pad with empty strings if needed
            while len(clean_row) < len(clean_headers):
                clean_row.append('')
            clean_data_rows.append(clean_row)
        
        if clean_data_rows:
            df = pd.DataFrame(clean_data_rows, columns=clean_headers)
            return df, summary_text.strip()
        
    except Exception as e:
        print(f"Complex table parsing error: {e}")
    
    return pd.DataFrame(), ""

# Enhanced table parser that handles your specific response format

def extract_and_parse_table(response_text):
    """Extract and parse table from complex response format"""
 
    try:
        if not response_text or '|' not in response_text:
            return pd.DataFrame()
        
        # Handle escaped newlines
        if '\\n' in response_text:
            lines = response_text.split('\\n')
        else:
            lines = response_text.split('\n')
        
        table_lines = []
        for line in lines:
            line = line.strip()
            if ('|' in line and 
                line.count('|') > 3 and 
                not line.replace('|', '').replace('-', '').replace(' ', '') == ''):
                table_lines.append(line)
        
        if len(table_lines) < 2:
            return pd.DataFrame()
        
        # Find header line
        header_line = None
        data_start = 0
        
        for i, line in enumerate(table_lines):
            if any(h in line for h in ['No', 'DocumentNr', 'Timestamp', 'CustomerName']):
                header_line = line
                data_start = i + 1
                break
        
        if not header_line:
            return pd.DataFrame()
        
        # Parse headers
        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        
        # Parse data
        data_rows = []
        for line in table_lines[data_start:]:
            row = [r.strip() for r in line.split('|') if r.strip()]
            if len(row) >= len(headers):
                data_rows.append(row[:len(headers)])
        
        if data_rows:
            return pd.DataFrame(data_rows, columns=headers)
        
    except Exception as e:
        print(f"Parse error: {e}")
    
    return pd.DataFrame()

def enhanced_chat_display():
    """Enhanced chat display with robust table parsing"""
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["type"] == "user":
                with st.chat_message("user"):
                    formatted = chat["message"].encode('utf-8').decode('unicode_escape')
                    st.markdown(f"""**[{chat['timestamp']}]**
                    {escape(formatted)}""")            
            
            else:
                with st.chat_message("assistant"):
                    st.write(f"""**[{chat['timestamp']}]**""")
                    
                    try:
                        message_content = chat["message"]
                        
                        # Handle JSON responses that might contain chart data
                        if isinstance(message_content, str):
                            try:
                                data = json.loads(message_content)
                                response_text = (
                                    data.get('output') or 
                                    data.get('response') or 
                                    data.get('text') or 
                                    data.get('answer') or
                                    str(data)
                                )
                                
                                # Check for chart data in the response
                                chart_data = data.get('chart_data')
                                if chart_data and chart_data.get('chart_json'):
                                    try:
                                        # Display the chart
                                        st.subheader("📊 Generated Visualization")
                                        chart_fig = pio.from_json(chart_data['chart_json'])
                                        st.plotly_chart(chart_fig, use_container_width=True)
                                        
                                        # Display chart info
                                        config = chart_data.get('config', {})
                                        if config:
                                            st.info(f"Chart Type: {config.get('chart_type', 'Unknown').title()} | Title: {config.get('title', 'Visualization')}")
                                        
                                        # Also show the text response if available
                                        if response_text and response_text != str(data) and not is_plotly_json(response_text):
                                            st.write("**Response:**")
                                            st.markdown(response_text)
                                        
                                    except Exception as e:
                                        st.error(f"Error displaying chart: {str(e)}")
                                        # Fallback to text display
                                        st.text("Chart data could not be rendered")
                                
                                # Check if the response itself looks like chart JSON
                                elif is_plotly_json(response_text):
                                    try:
                                        st.subheader("📊 Generated Visualization") 
                                        chart_fig = pio.from_json(response_text)
                                        st.plotly_chart(chart_fig, use_container_width=True)
                                        st.success("Chart generated successfully!")
                                    except Exception as e:
                                        st.error(f"Error rendering chart: {str(e)}")
                                        st.write("The response contains chart data but could not be displayed.")
                                
                                else:
                                    # Try to parse as table first
                                    df = extract_and_parse_table(response_text)
                                    
                                    if not df.empty:
                                        st.subheader("📊 Data Table")
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Download button
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            "📥 Download CSV",
                                            csv,
                                            f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            "text/csv",
                                            key=f"download_{chat['timestamp']}"
                                        )
                                    else:
                                        # Display as regular text
                                        st.markdown(response_text)
                                        
                            except json.JSONDecodeError:
                                # Check if raw content looks like Plotly JSON
                                if is_plotly_json(message_content):
                                    try:
                                        st.subheader("📊 Generated Visualization")
                                        chart_fig = pio.from_json(message_content)
                                        st.plotly_chart(chart_fig, use_container_width=True)
                                        st.success("Chart generated successfully!")
                                    except Exception as e:
                                        st.error(f"Error rendering chart: {str(e)}")
                                        st.write("Chart data detected but could not be rendered.")
                                else:
                                    # Regular text response
                                    st.markdown(message_content)
                        else:
                            st.markdown(str(message_content))
                        
                    except Exception as e:
                        st.error(f"Error displaying response: {str(e)}")
                        st.text("Could not display response properly")
                    
                    # Show metadata if available
                    if "metadata" in chat and not chat["metadata"]["success"]:
                        st.error(f"❌ Error: {chat['metadata']['error']}")
                    elif "metadata" in chat and chat["metadata"].get("response_time"):
                        st.caption(f"⏱️ Response time: {chat['metadata']['response_time']:.2f}s | Status: {chat['metadata']['status_code']}")

def simple_table_fix_for_sales_report():
    """Simple fix specifically for sales report tables"""
    # Replace your existing chat display loop with this:
    
    for chat in st.session_state.chat_history:
        if chat["type"] == "user":
            with st.chat_message("user"):
                formatted = chat["message"].encode('utf-8').decode('unicode_escape')
                st.markdown(f"""**[{chat['timestamp']}]**
                {escape(formatted)}""")            
               
        else:
            with st.chat_message("assistant"):
                st.write(f"""**[{chat['timestamp']}]**""")
                
                # Extract response content
                message_content = chat["message"]
                if isinstance(message_content, str):
                    try:
                        data = json.loads(message_content)
                        response_text = data.get('output', str(data))
                    except:
                        response_text = message_content
                else:
                    response_text = str(message_content)
                
                # Look for table pattern specifically
                if ('| No |' in response_text and 'DocumentNr' in response_text) or response_text.count('|') > 15:
                    # This looks like a table, try to parse it
                    df = extract_and_parse_table(response_text)
                    
                    if not df.empty:
                        st.subheader("📊 Sales Report Table")
                        st.dataframe(df, use_container_width=True)
                        
                        # Quick download
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"report_{datetime.now().strftime('%H%M%S')}.csv",
                            "text/csv",
                            key=f"dl_{chat['timestamp']}"
                        )
                    else:
                        # Parsing failed, show as text
                        st.markdown(response_text.replace('\\n', '\n'))
                else:
                    # Not a table, show as text
                    st.markdown(response_text.replace('\\n', '\n'))

class TableResponseParser:
    """Handles parsing and displaying table data from webhook responses"""
    
    @staticmethod
    def detect_table_content(text: str) -> bool:
        """Detect if the response contains table-like content"""
        table_indicators = [
            '|' in text and text.count('|') > 5,  # Multiple pipe separators
            'No | Customer Name |' in text,  # Specific table headers
            '---|---' in text,  # Markdown table separators
            re.search(r'\|\s*\d+\s*\|', text),  # Pattern like "| 1 |"
            re.search(r'\|\s*[A-Za-z\s]+\s*\|.*\|\s*[0-9,\.]+\s*\|', text),  # Data pattern
        ]
        return any(table_indicators)
    
    @staticmethod
    def clean_table_text(text: str) -> str:
        """Clean and prepare table text for parsing"""
        # Remove any introductory text before the actual table
        lines = text.split('\n')
        table_start = -1
        
        for i, line in enumerate(lines):
            # Look for the first line that looks like a table header or data
            if '|' in line and (
                'No |' in line or 
                'Customer' in line or 
                re.search(r'\|\s*\d+\s*\|', line)
            ):
                table_start = i
                break
        
        if table_start >= 0:
            # Take only the table part
            table_lines = lines[table_start:]
            
            # Remove any trailing explanatory text
            clean_lines = []
            for line in table_lines:
                line = line.strip()
                if ('this table' in line.lower() or 
                    'if you need' in line.lower() or 
                    'feel free' in line.lower()):
                    break
                if line:  # Keep non-empty lines
                    clean_lines.append(line)
            
            return '\n'.join(clean_lines)
        
        return text
    
    @staticmethod
    def parse_pipe_separated_table(text: str):
        """Parse pipe-separated table data - fixed version without regex issues"""
        try:
            lines = text.strip().split('\n')
            data_lines = []
            headers = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip separator lines (like ---|---|---) using simple string checks
                if ('---' in line and '|' in line) or line.replace('|', '').replace('-', '').replace(' ', '') == '':
                    continue
                    
                # Process lines with pipe separators
                if '|' in line:
                    # Clean and split the line
                    parts = [part.strip() for part in line.split('|')]
                    # Remove empty parts at start/end
                    while parts and not parts[0]:
                        parts.pop(0)
                    while parts and not parts[-1]:
                        parts.pop()
                    
                    if parts and len(parts) > 1:  # Need at least 2 columns
                        if headers is None:
                            # First valid line is headers
                            headers = parts
                        else:
                            # Data row - skip if it looks like a continuation of text
                            if not any(keyword in line.lower() for keyword in ['this table', 'if you need', 'feel free']):
                                data_lines.append(parts)
            
            if headers and data_lines:
                # Ensure all rows have the same number of columns as headers
                max_cols = len(headers)
                for i, row in enumerate(data_lines):
                    if len(row) < max_cols:
                        # Pad with empty strings
                        data_lines[i] = row + [''] * (max_cols - len(row))
                    elif len(row) > max_cols:
                        # Truncate
                        data_lines[i] = row[:max_cols]
                
                import pandas as pd
                df = pd.DataFrame(data_lines, columns=headers)
                return df
            
        except Exception as e:
            print(f"Error parsing table: {str(e)}")
        
        return None
    
    @staticmethod
    def parse_custom_table_format(text: str) -> pd.DataFrame:
        """Parse the specific format from your webhook response"""
        try:
            lines = text.strip().split('\n')
            data_rows = []
            
            # Look for the header pattern
            header_line = None
            for line in lines:
                if 'No | DocumentNr |' in line or 'Timestamp | CustomerName |' in line:
                    header_line = line
                    break
            
            if not header_line:
                return pd.DataFrame()
            
            # Extract headers
            headers = [h.strip() for h in header_line.split('|') if h.strip()]
            
            # Find data lines (lines that start with a number and contain |)
            for line in lines:
                line = line.strip()
                # Skip header, separator, and empty lines
                if not line or '---' in line or 'No | DocumentNr |' in line or 'Timestamp | CustomerName |' in line:
                    continue
                
                # Look for data lines (should start with a number)
                if re.match(r'^\d+\s*\|', line):
                    # Split by | and clean up
                    parts = [part.strip() for part in line.split('|')]
                    # Remove empty parts
                    parts = [part for part in parts if part]
                    
                    if len(parts) >= len(headers):
                        # Take only the number of columns we have headers for
                        data_rows.append(parts[:len(headers)])
                    else:
                        # Pad with empty strings if needed
                        data_rows.append(parts + [''] * (len(headers) - len(parts)))
            
            if data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                return df
                
        except Exception as e:
            st.error(f"Error parsing custom table format: {str(e)}")
        
        return pd.DataFrame()
    
    @staticmethod
    def display_table_response(response_text: str, title: str = "Data Table"):
        """Display table data with enhanced formatting"""
        try:
            # Clean the table text first
            clean_text = TableResponseParser.clean_table_text(response_text)
            
            # Try general pipe-separated format
            df = TableResponseParser.parse_pipe_separated_table(clean_text)
            
            # If that didn't work, try custom format
            if df.empty:
                df = TableResponseParser.parse_custom_table_format(clean_text)
            
            if not df.empty:
                st.subheader(f"📊 {title}")
                
                # Display basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    if any(df.dtypes == 'object'):
                        numeric_cols = len([col for col in df.columns if pd.to_numeric(df[col], errors='coerce').notna().any()])
                        st.metric("Numeric Columns", numeric_cols)
                
                # Display the table
                st.dataframe(df, use_container_width=True)
                
                # Add export options
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"table_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                with col2:
                    # Simple Excel export without openpyxl dependency
                    excel_buffer = io.StringIO()
                    df.to_csv(excel_buffer, index=False)
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        "📊 Download as CSV", 
                        excel_data,
                        f"table_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                with col3:
                    # Show column info
                    with st.expander("📋 Column Info"):
                        col_info = []
                        for col in df.columns:
                            col_info.append({
                                "Column": col,
                                "Type": "Text" if df[col].dtype == 'object' else "Numeric",
                                "Non-null Count": df[col].count(),
                                "Sample Value": str(df[col].iloc[0]) if len(df) > 0 else ""
                            })
                        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
                
                return True
            else:
                # If no table detected, show cleaned text instead of original
                # Only show the table part, not the full response
                if '|' in clean_text:
                    st.text(clean_text)
                else:
                    st.write(response_text)
                return False
                
        except Exception as e:
            st.error(f"Error displaying table: {str(e)}")
            st.write("Original response:")
            st.text(response_text)
            return False


# Modified chat rendering function

    
    @staticmethod
    def parse_custom_table_format(text: str) -> pd.DataFrame:
        """Parse the specific format from your webhook response"""
        try:
            lines = text.strip().split('\n')
            data_rows = []
            
            # Look for the header pattern
            header_line = None
            for line in lines:
                if 'No | DocumentNr |' in line or 'Timestamp | CustomerName |' in line:
                    header_line = line
                    break
            
            if not header_line:
                return pd.DataFrame()
            
            # Extract headers
            headers = [h.strip() for h in header_line.split('|') if h.strip()]
            
            # Find data lines (lines that start with a number and contain |)
            for line in lines:
                line = line.strip()
                # Skip header, separator, and empty lines
                if not line or '---' in line or 'No | DocumentNr |' in line or 'Timestamp | CustomerName |' in line:
                    continue
                
                # Look for data lines (should start with a number)
                if re.match(r'^\d+\s*\|', line):
                    # Split by | and clean up
                    parts = [part.strip() for part in line.split('|')]
                    # Remove empty parts
                    parts = [part for part in parts if part]
                    
                    if len(parts) >= len(headers):
                        # Take only the number of columns we have headers for
                        data_rows.append(parts[:len(headers)])
                    else:
                        # Pad with empty strings if needed
                        data_rows.append(parts + [''] * (len(headers) - len(parts)))
            
            if data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                return df
                
        except Exception as e:
            st.error(f"Error parsing custom table format: {str(e)}")
        
        return pd.DataFrame()
    
    @staticmethod
    def display_table_response(response_text: str, title: str = "Data Table"):
        """Display table data with enhanced formatting"""
        try:
            # Try custom format first (for your specific webhook format)
            df = TableResponseParser.parse_custom_table_format(response_text)
            
            # If custom format didn't work, try general pipe-separated format
            if df.empty:
                df = TableResponseParser.parse_pipe_separated_table(response_text)
            
            if not df.empty:
                st.subheader(f"📊 {title}")
                
                # Display basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    if any(df.dtypes == 'object'):
                        numeric_cols = len([col for col in df.columns if pd.to_numeric(df[col], errors='coerce').notna().any()])
                        st.metric("Numeric Columns", numeric_cols)
                
                # Display the table
                st.dataframe(df, use_container_width=True)
                
                # Add export options
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"table_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                with col2:
                    excel_buffer = StringIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        "📊 Download Excel", 
                        excel_data,
                        f"table_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col3:
                    # Show column info
                    with st.expander("📋 Column Info"):
                        col_info = []
                        for col in df.columns:
                            col_info.append({
                                "Column": col,
                                "Type": "Text" if df[col].dtype == 'object' else "Numeric",
                                "Non-null Count": df[col].count(),
                                "Sample Value": str(df[col].iloc[0]) if len(df) > 0 else ""
                            })
                        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
                
                return True
            else:
                # If no table detected, show original text
                st.write(response_text)
                return False
                
        except Exception as e:
            st.error(f"Error displaying table: {str(e)}")
            st.write("Original response:")
            st.text(response_text)
            return False
        
# Enhanced chart detection function - ADD THIS TO YOUR FILE
def is_advanced_plotly_json(text):
    """Enhanced detection for Plotly JSON with more specific checks"""
    if not isinstance(text, str) or len(text) < 100:
        return False
    
    # More specific Plotly indicators
    plotly_indicators = [
        '"data":',
        '"layout":',
        '"config":',
        '"chart_type":',
        '"x_column":',
        '"y_column":',
        '"data_summary":'
    ]
    
    # Check for chart structure
    has_plotly_structure = any(indicator in text for indicator in plotly_indicators)
    
    # Additional check: look for common chart elements
    chart_elements = [
        '"type":"bar"',
        '"type":"line"', 
        '"type":"scatter"',
        '"xaxis":',
        '"yaxis":',
        '"marker":',
        '"ProductName"',  # Specific to your data
        '"Amount"'        # Specific to your data
    ]
    
    has_chart_elements = any(element in text for element in chart_elements)
    
    return has_plotly_structure or has_chart_elements

def render_enhanced_chat_with_table_fix():
    """Enhanced chat rendering with specific table fix"""
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["type"] == "user":
                with st.chat_message("user"):
                    formatted = chat["message"].encode('utf-8').decode('unicode_escape')
                    st.markdown(f"""**[{chat['timestamp']}]**
                    {escape(formatted)}""")            
            
            else:
                with st.chat_message("assistant"):
                    st.write(f"""**[{chat['timestamp']}]**""")
                    
                    try:
                        message_content = chat["message"]
                        chart_rendered = False
                        
                        # Handle JSON responses that might contain chart data
                        if isinstance(message_content, str):
                            try:
                                data = json.loads(message_content)
                                response_text = (
                                    data.get('output') or 
                                    data.get('response') or 
                                    data.get('text') or 
                                    data.get('answer') or
                                    str(data)
                                )
                                
                                # Check for chart data in the response structure
                                chart_data = data.get('chart_data')
                                if chart_data and chart_data.get('chart_json'):
                                    try:
                                        st.subheader("📊 Generated Visualization")
                                        chart_fig = pio.from_json(chart_data['chart_json'])
                                        st.plotly_chart(chart_fig, use_container_width=True)
                                        chart_rendered = True
                                        
                                        # Display chart info
                                        config = chart_data.get('config', {})
                                        if config:
                                            st.info(f"**Chart:** {config.get('title', 'Data Visualization')} | Type: {config.get('chart_type', 'Unknown').title()}")
                                        
                                    except Exception as e:
                                        st.error(f"Error displaying structured chart: {str(e)}")
                                
                                # NEW: Check if the raw response is chart JSON (this is the key fix)
                                if not chart_rendered and is_advanced_plotly_json(response_text):
                                    try:
                                        st.subheader("📊 Product Sales Chart")
                                        
                                        # Clean the JSON string if needed
                                        clean_json = response_text.strip()
                                        if clean_json.startswith('```json'):
                                            clean_json = clean_json.replace('```json', '').replace('```', '').strip()
                                        
                                        chart_fig = pio.from_json(clean_json)
                                        st.plotly_chart(chart_fig, use_container_width=True)
                                        chart_rendered = True
                                        
                                        # Extract chart info from the JSON
                                        try:
                                            chart_config = json.loads(clean_json)
                                            if 'config' in chart_config:
                                                config_info = chart_config['config']
                                                st.success(f"✅ Chart created: {config_info.get('title', 'Product Sales')} ({config_info.get('chart_type', 'bar')} chart)")
                                        except:
                                            st.success("✅ Chart generated successfully!")
                                        
                                    except Exception as e:
                                        st.error(f"Error rendering chart from response: {str(e)}")
                                        st.write("**Debug:** The response contains chart data but couldn't be rendered:")
                                        st.code(response_text[:200] + "..." if len(response_text) > 200 else response_text)
                                
                                # Display text response only if no chart was rendered
                                if not chart_rendered:
                                    # Try to parse as table first
                                    df = extract_and_parse_table(response_text)
                                    
                                    if not df.empty:
                                        st.subheader("📊 Data Table")
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Download button
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            "📥 Download CSV",
                                            csv,
                                            f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            "text/csv",
                                            key=f"download_{chat['timestamp']}"
                                        )
                                    else:
                                        # Display as regular text
                                        st.markdown(response_text[:500] + "..." if len(response_text) > 500 else response_text)
                                        
                            except json.JSONDecodeError:
                                # Check if raw content looks like Plotly JSON (fallback)
                                if is_advanced_plotly_json(message_content):
                                    try:
                                        st.subheader("📊 Generated Chart")
                                        chart_fig = pio.from_json(message_content)
                                        st.plotly_chart(chart_fig, use_container_width=True)
                                        st.success("✅ Chart rendered successfully!")
                                        chart_rendered = True
                                    except Exception as e:
                                        st.error(f"Error rendering fallback chart: {str(e)}")
                                
                                if not chart_rendered:
                                    # Regular text response
                                    display_text = message_content[:500] + "..." if len(message_content) > 500 else message_content
                                    st.markdown(display_text)
                        else:
                            st.markdown(str(message_content))
                        
                    except Exception as e:
                        st.error(f"Error displaying response: {str(e)}")
                        st.text("Could not display response properly")
                    
                    # Show metadata if available
                    if "metadata" in chat and not chat["metadata"]["success"]:
                        st.error(f"❌ Error: {chat['metadata']['error']}")
                    elif "metadata" in chat and chat["metadata"].get("response_time"):
                        st.caption(f"⏱️ Response time: {chat['metadata']['response_time']:.2f}s | Status: {chat['metadata']['status_code']}")


def simple_table_display_fix():
    """Simple fix to replace your current chat display"""
    # Replace the for loop in your chat display with this:
    
    for chat in st.session_state.chat_history:
        if chat["type"] == "user":
            with st.chat_message("user"):
                formatted = chat["message"].encode('utf-8').decode('unicode_escape')
                st.markdown(f"""**[{chat['timestamp']}]**
                {escape(formatted)}""")            
               
        else:
            with st.chat_message("assistant"):
                st.write(f"""**[{chat['timestamp']}]**""")
                
                # Parse the table using the new function
                df, summary = parse_complex_table_response(chat["message"])
                
                if not df.empty:
                    st.subheader("📊 Sales Report Table")
                    if summary:
                        st.info(summary)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"report_{datetime.now().strftime('%H%M%S')}.csv",
                        "text/csv",
                        key=f"dl_{chat['timestamp']}"
                    )
                else:
                    # Display as text if table parsing failed
                    try:
                        data = json.loads(chat["message"])
                        if 'response' in data:
                            inner = json.loads(data['response'])
                            text = inner.get('output', str(data)).replace('\\n', '\n')
                        else:
                            text = data.get('output', str(data)).replace('\\n', '\n')
                        st.markdown(text)
                    except:
                        st.text(str(chat["message"]))

# Modified chat rendering function
def render_enhanced_chat_response(chat):
    """Enhanced chat response rendering with table support"""
    with st.chat_message("assistant"):
        st.write(f"""**[{chat['timestamp']}]**""")
        
        try:
            message_content = chat["message"]
            
            if isinstance(message_content, str):
                try:
                    data = json.loads(message_content)
                    response_text = (
                        data.get('response') or 
                        data.get('text') or 
                        data.get('answer') or
                        data.get('output') or  # Added 'output' field
                        data.get('result') or
                        str(data)
                    )
                    
                    # Check if response contains table data
                    if TableResponseParser.detect_table_content(response_text):
                        # Display as table
                        table_displayed = TableResponseParser.display_table_response(
                            response_text, 
                            "Query Results"
                        )
                        
                        if not table_displayed:
                            # Fallback to text display
                            st.markdown(response_text)
                    else:
                        # Regular text response
                        st.markdown(response_text)
                    
                    # Display chart if available (existing functionality)
                    chart_data = data.get('chart_data')
                    if chart_data and chart_data.get('chart_json'):
                        try:
                            st.subheader("📊 Generated Visualization")
                            chart_fig = pio.from_json(chart_data['chart_json'])
                            st.plotly_chart(chart_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying chart: {str(e)}")
                    
                except json.JSONDecodeError:
                    # Not JSON, check if it's table data
                    if TableResponseParser.detect_table_content(message_content):
                        TableResponseParser.display_table_response(message_content, "Query Results")
                    else:
                        st.markdown(message_content)
            else:
                # Handle non-string responses
                if hasattr(message_content, 'get'):
                    response_text = (
                        message_content.get('response') or 
                        message_content.get('text') or 
                        message_content.get('answer') or
                        message_content.get('output') or
                        message_content.get('result') or
                        str(message_content)
                    )
                    
                    if TableResponseParser.detect_table_content(response_text):
                        TableResponseParser.display_table_response(response_text, "Query Results")
                    else:
                        st.markdown(response_text)
                else:
                    st.markdown(str(message_content))
                    
        except Exception as e:
            st.error(f"Error rendering response: {str(e)}")
            st.text(f"Raw response: {chat['message']}")
        
        # Show metadata if available
        if "metadata" in chat and not chat["metadata"]["success"]:
            st.error(f"❌ Error: {chat['metadata']['error']}")
        elif "metadata" in chat and chat["metadata"].get("response_time"):
            st.caption(f"⏱️ Response time: {chat['metadata']['response_time']:.2f}s | Status: {chat['metadata']['status_code']}")

def simple_table_parser(response_text):
    """Fixed table parser that returns empty DataFrame instead of None"""
    import pandas as pd
    
    try:
        # Basic checks
        if not response_text or '|' not in response_text:
            return pd.DataFrame()  # Return empty DataFrame instead of None
            
        if response_text.count('|') < 6:  # Need multiple separators for a table
            return pd.DataFrame()  # Return empty DataFrame instead of None
        
        lines = response_text.split('\n')
        good_lines = []
        
        # Find lines that look like table rows
        for line in lines:
            line = line.strip()
            if (line and 
                '|' in line and 
                '---' not in line and
                'this table' not in line.lower() and
                'if you need' not in line.lower() and
                'feel free' not in line.lower() and
                'here is a summary' not in line.lower()):
                good_lines.append(line)
        
        if len(good_lines) < 2:
            return pd.DataFrame()  # Return empty DataFrame instead of None
            
        # Parse the first line as headers
        first_line = good_lines[0]
        headers = [h.strip() for h in first_line.split('|') if h.strip()]
        
        if len(headers) < 2:
            return pd.DataFrame()  # Return empty DataFrame instead of None
            
        # Parse remaining lines as data
        data_rows = []
        for line in good_lines[1:]:
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(row) >= len(headers):
                data_rows.append(row[:len(headers)])  # Take only as many columns as headers
        
        if data_rows:
            return pd.DataFrame(data_rows, columns=headers)
        else:
            return pd.DataFrame()  # Return empty DataFrame instead of None
        
    except Exception as e:
        print(f"Simple parser error: {e}")
        return pd.DataFrame()  # Return empty DataFrame instead of None
def display_table_response_fixed(response_text, title="Query Results"):
    """Fixed display function with proper None/empty checking"""
    try:
        # Try the simple parser
        df = simple_table_parser(response_text)
        
        # Check if we got a valid DataFrame with data
        if df is not None and not df.empty:
            st.subheader(f"📊 {title}")
            
            # Show basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Size", f"{len(df)} x {len(df.columns)}")
            
            # Display the table
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
            return True
        else:
            # Show as text if not a table or parsing failed
            st.markdown(response_text)
            return False
            
    except Exception as e:
        st.error(f"Display error: {str(e)}")
        st.text(response_text)
        return False
    
def render_chat_with_fixed_table_parsing():
    """Complete chat rendering with fixed table parsing"""
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["type"] == "user":
                with st.chat_message("user"):
                    formatted = chat["message"].encode('utf-8').decode('unicode_escape')
                    st.markdown(f"""**[{chat['timestamp']}]**
                    {escape(formatted)}""")            
                   
            else:
                with st.chat_message("assistant"):
                    st.write(f"""**[{chat['timestamp']}]**""")
                    
                    try:
                        message_content = chat["message"]
                        
                        # Handle JSON responses
                        if isinstance(message_content, str):
                            try:
                                data = json.loads(message_content)
                                response_text = (
                                    data.get('output') or 
                                    data.get('response') or 
                                    data.get('text') or 
                                    data.get('answer') or
                                    str(data)
                                )
                            except json.JSONDecodeError:
                                response_text = message_content
                        else:
                            response_text = str(message_content)
                        
                        # Display with fixed table support
                        display_table_response_fixed(response_text)
                        
                        # Handle charts if available
                        if isinstance(message_content, str):
                            try:
                                data = json.loads(message_content)
                                chart_data = data.get('chart_data')
                                if chart_data and chart_data.get('chart_json'):
                                    st.subheader("📊 Visualization")
                                    chart_fig = pio.from_json(chart_data['chart_json'])
                                    st.plotly_chart(chart_fig, use_container_width=True)
                            except:
                                pass
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.text(f"Raw: {chat['message']}")
                    
                    # Show metadata
                    if "metadata" in chat and not chat["metadata"]["success"]:
                        st.error(f"❌ Error: {chat['metadata']['error']}")
                    elif "metadata" in chat and chat["metadata"].get("response_time"):
                        st.caption(f"⏱️ {chat['metadata']['response_time']:.2f}s | Status: {chat['metadata']['status_code']}")

def simple_chat_display_fix():
    """Ultra-simple chat display that handles your specific table format"""
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["type"] == "user":
                with st.chat_message("user"):
                    formatted = chat["message"].encode('utf-8').decode('unicode_escape')
                    st.markdown(f"""**[{chat['timestamp']}]**
                    {escape(formatted)}""")            
                   
            else:
                with st.chat_message("assistant"):
                    st.write(f"""**[{chat['timestamp']}]**""")
                    
                    try:
                        message_content = chat["message"]
                        
                        # Extract the actual response
                        if isinstance(message_content, str):
                            try:
                                data = json.loads(message_content)
                                response_text = data.get('output', str(data))
                            except:
                                response_text = message_content
                        else:
                            response_text = str(message_content)
                        
                        # Simple table detection and parsing
                        if ('|' in response_text and 
                            response_text.count('|') > 10 and 
                            ('No |' in response_text or 'DocumentNr' in response_text)):
                            
                            # Extract table lines
                            lines = response_text.split('\n')
                            table_lines = []
                            
                            for line in lines:
                                line = line.strip()
                                if (line and '|' in line and 
                                    '---' not in line and 
                                    'here is a summary' not in line.lower() and
                                    'if you need' not in line.lower()):
                                    table_lines.append(line)
                            
                            if len(table_lines) >= 2:
                                try:
                                    # Parse headers - clean them up
                                    header_parts = table_lines[0].split('|')
                                    headers = []
                                    for part in header_parts:
                                        part = part.strip()
                                        if part:
                                            headers.append(part)
                                    
                                    # Parse data rows
                                    data_rows = []
                                    for line in table_lines[1:]:
                                        row_parts = line.split('|')
                                        row_data = []
                                        for part in row_parts:
                                            part = part.strip()
                                            if part:
                                                row_data.append(part)
                                        
                                        # Only add rows that have enough columns
                                        if len(row_data) >= len(headers):
                                            data_rows.append(row_data[:len(headers)])
                                    
                                    # Create and display DataFrame
                                    if headers and data_rows:
                                        import pandas as pd
                                        df = pd.DataFrame(data_rows, columns=headers)
                                        
                                        st.subheader("📊 Product Sales Table")
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Download option
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            "📥 Download CSV",
                                            csv,
                                            f"sales_table_{datetime.now().strftime('%H%M%S')}.csv",
                                            "text/csv",
                                            key=f"download_table_{chat['timestamp']}"
                                        )
                                    else:
                                        st.markdown(response_text)
                                        
                                except Exception as parse_error:
                                    st.error(f"Table parse error: {parse_error}")
                                    st.text(response_text)
                            else:
                                st.markdown(response_text)
                        else:
                            # Not a table, display as text
                            st.markdown(response_text)
                        
                    except Exception as e:
                        st.error(f"Chat display error: {str(e)}")
                        st.text(f"Raw message: {chat['message']}")

class WebhookManager:
    """Handles webhook operations with fix for empty responses"""
    
    @staticmethod
    def send_message(webhook_url: str, message: str, timeout: int = 180) -> Dict[str, Any]:
        """Send message to webhook and return response"""
        try:
            # Validate URL
            if not webhook_url.startswith(('http://', 'https://')):
                return {
                    "success": False,
                    "error": "Invalid webhook URL format",
                    "status_code": None,
                    "response_time": None
                }
            
            # Prepare the payload - optimized for n8n workflow with user context
            payload = {
                "message": message,
                "query": message,  # Add explicit query field for Vector Store Retriever
                "timestamp": datetime.now().isoformat(),
                "source": "streamlit_app"
            }
            
            # Add user context if available
            if st.session_state.current_user:
                payload["user"] = {
                    "username": st.session_state.current_user["username"],
                    "role": st.session_state.current_user["role"]
                }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Streamlit-QdrantApp/1.0'
            }
            
            # Send POST request to webhook
            start_time = time.time()
            response = requests.post(
                webhook_url, 
                json=payload, 
                headers=headers,
                timeout=timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # First check if response has any content at all
                if not response.text or response.text.strip() == "":
                    return {
                        "success": False,
                        "error": "Webhook returned completely empty response (no content)",
                        "status_code": response.status_code,
                        "response_time": response_time,
                        "debug_info": {
                            "response_headers": dict(response.headers),
                            "response_length": len(response.text) if response.text else 0
                        }
                    }
                
                try:
                    response_data = response.json()
                    # Extract the actual response text from n8n structure
                    if isinstance(response_data, dict):
                        # Handle different n8n response formats
                        actual_response = (
                            response_data.get('response') or 
                            response_data.get('text') or 
                            response_data.get('answer') or
                            response_data.get('result') or
                            response_data.get('output') or
                            str(response_data) if response_data else None
                        )
                    else:
                        actual_response = str(response_data)
                    
                    # Check if we got meaningful content
                    if not actual_response or actual_response.strip() == "" or actual_response == "{}":
                        return {
                            "success": False,
                            "error": f"Webhook returned empty response content. Raw response: {response_data}",
                            "status_code": response.status_code,
                            "response_time": response_time,
                            "raw_response": response_data
                        }
                    
                    return {
                        "success": True,
                        "response": actual_response,
                        "raw_response": response_data,
                        "status_code": response.status_code,
                        "response_time": response_time
                    }
                    
                except json.JSONDecodeError as e:
                    # The webhook returned non-JSON content
                    if response.text and response.text.strip():
                        # If it's non-empty text, treat it as the response
                        return {
                            "success": True,
                            "response": response.text.strip(),
                            "status_code": response.status_code,
                            "response_time": response_time,
                            "debug_info": {
                                "note": "Response was not JSON, treating as plain text",
                                "content_type": response.headers.get('content-type', 'unknown')
                            }
                        }
                    else:
                        # Empty text and failed JSON parsing
                        return {
                            "success": False,
                            "error": f"Webhook returned invalid/empty response. Content: '{response.text}' | JSON error: {str(e)}",
                            "status_code": response.status_code,
                            "response_time": response_time,
                            "debug_info": {
                                "response_headers": dict(response.headers),
                                "response_text": response.text,
                                "json_error": str(e)
                            }
                        }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code,
                    "response_time": response_time
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out",
                "status_code": None,
                "response_time": timeout
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}",
                "status_code": None,
                "response_time": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "status_code": None,
                "response_time": None
            }

def create_simple_customer_chart():
    """Create a simple customer sales chart when main chart fails"""
    import plotly.graph_objects as go
    
    # Sample customer data
    customers = ['Customer A', 'Customer B', 'Customer C', 'Customer D', 'Customer E']
    sales = [1250.50, 890.25, 1450.75, 675.00, 2100.30]
    
    fig = go.Figure(data=[
        go.Bar(
            x=customers,
            y=sales,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
            text=[f'${amt:,.2f}' for amt in sales],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Customer Sales Report',
        xaxis_title='Customers',
        yaxis_title='Sales Amount ($)',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig

def render_chat_tab():
    render_chat_tab_with_questions()

def render_chat_tab_with_questions():
    """Enhanced chat tab with CSV visualization support"""
    st.header("💬 Enhanced Webhook Chat with CSV Visualization")
    st.markdown("Chat with your documents and generate visualizations from CSV data")
    
    # Check for selected question from question manager
    selected_question = st.session_state.get("selected_question", "")
    if selected_question:
        st.success(f"🎯 **Selected Question:** {selected_question}")
        if st.button("🗑️ Clear Selected Question", key="clear_selected_question"):
            st.session_state.selected_question = ""
            st.rerun()
    
    # Show current model and collection info
    if st.session_state.model and st.session_state.user_collections:
        st.info(f"🤖 Using {st.session_state.model_type}: {st.session_state.model}")
        st.info(f"📚 Available collections: {', '.join(st.session_state.user_collections)}")
        
        # Collection selection for CSV visualization
        viz_collection = st.selectbox(
            "Select Collection for CSV Visualization",
            st.session_state.user_collections,
            key="viz_collection_select",
            help="Choose which collection to analyze for CSV data visualizations"
        )
    
    # Show user context
    if st.session_state.current_user:
        user_role = st.session_state.current_user.get("role")
        if user_role == "super_admin":
            st.info("👑 Super Admin: Your queries will have access to all documents")
        else:
            st.info("👤 User: Your queries will only access your own documents")
    
    # Webhook configuration (same as before)
    with st.expander("⚙️ Webhook Configuration", expanded=False):
        webhook_url = st.text_input(
            "Webhook URL", 
            value=st.session_state.webhook_url,
            placeholder="http://137.184.86.182:5678/webhook/acty",
            key="webhook_url_input"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            timeout = st.slider("Timeout (seconds)", 5, 60, 30, key="webhook_timeout")
        with col2:
            if st.button("💾 Save Webhook URL", key="save_webhook_btn"):
                st.session_state.webhook_url = webhook_url
                st.success("✅ Webhook URL saved!")
                auto_save_config_if_enabled()
        
        # Test webhook connection
        if st.button("🔍 Test Webhook Connection", key="test_webhook_btn"):
            if webhook_url.strip():
                with st.spinner("Testing webhook..."):
                    test_response = WebhookManager.send_message(webhook_url, "How to reduce hypertension?", timeout)
                    if test_response["success"]:
                        st.success(f"✅ Webhook is working! Response time: {test_response['response_time']:.2f}s")
                    else:
                        st.error(f"❌ Webhook test failed: {test_response['error']}")
            else:
                st.error("❌ Please enter a webhook URL")
    
    # CSV Visualization examples
    with st.expander("📊 CSV Visualization Examples", expanded=False):
        st.markdown("""
        **Example queries for CSV visualization:**
        - "Create a bar chart showing sales by month"
        - "Plot the relationship between price and quantity"
        - "Show a pie chart of product categories"
        - "Display a line chart of revenue trends"
        - "Generate a histogram of customer ages"
        - "Create a heatmap of correlations"
        
        **The AI will automatically:**
        1. Detect visualization requests in your query
        2. Extract CSV data from the selected collection
        3. Analyze the data structure
        4. Recommend and create the most appropriate chart
        5. Display the visualization in the chat
        6. Include charts in PDF exports
        """)
    
    # Chat interface
    st.subheader("💬 Chat Interface")
    
    # Question management integration (same as before)
    if st.session_state.current_user:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📋 Browse Saved Questions", key="browse_saved_questions"):
                st.session_state.show_question_manager = True
        
        with col2:
            # Quick access to recent/favorite questions
            username = st.session_state.current_user["username"]
            questions = QuestionManager.load_user_questions(username)
            if questions:
                favorite_questions = [q for q in questions if q.get("favorite", False)]
                if favorite_questions:
                    selected_fav = st.selectbox(
                        "⭐ Quick access to favorites",
                        ["Select a favorite question..."] + [q["question"] for q in favorite_questions],
                        key="quick_favorite_select"
                    )
                    if selected_fav != "Select a favorite question...":
                        st.session_state.selected_question = selected_fav
                        # Find and update usage
                        for q in favorite_questions:
                            if q["question"] == selected_fav:
                                QuestionManager.update_question_usage(username, q["id"])
                                break
                        st.rerun()
    
    # Display chat history with chart support
    chat_container = st.container()
    with chat_container:
        for chat_index, chat in enumerate(st.session_state.chat_history):
            if chat["type"] == "user":
                with st.chat_message("user"):
                    formatted = chat["message"].encode('utf-8').decode('unicode_escape')
                    st.markdown(f"""**[{chat['timestamp']}]**
                    {escape(formatted)}""")            
            
            else:
                with st.chat_message("assistant"):
                    st.write(f"""**[{chat['timestamp']}]**""")
                    
                    try:
                        message_content = chat["message"]
                        content_rendered = False
                        
                        # Step 1: Try to parse as JSON
                        try:
                            if isinstance(message_content, str):
                                data = json.loads(message_content)
                                response_text = data.get('response', '')
                                
                                # Method 1: Direct chart_data field
                                if 'chart_data' in data and data['chart_data']:
                                    st.write("**Method 1: Found chart_data field**")
                                    try:
                                        chart_json = data['chart_data'].get('chart_json', '')
                                        
                                        # Clean the chart JSON data
                                        if isinstance(chart_json, str) and chart_json.strip():
                                            try:
                                                chart_data = json.loads(chart_json)
                                                # Remove invalid properties
                                                invalid_keys = ['response', 'error', 'message', 'status']
                                                for key in invalid_keys:
                                                    if key in chart_data:
                                                        del chart_data[key]
                                                
                                                chart_json = json.dumps(chart_data)
                                            except:
                                                pass
                                            
                                            unique_key = f"chart_{chat_index}_{chat['timestamp']}"
                                            fig = pio.from_json(chart_json)
                                            st.plotly_chart(fig, use_container_width=True, key=unique_key)
                                            content_rendered = True
                                            st.success("Chart rendered successfully!")
                                            
                                    except Exception as e:
                                        st.error(f"Method 1 failed: {e}")
                                
                                # Method 2: Check for chart JSON in response
                                if not content_rendered and response_text and len(response_text) > 100:
                                    st.write("**Method 2: Checking response field for chart JSON**")
                                    try:
                                        # Only try if it looks like JSON
                                        if response_text.strip().startswith('{') and '"data":' in response_text:
                                            response_data = json.loads(response_text.strip())
                                            
                                            # Clean invalid properties
                                            invalid_keys = ['response', 'error', 'message', 'status']
                                            for key in invalid_keys:
                                                if key in response_data:
                                                    del response_data[key]
                                            
                                            fig = pio.from_json(json.dumps(response_data))
                                            method2_key = f"method2_chart_{chat_index}_{chat['timestamp']}"
                                            st.plotly_chart(fig, use_container_width=True, key=method2_key)
                                            content_rendered = True
                                            st.success("Chart rendered via Method 2")
                                            
                                    except Exception as e:
                                        st.error(f"Method 2 failed: {e}")
                                
                                # Method 3: SMART DETECTION - Check if it's table data
                                if not content_rendered:
                                    st.write("**Smart Detection: Analyzing response content**")
                                    
                                    # Check if the response contains table-like data
                                    if ('|' in response_text and response_text.count('|') > 5) or ('---' in response_text):
                                        st.write("**Table data detected - displaying as table**")
                                        try:
                                            df = extract_and_parse_table(response_text)
                                            if not df.empty:
                                                st.subheader("📊 Product Sales Data")
                                                st.dataframe(df, use_container_width=True)
                                                
                                                # Download option
                                                csv = df.to_csv(index=False)
                                                st.download_button(
                                                    "📥 Download CSV",
                                                    csv,
                                                    f"product_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                    "text/csv",
                                                    key=f"table_download_{chat_index}"
                                                )
                                                
                                                # Offer to create chart from this data
                                                if st.button(f"📊 Create Chart from This Data", key=f"chart_from_table_{chat_index}"):
                                                    try:
                                                        table_chart = create_chart_from_table(df)
                                                        chart_key = f"table_chart_{chat_index}_{chat['timestamp']}"
                                                        st.plotly_chart(table_chart, use_container_width=True, key=chart_key)
                                                        st.success("Chart created from table data!")
                                                    except Exception as e:
                                                        st.error(f"Chart creation from table failed: {e}")
                                                
                                                content_rendered = True
                                            else:
                                                st.write("Table parsing failed, showing raw text")
                                        except Exception as e:
                                            st.error(f"Table parsing error: {e}")
                                    
                                    # If not table data, check if it's a text response about specific products
                                    elif any(keyword in response_text.lower() for keyword in ['acty', 'product', 'sales', 'ไม่พบ', 'not found']):
                                        st.write("**Product-specific query detected**")
                                        
                                        # Show the text response
                                        st.info("📝 Response about specific product:")
                                        st.markdown(response_text)
                                        
                                        # Offer to create a sample chart
                                        if st.button(f"📊 Create Sample Chart for Product Analysis", key=f"product_sample_{chat_index}"):
                                            try:
                                                product_chart = create_product_analysis_chart()
                                                sample_key = f"product_chart_{chat_index}_{chat['timestamp']}"
                                                st.plotly_chart(product_chart, use_container_width=True, key=sample_key)
                                                st.success("Sample product analysis chart created!")
                                            except Exception as e:
                                                st.error(f"Sample chart creation failed: {e}")
                                        
                                        content_rendered = True
                                
                                # Fallback: Show response text
                                if not content_rendered:
                                    st.write("**No chart could be rendered. Showing text response:**")
                                    
                                    # Always offer sample chart option
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button(f"📊 Create Sample Customer Sales Chart", key=f"sample_btn_{chat_index}"):
                                            try:
                                                sample_fig = create_guaranteed_customer_chart()
                                                sample_key = f"sample_chart_{chat_index}_{chat['timestamp']}"
                                                st.plotly_chart(sample_fig, use_container_width=True, key=sample_key)
                                                st.success("Sample chart created!")
                                            except Exception as e:
                                                st.error(f"Sample chart creation failed: {e}")
                                    
                                    with col2:
                                        if st.button(f"📈 Create Product Analysis Chart", key=f"product_btn_{chat_index}"):
                                            try:
                                                product_fig = create_product_analysis_chart()
                                                product_key = f"product_chart_{chat_index}_{chat['timestamp']}"
                                                st.plotly_chart(product_fig, use_container_width=True, key=product_key)
                                                st.success("Product analysis chart created!")
                                            except Exception as e:
                                                st.error(f"Product chart creation failed: {e}")
                                    
                                    # Show response text
                                    st.text_area("Response (truncated)", response_text[:500] + "..." if len(response_text) > 500 else response_text, 
                                            height=100, key=f"resp_{chat_index}")
                                    
                                    if len(response_text) > 500:
                                        with st.expander("Show full response"):
                                            st.text(response_text)
                            
                            else:
                                st.markdown(str(message_content))
                        
                        except json.JSONDecodeError:
                            # Handle non-JSON responses
                            st.write("**Non-JSON response detected**")
                            
                            # Check if it's table data
                            if isinstance(message_content, str) and ('|' in message_content or '---' in message_content):
                                try:
                                    df = extract_and_parse_table(message_content)
                                    if not df.empty:
                                        st.subheader("📊 Data Table")
                                        st.dataframe(df, use_container_width=True)
                                        content_rendered = True
                                except:
                                    pass
                            
                            if not content_rendered:
                                st.markdown(str(message_content)[:500] + "..." if len(str(message_content)) > 500 else str(message_content))
                        
                    except Exception as e:
                        st.error(f"Error processing message: {str(e)}")
                    
                    # Show metadata
                    if "metadata" in chat and not chat["metadata"]["success"]:
                        st.error(f"Error: {chat['metadata']['error']}")
                    elif "metadata" in chat and chat["metadata"].get("response_time"):
                        st.caption(f"Response time: {chat['metadata']['response_time']:.2f}s")
                        
    # Enhanced chat input with question management
    with st.container():
        col1, col2, col3 = st.columns([4, 1, 1])
        
        with col1:
            # Use selected question or allow manual input
            if selected_question:
                user_message = st.text_input(
                    "Ask a question about your documents or request a visualization:",
                    value=selected_question,
                    key="webhook_chat_input"
                )
            else:
                user_message = st.text_input(
                    "Ask a question about your documents or request a visualization:",
                    placeholder="Create a bar chart showing the data trends...",
                    key="webhook_chat_input"
                )
        
        with col2:
            send_button = st.button("📤 Send", key="send_webhook_message_fixed")
        
        with col3:
            # Save current question button
            if user_message.strip() and st.session_state.current_user:
                if st.button("💾 Save Q", key="save_current_question", help="Save this question"):
                    st.session_state.show_save_question_modal = True
        
        # Save question modal (same as before)
        if st.session_state.get("show_save_question_modal", False) and user_message.strip():
            with st.container():
                st.info("💾 **Save This Question**")
                
                col1, col2 = st.columns(2)
                with col1:
                    save_category = st.selectbox(
                        "Category",
                        ["General", "Technical", "Business", "Research", "Personal", "Visualization"],
                        key="save_question_category"
                    )
                
                with col2:
                    save_tags = st.text_input(
                        "Tags (comma-separated)",
                        placeholder="ai, analysis, charts, csv",
                        key="save_question_tags"
                    )
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("✅ Save Question", key="confirm_save_question"):
                        username = st.session_state.current_user["username"]
                        tags = [tag.strip() for tag in save_tags.split(",") if tag.strip()] if save_tags else []
                        
                        if QuestionManager.add_question(username, user_message, save_category, tags):
                            st.success("✅ Question saved!")
                            st.session_state.show_save_question_modal = False
                            st.rerun()
                        else:
                            st.error("❌ Question already exists or failed to save")
                
                with col_b:
                    if st.button("❌ Cancel", key="cancel_save_question"):
                        st.session_state.show_save_question_modal = False
                        st.rerun()
        
        # Chat controls
        col3, col4, col5 = st.columns([1, 1, 1])
        with col3:
            if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col4:
            if st.button("📥 Export Chat", key="export_chat_btn"):
                if st.session_state.chat_history:
                    try:
                        # Initialize enhanced PDF generator
                        pdf_generator = EnhancedThaiPDFGenerator()
                        
                        # Generate PDF with charts support
                        with st.spinner("🔄 Generating Enhanced PDF with Charts..."):
                            pdf_data = pdf_generator.generate_chat_pdf_with_charts(
                                st.session_state.chat_history,
                                st.session_state.current_user
                            )
                        
                        # Offer download
                        st.download_button(
                            label="📄 Download Enhanced PDF",
                            data=pdf_data,
                            file_name=f"chat_export_with_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key="download_enhanced_pdf_btn"
                        )
                        
                        # Also offer JSON export with chart data
                        clean_chat = []
                        for chat in st.session_state.chat_history:
                            if chat["type"] == "user":
                                clean_chat.append({"question": chat["message"]})
                            elif chat["type"] == "assistant" and clean_chat:
                                try:
                                    message_content = chat["message"]
                                    if isinstance(message_content, str):
                                        try:
                                            data = json.loads(message_content)
                                            response_text = (
                                                data.get('response') or 
                                                data.get('text') or 
                                                data.get('answer') or
                                                str(data)
                                            )
                                            clean_chat[-1]["response"] = response_text
                                            
                                            # Include chart data if available
                                            if data.get('chart_data'):
                                                clean_chat[-1]["chart_data"] = data['chart_data']
                                                
                                        except json.JSONDecodeError:
                                            clean_chat[-1]["response"] = message_content
                                    else:
                                        clean_chat[-1]["response"] = str(message_content)
                                except Exception:
                                    clean_chat[-1]["response"] = str(chat["message"])
                        
                        st.download_button(
                            label="💾 Download JSON with Charts",
                            data=json.dumps(clean_chat, indent=2, ensure_ascii=False),
                            file_name=f"chat_export_with_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key="download_json_charts_btn"
                        )
                        
                        st.success("✅ Enhanced export options generated!")
                        
                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")
                        st.error("💡 Please ensure Plotly and kaleido are installed: `pip install plotly kaleido`")
                else:
                    st.info("📝 No chat history to export")
        
        with col5:
            debug_mode = st.checkbox("🔍 Debug Mode", key="debug_mode_cb")
    
    # Process chat message with CSV visualization support
    if send_button and user_message.strip():
        if not st.session_state.webhook_url.strip():
            st.error("❌ Please configure a webhook URL first")
            return
        
        if not viz_collection:
            st.error("❌ Please select a collection for CSV visualization")
            return
        
        # Clear selected question after sending
        if st.session_state.get("selected_question"):
            st.session_state.selected_question = ""
        
        # Add user message to history
        user_chat = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sender": "You",
            "message": user_message,
            "type": "user"
        }
        st.session_state.chat_history.append(user_chat)

        # Enhanced debugging information
        if debug_mode:
            st.write("🔍 **Pre-Request Debug Info:**")
            st.write(f"- Webhook URL: {st.session_state.webhook_url}")
            st.write(f"- Message: {user_message}")
            st.write(f"- Collection: {viz_collection}")
            st.write(f"- Timeout: {timeout}")
            
            # Test webhook connectivity
            st.write("**Testing webhook connectivity...**")
            test_response = WebhookManager.send_message(st.session_state.webhook_url, "connectivity test", 10)
            st.write(f"- Connection test result: {test_response.get('success', False)}")
            if not test_response.get('success'):
                st.error(f"Connection test failed: {test_response.get('error', 'Unknown error')}")
            else:
                st.success("Connection test passed")
            
            # Check if we can reach the server
            import requests
            try:
                # Try a simple GET request to the base URL
                base_url = "/".join(st.session_state.webhook_url.split("/")[:-2])  # Remove /webhook/acty
                get_response = requests.get(base_url, timeout=5)
                st.write(f"- Server accessibility (GET {base_url}): {get_response.status_code}")
            except Exception as e:
                st.write(f"- Server accessibility error: {str(e)}")
        
        # Send to enhanced webhook with CSV visualization support
        with st.spinner("📤 Processing message and generating visualizations..."):
            if debug_mode:
                st.write("🔍 **Sending enhanced webhook request...**")
                st.json({
                    "url": st.session_state.webhook_url,
                    "message": user_message,
                    "collection": viz_collection,
                    "timestamp": datetime.now().isoformat(),
                    "user": st.session_state.current_user,
                    "visualization_enabled": True
                })
            
            webhook_response = EnhancedWebhookManager.process_csv_visualization_request(
                st.session_state.webhook_url, 
                user_message,
                viz_collection,
                timeout
            )
            
            if debug_mode:
                st.write("📥 **Received enhanced webhook response:**")
                st.json(webhook_response)

                # Additional debug info if available
                if 'debug_info' in webhook_response:
                    st.write("🔍 **Debug Details:**")
                    st.json(webhook_response['debug_info'])

        # Enhanced error handling for empty responses
        if not webhook_response.get("success"):
            error_msg = webhook_response.get("error", "Unknown error")
            st.error(f"❌ Webhook Error: {error_msg}")
            
            # Provide specific guidance based on error type
            if "empty response" in error_msg.lower():
                st.warning("💡 **Troubleshooting Tips:**")
                st.write("1. Check if your webhook service is running properly")
                st.write("2. Verify the webhook URL is correct")
                st.write("3. Check webhook service logs for errors")
                st.write("4. Ensure your webhook returns JSON format like: `{\"response\": \"your message here\"}`")
            
            # Don't add failed responses to chat history
            return            

        # Add assistant response to history with chart data
        response_content = None
        if webhook_response.get("success"):
            response_data = {
                "response": webhook_response.get("response", "No response content")
            }
            
            # Include chart data if available
            if webhook_response.get("chart_data"):
                response_data["chart_data"] = webhook_response["chart_data"]
            
            #response_content = json.dumps(response_data)
            response_content = json.dumps(response_data, default=str)
        else:
            response_content = f"Error: {webhook_response.get('error', 'Unknown error')}"
        
        assistant_chat = assistant_chat = process_webhook_response_enhanced(webhook_response)
        st.session_state.chat_history.append(assistant_chat)
        
        # Clear input and refresh
        st.rerun()

def create_chart_from_table(df):
    """Create a chart from table data"""
    import plotly.graph_objects as go
    
    if len(df.columns) >= 2:
        # Use first column as x-axis, second as y-axis
        x_col = df.columns[0]
        y_col = df.columns[1]
        
        # Clean data
        x_data = df[x_col].astype(str).tolist()[:10]  # Limit to 10 items
        y_data = pd.to_numeric(df[y_col], errors='coerce').fillna(0).tolist()[:10]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_data,
            y=y_data,
            text=[f'{val:.1f}' if val > 0 else '0' for val in y_data],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'{y_col} by {x_col}',
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        raise ValueError("Not enough columns for chart creation")

def create_product_analysis_chart():
    """Create a product analysis chart"""
    import plotly.graph_objects as go
    
    products = ['Product Acty', 'Product Beta', 'Product Gamma', 'Product Delta']
    sales = [2500, 1800, 3200, 1200]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=products,
        y=sales,
        text=[f'${s:,}' for s in sales],
        textposition='auto',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='Product Sales Analysis',
        xaxis_title='Products',
        yaxis_title='Sales Amount ($)',
        template='plotly_white',
        height=500
    )
    
    return fig

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif hasattr(obj, 'dtype'):  # other pandas/numpy objects
            return str(obj)
        return super().default(obj)
    

def render_collections_tab():
    """Render collections overview tab with user isolation"""
    import time
    
    st.header("📊 Collections Overview")
    
    # Initialize delete confirmations in session state
    if 'delete_confirmations' not in st.session_state:
        st.session_state.delete_confirmations = {}
    
    # Show user access info
    if st.session_state.current_user:
        user_role = st.session_state.current_user.get("role")
        if user_role == "super_admin":
            st.info("👑 Super Admin: You can see and manage all collections")
            collections_to_show = st.session_state.collections
        else:
            st.info("👤 User: You can only see and manage your own collections")
            collections_to_show = st.session_state.user_collections
    else:
        collections_to_show = []
    
    if collections_to_show:
        for collection in collections_to_show:
            with st.expander(f"📂 Collection: {collection}", expanded=False):
                try:
                    collection_info = st.session_state.qdrant_client.get_collection(collection)
                    
                    # Basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Documents", collection_info.points_count)
                    with col2:
                        st.metric("Vector Size", f"{collection_info.config.params.vectors.size}D")
                    with col3:
                        st.metric("Distance", collection_info.config.params.vectors.distance.name)
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        inspect_key = f"inspect_{collection}_{hash(collection)}"
                        if st.button(f"🔍 Inspect Files", key=inspect_key):
                            # Store the inspection state in session state
                            st.session_state[f"inspecting_{collection}"] = True
                            st.rerun()
                    
                    # Show inspection results if inspection was clicked
                    if st.session_state.get(f"inspecting_{collection}", False):
                        try:
                            # Create user filter for search if not super admin
                            search_filter = None
                            if st.session_state.current_user:
                                user_role = st.session_state.current_user.get("role")
                                username = st.session_state.current_user.get("username")
                                
                                if user_role != "super_admin":
                                    search_filter = Filter(
                                        must=[
                                            FieldCondition(
                                                key="metadata.uploaded_by",
                                                match=MatchValue(value=username)
                                            )
                                        ]
                                    )
                            
                            # Get points with user filtering
                            if search_filter:
                                all_points, _ = st.session_state.qdrant_client.scroll(
                                    collection_name=collection,
                                    scroll_filter=search_filter,
                                    limit=1000,
                                    with_payload=True
                                )
                            else:
                                all_points, _ = st.session_state.qdrant_client.scroll(
                                    collection_name=collection,
                                    limit=1000,
                                    with_payload=True
                                )
                            
                            if all_points:
                                # Group points by filename/source to show files instead of individual chunks
                                files_data = {}
                                
                                for point in all_points:
                                    payload = point.payload
                                    metadata = payload.get('metadata', {})
                                    
                                    # Get filename from metadata
                                    filename = metadata.get('filename') or metadata.get('source') or 'Unknown File'
                                    
                                    if filename not in files_data:
                                        files_data[filename] = {
                                            'points': [],
                                            'total_chunks': 0,
                                            'file_type': metadata.get('file_type', 'Unknown'),
                                            'file_size': metadata.get('file_size', 0),
                                            'upload_timestamp': metadata.get('upload_timestamp', 'Unknown'),
                                            'uploaded_by': metadata.get('uploaded_by', 'Unknown'),
                                            'total_text_length': 0,
                                            'chunk_types': set(),
                                            'page_numbers': set()
                                        }
                                    
                                    files_data[filename]['points'].append(point)
                                    files_data[filename]['total_chunks'] += 1
                                    
                                    # Aggregate file statistics
                                    text_content = (
                                        payload.get('pageContent') or 
                                        payload.get('text') or 
                                        payload.get('content') or 
                                        ''
                                    )
                                    files_data[filename]['total_text_length'] += len(str(text_content))
                                    
                                    if metadata.get('chunk_type'):
                                        files_data[filename]['chunk_types'].add(metadata['chunk_type'])
                                    if metadata.get('page_number'):
                                        files_data[filename]['page_numbers'].add(metadata['page_number'])
                                
                                st.write(f"**📁 Files in Collection ({len(files_data)} files):**")
                                
                                # Display each file with its aggregated information
                                for filename, file_info in files_data.items():
                                    if filename == 'Unknown File':
                                        continue  # Skip unknown files for now
                                    
                                    # Create unique key for this file
                                    file_id = f"{collection}_{filename}"
                                    
                                    with st.container():
                                        # Check if this file is in confirmation mode
                                        is_confirming = st.session_state.delete_confirmations.get(file_id, False)
                                        
                                        if is_confirming:
                                            # CONFIRMATION MODE - Show big red warning
                                            st.error(f"🚨 **CONFIRM DELETE FILE: {filename}**")
                                            st.write(f"**This will delete {file_info['total_chunks']} chunks from this file**")
                                            st.write("**This action cannot be undone!**")
                                            
                                            # Confirmation buttons
                                            conf_col1, conf_col2, conf_col3 = st.columns([1, 1, 2])
                                            
                                            with conf_col1:
                                                if st.button("🔥 DELETE FILE", key=f"confirm_yes_{file_id}", type="primary"):
                                                    try:
                                                        with st.spinner(f"🔄 Deleting file '{filename}' ({file_info['total_chunks']} chunks)..."):
                                                            
                                                            # Get count before deletion
                                                            before_count = st.session_state.qdrant_client.get_collection(collection).points_count
                                                            
                                                            # Create filter to match all chunks from this file
                                                            delete_filter = Filter(
                                                                should=[  # Use 'should' (OR) to catch both filename and source
                                                                    FieldCondition(
                                                                        key="metadata.filename",
                                                                        match=MatchValue(value=filename)
                                                                    ),
                                                                    FieldCondition(
                                                                        key="metadata.source",
                                                                        match=MatchValue(value=filename)
                                                                    )
                                                                ]
                                                            )
                                                            
                                                            # Add user filter for non-super admin users
                                                            if st.session_state.current_user and st.session_state.current_user.get("role") != "super_admin":
                                                                delete_filter.must = [
                                                                    FieldCondition(
                                                                        key="metadata.uploaded_by",
                                                                        match=MatchValue(value=st.session_state.current_user["username"])
                                                                    )
                                                                ]
                                                            
                                                            # Perform deletion
                                                            result = st.session_state.qdrant_client.delete(
                                                                collection_name=collection,
                                                                points_selector=delete_filter
                                                            )
                                                            
                                                            # Wait for operation to complete
                                                            time.sleep(1)
                                                            
                                                            # Check if deletion was successful
                                                            after_count = st.session_state.qdrant_client.get_collection(collection).points_count
                                                            deleted_count = before_count - after_count
                                                            
                                                            if deleted_count > 0:
                                                                st.success(f"✅ Successfully deleted file '{filename}'!")
                                                                st.success(f"📊 Deleted {deleted_count} chunks | Collection size: {before_count} → {after_count}")
                                                                
                                                                # Clear confirmation state
                                                                st.session_state.delete_confirmations[file_id] = False
                                                                
                                                                # Clear inspection state to refresh the view
                                                                st.session_state[f"inspecting_{collection}"] = False
                                                                
                                                                st.balloons()
                                                                time.sleep(1)
                                                                st.rerun()
                                                            else:
                                                                st.error("❌ No chunks were deleted - file may not exist or you don't have permission")
                                                                
                                                    except Exception as e:
                                                        st.error(f"❌ Deletion error: {str(e)}")
                                            
                                            with conf_col2:
                                                if st.button("❌ CANCEL", key=f"confirm_no_{file_id}"):
                                                    # Clear confirmation state
                                                    st.session_state.delete_confirmations[file_id] = False
                                                    st.info("✅ Cancelled file deletion")
                                                    st.rerun()
                                            
                                            with conf_col3:
                                                st.write("") # Empty space
                                        
                                        else:
                                            # NORMAL MODE - Show file info and delete button
                                            file_header_col1, file_header_col2 = st.columns([5, 1])
                                            
                                            with file_header_col1:
                                                st.write(f"**📄 {filename}**")
                                                
                                                # Show owner info if super admin
                                                if st.session_state.current_user and st.session_state.current_user.get("role") == "super_admin":
                                                    uploaded_by = file_info.get('uploaded_by', 'Unknown')
                                                    st.caption(f"👤 Owner: {uploaded_by}")
                                                
                                                # Show file statistics
                                                stats_parts = []
                                                stats_parts.append(f"{file_info['total_chunks']} chunks")
                                                
                                                if file_info['file_type'] != 'Unknown':
                                                    stats_parts.append(f"Type: {file_info['file_type']}")
                                                
                                                if file_info['file_size'] > 0:
                                                    # Format file size
                                                    if file_info['file_size'] > 1024*1024:
                                                        size_str = f"{file_info['file_size']/(1024*1024):.2f} MB"
                                                    elif file_info['file_size'] > 1024:
                                                        size_str = f"{file_info['file_size']/1024:.2f} KB"
                                                    else:
                                                        size_str = f"{file_info['file_size']} bytes"
                                                    stats_parts.append(f"Size: {size_str}")
                                                
                                                if file_info['chunk_types']:
                                                    chunk_types_str = ", ".join(file_info['chunk_types'])
                                                    stats_parts.append(f"Chunks: {chunk_types_str}")
                                                
                                                st.caption(" | ".join(stats_parts))
                                            
                                            with file_header_col2:
                                                # Only show delete button if user owns the file or is super admin
                                                can_delete = False
                                                if st.session_state.current_user:
                                                    user_role = st.session_state.current_user.get("role")
                                                    username = st.session_state.current_user.get("username")
                                                    uploaded_by = file_info.get('uploaded_by', 'Unknown')
                                                    
                                                    if user_role == "super_admin" or username == uploaded_by:
                                                        can_delete = True
                                                
                                                if can_delete:
                                                    if st.button("🗑️", key=f"delete_file_{file_id}", help=f"Delete entire file: {filename}", type="secondary"):
                                                        # Set confirmation state
                                                        st.session_state.delete_confirmations[file_id] = True
                                                        st.rerun()
                                                else:
                                                    st.write("🔒")  # Locked icon for files user can't delete
                                            
                                            # Show file details in collapsible section
                                            show_details_key = f"show_details_{file_id}"
                                            if show_details_key not in st.session_state:
                                                st.session_state[show_details_key] = False
                                            
                                            if st.button(f"📋 {'Hide' if st.session_state[show_details_key] else 'Show'} File Details", key=f"toggle_details_{file_id}"):
                                                st.session_state[show_details_key] = not st.session_state[show_details_key]
                                                st.rerun()
                                            
                                            if st.session_state[show_details_key]:
                                                # Show file details
                                                st.write("**📁 File Information:**")
                                                detail_col1, detail_col2 = st.columns(2)
                                                
                                                with detail_col1:
                                                    st.write(f"• **Total Chunks:** {file_info['total_chunks']}")
                                                    st.write(f"• **File Type:** {file_info['file_type']}")
                                                    if file_info['file_size'] > 0:
                                                        if file_info['file_size'] > 1024*1024:
                                                            size_display = f"{file_info['file_size']/(1024*1024):.2f} MB"
                                                        elif file_info['file_size'] > 1024:
                                                            size_display = f"{file_info['file_size']/1024:.2f} KB"
                                                        else:
                                                            size_display = f"{file_info['file_size']} bytes"
                                                        st.write(f"• **File Size:** {size_display}")
                                                
                                                with detail_col2:
                                                    st.write(f"• **Total Text:** {file_info['total_text_length']:,} characters")
                                                    if file_info['chunk_types']:
                                                        st.write(f"• **Chunk Types:** {', '.join(file_info['chunk_types'])}")
                                                    if file_info['page_numbers']:
                                                        page_list = sorted(file_info['page_numbers'])
                                                        if len(page_list) <= 10:
                                                            st.write(f"• **Pages:** {', '.join(map(str, page_list))}")
                                                        else:
                                                            st.write(f"• **Pages:** {page_list[0]}-{page_list[-1]} ({len(page_list)} pages)")
                                                
                                                # Show upload timestamp and owner info
                                                if file_info['upload_timestamp'] != 'Unknown':
                                                    st.write(f"• **Uploaded:** {file_info['upload_timestamp']}")
                                                
                                                if st.session_state.current_user and st.session_state.current_user.get("role") == "super_admin":
                                                    st.write(f"• **Uploaded By:** {file_info.get('uploaded_by', 'Unknown')}")
                                                
                                                # Sample content from first chunk
                                                if file_info['points']:
                                                    first_chunk = file_info['points'][0]
                                                    sample_text = (
                                                        first_chunk.payload.get('pageContent') or 
                                                        first_chunk.payload.get('text') or 
                                                        first_chunk.payload.get('content') or 
                                                        'No content found'
                                                    )
                                                    
                                                    st.write("**📝 Sample Content (First Chunk):**")
                                                    if len(str(sample_text)) > 300:
                                                        st.write(str(sample_text)[:300] + "...")
                                                        
                                                        # Option to show full first chunk
                                                        show_full_key = f"show_full_sample_{file_id}"
                                                        if st.button("📖 Show Full First Chunk", key=f"toggle_sample_{file_id}"):
                                                            if show_full_key not in st.session_state:
                                                                st.session_state[show_full_key] = False
                                                            st.session_state[show_full_key] = not st.session_state[show_full_key]
                                                            st.rerun()
                                                        
                                                        if st.session_state.get(show_full_key, False):
                                                            st.text_area("Full First Chunk", str(sample_text), height=200, key=f"sample_content_{file_id}")
                                                    else:
                                                        st.write(str(sample_text))
                                    
                                    # Add visual separator between files
                                    st.divider()
                                
                                # Handle unknown files separately
                                if 'Unknown File' in files_data:
                                    unknown_info = files_data['Unknown File']
                                    st.warning(f"⚠️ Found {unknown_info['total_chunks']} chunks without filename metadata")
                                
                                # Add button to hide inspection results
                                if st.button("🔼 Hide File Inspection", key=f"hide_inspect_{collection}"):
                                    st.session_state[f"inspecting_{collection}"] = False
                                    st.rerun()
                            else:
                                st.warning("No documents found in collection or you don't have access")
                                        
                        except Exception as e:
                            st.error(f"❌ Failed to inspect collection: {str(e)}")
                            st.write(f"Error details: {str(e)}")
                    
                    with col2:
                        stats_key = f"stats_{collection}_{hash(collection)}"
                        if st.button(f"📈 Stats", key=stats_key):
                            try:
                                # Get collection info first
                                collection_info = st.session_state.qdrant_client.get_collection(collection)
                                total_points = collection_info.points_count
                                
                                st.write("**Collection Statistics:**")
                                st.write(f"- Total documents: {total_points}")
                                st.write(f"- Vector dimension: {collection_info.config.params.vectors.size}")
                                
                                if total_points > 0:
                                    # Apply user filter for non-super admin users
                                    search_filter = None
                                    if st.session_state.current_user:
                                        user_role = st.session_state.current_user.get("role")
                                        username = st.session_state.current_user.get("username")
                                        
                                        if user_role != "super_admin":
                                            search_filter = Filter(
                                                must=[
                                                    FieldCondition(
                                                        key="metadata.uploaded_by",
                                                        match=MatchValue(value=username)
                                                    )
                                                ]
                                            )
                                    
                                    # Get sample for analysis with user filtering
                                    sample_size = min(100, total_points)
                                    if search_filter:
                                        points, _ = st.session_state.qdrant_client.scroll(
                                            collection_name=collection,
                                            scroll_filter=search_filter,
                                            limit=sample_size,
                                            with_payload=True
                                        )
                                    else:
                                        points, _ = st.session_state.qdrant_client.scroll(
                                            collection_name=collection,
                                            limit=sample_size,
                                            with_payload=True
                                        )
                                    
                                    if points:
                                        # Analyze content and group by files
                                        text_lengths = []
                                        chunk_types = {}
                                        files_count = set()
                                        file_types = set()
                                        uploaders = set()
                                        
                                        for point in points:
                                            payload = point.payload
                                            
                                            # Get text content from various possible fields
                                            text = (
                                                payload.get('pageContent') or 
                                                payload.get('text') or 
                                                payload.get('content') or 
                                                ''
                                            )
                                            
                                            if text:
                                                text_lengths.append(len(str(text)))
                                            
                                            # Analyze metadata
                                            metadata = payload.get('metadata', {})
                                            if metadata:
                                                chunk_type = metadata.get('chunk_type', 'unknown')
                                                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                                                
                                                # Count unique files
                                                filename = metadata.get('filename') or metadata.get('source')
                                                if filename:
                                                    files_count.add(filename)
                                                    
                                                if 'file_type' in metadata:
                                                    file_types.add(metadata['file_type'])
                                                
                                                # Track uploaders (for super admin)
                                                if st.session_state.current_user and st.session_state.current_user.get("role") == "super_admin":
                                                    uploader = metadata.get('uploaded_by', 'Unknown')
                                                    uploaders.add(uploader)
                                        
                                        # Display statistics
                                        st.write(f"- **Accessible files:** {len(files_count)}")
                                        st.write(f"- **Accessible chunks:** {len(points)}")
                                        
                                        if st.session_state.current_user and st.session_state.current_user.get("role") == "super_admin" and uploaders:
                                            st.write(f"- **File owners:** {', '.join(uploaders)}")
                                        
                                        if text_lengths:
                                            st.write("**Content Analysis:**")
                                            st.write(f"- Average chunk length: {np.mean(text_lengths):.0f} characters")
                                            st.write(f"- Min/Max chunk length: {min(text_lengths)}/{max(text_lengths)} characters")
                                            st.write(f"- Total characters: {sum(text_lengths):,}")
                                        
                                        if chunk_types:
                                            st.write("**Chunk Types:**")
                                            for chunk_type, count in chunk_types.items():
                                                percentage = (count / len(points)) * 100
                                                st.write(f"- {chunk_type}: {count} ({percentage:.1f}%)")
                                        
                                        if file_types:
                                            st.write("**File Types:**")
                                            st.write(f"- {', '.join(file_types)}")
                                        
                                        st.write(f"**Analysis based on {len(points)} accessible chunks from {len(files_count)} files**")
                                    else:
                                        st.warning("Could not retrieve document samples or no access to documents")
                                else:
                                    st.info("Collection is empty")
                                
                            except Exception as e:
                                st.error(f"❌ Failed to get stats: {str(e)}")
                                st.write(f"Error details: {str(e)}")
                    
                    with col3:
                        delete_key = f"delete_{collection}_{hash(collection)}"
                        
                        # Only show delete button if user can delete the collection
                        can_delete_collection = False
                        if st.session_state.current_user:
                            user_role = st.session_state.current_user.get("role")
                            username = st.session_state.current_user.get("username")
                            
                            if user_role == "super_admin":
                                can_delete_collection = True
                            elif collection.startswith(get_user_collection_prefix(username)):
                                can_delete_collection = True
                        
                        if can_delete_collection:
                            if st.button(f"🗑️ Delete Collection", key=delete_key):
                                # Use session state to track confirmation
                                confirm_key = f"confirm_delete_{collection}"
                                if confirm_key not in st.session_state:
                                    st.session_state[confirm_key] = False
                                
                                st.session_state[confirm_key] = True
                            
                            # Show confirmation if delete was clicked
                            confirm_key = f"confirm_delete_{collection}"
                            if st.session_state.get(confirm_key, False):
                                st.warning(f"⚠️ Are you sure you want to delete '{collection}'?")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if st.button(f"✅ Yes, Delete", key=f"yes_delete_{collection}"):
                                        try:
                                            st.session_state.qdrant_client.delete_collection(collection)
                                            if collection in st.session_state.collections:
                                                st.session_state.collections.remove(collection)
                                            if collection in st.session_state.user_collections:
                                                st.session_state.user_collections.remove(collection)
                                            st.session_state[confirm_key] = False
                                            st.success(f"✅ Deleted collection: {collection}")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Failed to delete collection: {str(e)}")
                                with col_b:
                                    if st.button(f"❌ Cancel", key=f"cancel_delete_{collection}"):
                                        st.session_state[confirm_key] = False
                                        st.rerun()
                        else:
                            st.write("🔒 No permission")  # User can't delete this collection
                
                except Exception as e:
                    st.error(f"❌ Failed to get collection info: {str(e)}")
    else:
        st.info("📁 No collections found. Create your first collection in the Upload tab!")

# Add installation check for ReportLab
def check_reportlab_installation():
    """Check if ReportLab is installed and show installation instructions"""
    try:
        import reportlab
        return True
    except ImportError:
        st.error("📦 ReportLab not installed!")
        st.markdown("""
        **To install ReportLab for Thai-English PDF support:**
        ```bash
        pip install reportlab
        ```
        
        **For better Thai font support, also install:**
        ```bash
        # Download Thai fonts (optional)
        pip install google-fonts-noto
        ```
        """)
        return False
    
def render_status_indicators():
    """Render connection and model status"""
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.connection_status:
            st.success("🟢 Qdrant Connected")
        else:
            st.error("🔴 Qdrant Disconnected")
    
    with col2:
        if st.session_state.model_status:
            st.success(f"🟢 Model Loaded: {st.session_state.model_type}")
        else:
            st.error("🔴 No Model Loaded")
    
    with col3:
        if st.session_state.current_user:
            collections_count = len(st.session_state.user_collections)
            st.info(f"📚 Collections: {collections_count}")
        else:
            st.error("🔴 Not Authenticated")

class QdrantManager:
    """Handles Qdrant operations"""
    
    @staticmethod
    def connect(host: str = "......", port: int = 6333) -> Optional[QdrantClient]:
        """Connect to Qdrant instance with proper error handling"""
        try:
            client = QdrantClient(host=host, port=port)
            # Test connection
            client.get_collections()
            return client
        except ConnectionError:
            st.error("❌ Cannot connect to Qdrant. Is the service running?")
        except TimeoutError:
            st.error("❌ Qdrant connection timed out")
        except Exception as e:
            st.error(f"❌ Failed to connect to Qdrant: {str(e)}")
        return None
    
    @staticmethod
    def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 1536) -> bool:
        """Create a new collection in Qdrant"""
        try:
            # Check if collection already exists
            try:
                existing_collection = client.get_collection(collection_name)
                st.warning(f"Collection '{collection_name}' already exists with vector size {existing_collection.config.params.vectors.size}")
                return False
            except:
                # Collection doesn't exist, proceed to create
                pass
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            st.success(f"✅ Created collection '{collection_name}' with vector size {vector_size}")
            return True
        except Exception as e:
            st.error(f"❌ Failed to create collection: {str(e)}")
            return False
    
    @staticmethod
    def check_compatibility(client: QdrantClient, collection_name: str, expected_vector_size: int) -> bool:
        """Check if collection exists and has compatible vector size"""
        try:
            collection_info = client.get_collection(collection_name)
            actual_size = collection_info.config.params.vectors.size
            if actual_size != expected_vector_size:
                st.error(f"❌ Vector dimension mismatch! Collection '{collection_name}' expects {actual_size}D vectors, but your model produces {expected_vector_size}D vectors.")
                return False
            return True
        except Exception:
            # Collection doesn't exist, which is fine
            return True

def render_question_manager_sidebar_fixed():
    """Fixed question manager in sidebar with Use buttons"""
    if not st.session_state.current_user:
        return None
    
    username = st.session_state.current_user["username"]
    
    with st.sidebar.expander("💾 Saved Questions", expanded=False):
        questions = QuestionManager.load_user_questions(username)
        
        if questions:
            st.write(f"📊 **{len(questions)} saved questions**")
            
            # Quick stats
            favorites = len([q for q in questions if q.get("favorite", False)])
            if favorites > 0:
                st.write(f"⭐ {favorites} favorites")
            
            # Quick access to recent questions
            try:
                recent_questions = sorted(
                    questions, 
                    key=lambda x: x.get("last_used", "") or "",
                    reverse=True
                )[:3]
                
                # Filter out questions that have never been used
                recent_questions = [q for q in recent_questions if q.get("last_used")]
                
                if recent_questions:
                    st.write("**🕒 Recent:**")
                    for q in recent_questions:
                        question_text = q["question"]
                        if len(question_text) > 25:
                            question_text = question_text[:25] + "..."
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(question_text)
                        with col2:
                            if st.button("🎯", key=f"sidebar_recent_{q['id']}", help="Use this question"):
                                QuestionManager.update_question_usage(username, q["id"])
                                st.session_state.selected_question = q["question"]
                                st.rerun()
                
            except Exception as e:
                st.error(f"Error loading recent questions: {str(e)}")
            
            # Quick access to favorites
            try:
                favorite_questions = [q for q in questions if q.get("favorite", False)]
                if favorite_questions:
                    st.write("**⭐ Favorites:**")
                    for q in favorite_questions[:2]:  # Show top 2 favorites
                        question_text = q["question"]
                        if len(question_text) > 25:
                            question_text = question_text[:25] + "..."
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(question_text)
                        with col2:
                            if st.button("🎯", key=f"sidebar_fav_{q['id']}", help="Use this question"):
                                QuestionManager.update_question_usage(username, q["id"])
                                st.session_state.selected_question = q["question"]
                                st.rerun()
                            
            except Exception as e:
                st.error(f"Error loading favorite questions: {str(e)}")
        else:
            st.write("📝 No saved questions yet")
        
        # Link to main question manager
        if st.button("🔧 Manage Questions", key="sidebar_manage_questions"):
            st.session_state.show_question_manager = True
    
    return None

class QuestionManager:
    """Handles saving and retrieving user questions"""
    
    @staticmethod
    def ensure_questions_dir():
        """Ensure saved questions directory exists"""
        if not os.path.exists(SAVED_QUESTIONS_DIR):
            os.makedirs(SAVED_QUESTIONS_DIR)
    
    @staticmethod
    def get_user_questions_path(username: str) -> str:
        """Get the questions file path for a specific user"""
        QuestionManager.ensure_questions_dir()
        return os.path.join(SAVED_QUESTIONS_DIR, f"{username}_questions.json")
    
    @staticmethod
    def load_user_questions(username: str) -> List[Dict[str, Any]]:
        """Load saved questions for a user"""
        try:
            questions_path = QuestionManager.get_user_questions_path(username)
            if os.path.exists(questions_path):
                with open(questions_path, 'r', encoding='utf-8') as f:
                    questions_data = json.load(f)
                return questions_data.get("questions", [])
            else:
                return []
        except Exception as e:
            st.error(f"❌ Error loading saved questions: {str(e)}")
            return []
    
    @staticmethod
    def save_user_questions(username: str, questions: List[Dict[str, Any]]) -> bool:
        """Save questions for a user"""
        try:
            questions_path = QuestionManager.get_user_questions_path(username)
            
            questions_data = {
                "user": username,
                "questions": questions,
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(questions_path, 'w', encoding='utf-8') as f:
                json.dump(questions_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            st.error(f"❌ Error saving questions: {str(e)}")
            return False
    
    @staticmethod
    def add_question(username: str, question: str, category: str = "General", tags: List[str] = None) -> bool:
        """Add a new question to user's saved questions with proper field initialization"""
        try:
            questions = QuestionManager.load_user_questions(username)
            
            # Check for duplicates
            for existing_q in questions:
                if (existing_q.get("question", "") or "").strip().lower() == question.strip().lower():
                    return False  # Duplicate found
            
            new_question = {
                "id": str(uuid.uuid4()),
                "question": question.strip(),
                "category": category or "General",
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "used_count": 0,
                "last_used": "",  # Initialize as empty string instead of None
                "favorite": False
            }
            
            questions.append(new_question)
            return QuestionManager.save_user_questions(username, questions)
        except Exception as e:
            st.error(f"❌ Error adding question: {str(e)}")
            return False
    
    @staticmethod
    def update_question_usage(username: str, question_id: str) -> bool:
        """Update question usage statistics with proper None handling"""
        try:
            questions = QuestionManager.load_user_questions(username)
            
            for question in questions:
                if question.get("id") == question_id:
                    question["used_count"] = (question.get("used_count") or 0) + 1
                    question["last_used"] = datetime.now().isoformat()
                    break
            
            return QuestionManager.save_user_questions(username, questions)
        except Exception as e:
            st.error(f"❌ Error updating question usage: {str(e)}")
            return False
    
    @staticmethod
    def delete_question(username: str, question_id: str) -> bool:
        """Delete a saved question"""
        try:
            questions = QuestionManager.load_user_questions(username)
            questions = [q for q in questions if q.get("id") != question_id]
            return QuestionManager.save_user_questions(username, questions)
        except Exception as e:
            st.error(f"❌ Error deleting question: {str(e)}")
            return False
    
    @staticmethod
    def toggle_favorite(username: str, question_id: str) -> bool:
        """Toggle favorite status of a question"""
        try:
            questions = QuestionManager.load_user_questions(username)
            
            for question in questions:
                if question.get("id") == question_id:
                    question["favorite"] = not question.get("favorite", False)
                    break
            
            return QuestionManager.save_user_questions(username, questions)
        except Exception as e:
            st.error(f"❌ Error toggling favorite: {str(e)}")
            return False
    
    @staticmethod
    def get_questions_by_category(username: str, category: str = None) -> List[Dict[str, Any]]:
        """Get questions filtered by category"""
        questions = QuestionManager.load_user_questions(username)
        
        if category and category != "All":
            return [q for q in questions if q.get("category") == category]
        
        return questions
    
    @staticmethod
    def get_question_categories(username: str) -> List[str]:
        """Get all categories used by the user"""
        questions = QuestionManager.load_user_questions(username)
        categories = set()
        
        for question in questions:
            category = question.get("category", "General")
            categories.add(category)
        
        return sorted(list(categories)) if categories else ["General"]
    
    @staticmethod
    def search_questions(username: str, search_term: str) -> List[Dict[str, Any]]:
        """Search questions by text content or tags with None handling"""
        questions = QuestionManager.load_user_questions(username)
        search_term = search_term.lower().strip()
        
        if not search_term:
            return questions
        
        filtered_questions = []
        for question in questions:
            try:
                # Search in question text
                question_text = question.get("question", "") or ""
                if search_term in question_text.lower():
                    filtered_questions.append(question)
                    continue
                
                # Search in tags
                tags = question.get("tags", []) or []
                if any(search_term in (tag or "").lower() for tag in tags):
                    filtered_questions.append(question)
                    continue
                
                # Search in category
                category = question.get("category", "") or ""
                if search_term in category.lower():
                    filtered_questions.append(question)
            except Exception as e:
                # Skip this question if there's an error
                continue
        
        return filtered_questions
    
    @staticmethod
    def export_questions(username: str) -> str:
        """Export user questions as JSON"""
        questions = QuestionManager.load_user_questions(username)
        export_data = {
            "user": username,
            "questions": questions,
            "exported_at": datetime.now().isoformat(),
            "total_questions": len(questions)
        }
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def get_recent_questions(username: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recently used questions with proper None handling"""
        try:
            questions = QuestionManager.load_user_questions(username)
            
            # Filter out questions that have never been used
            used_questions = [q for q in questions if q.get("last_used") and q.get("last_used") != ""]
            
            # Sort by last_used date
            used_questions.sort(
                key=lambda x: x.get("last_used", ""),
                reverse=True
            )
            
            return used_questions[:limit]
        except Exception as e:
            st.error(f"❌ Error getting recent questions: {str(e)}")
            return []
        
    @staticmethod
    def get_popular_questions(username: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most used questions with proper None handling"""
        try:
            questions = QuestionManager.load_user_questions(username)
            
            # Filter questions that have been used
            used_questions = [q for q in questions if (q.get("used_count") or 0) > 0]
            
            # Sort by usage count
            used_questions.sort(
                key=lambda x: x.get("used_count", 0) or 0,
                reverse=True
            )
            
            return used_questions[:limit]
        except Exception as e:
            st.error(f"❌ Error getting popular questions: {str(e)}")
            return []
    
    @staticmethod
    def import_questions(username: str, questions_json: str) -> bool:
        """Import questions from JSON"""
        try:
            import_data = json.loads(questions_json)
            imported_questions = import_data.get("questions", [])
            
            if not imported_questions:
                st.warning("⚠️ No questions found in import data")
                return False
            
            existing_questions = QuestionManager.load_user_questions(username)
            existing_texts = {q.get("question", "").strip().lower() for q in existing_questions}
            
            new_questions = []
            duplicates = 0
            
            for question in imported_questions:
                question_text = question.get("question", "").strip()
                if question_text.lower() not in existing_texts:
                    # Assign new ID and update timestamps
                    question["id"] = str(uuid.uuid4())
                    question["created_at"] = datetime.now().isoformat()
                    question["used_count"] = 0
                    question["last_used"] = None
                    new_questions.append(question)
                    existing_texts.add(question_text.lower())
                else:
                    duplicates += 1
            
            if new_questions:
                all_questions = existing_questions + new_questions
                success = QuestionManager.save_user_questions(username, all_questions)
                if success:
                    st.success(f"✅ Imported {len(new_questions)} new questions")
                    if duplicates > 0:
                        st.info(f"ℹ️ Skipped {duplicates} duplicate questions")
                return success
            else:
                st.warning("⚠️ No new questions to import (all duplicates)")
                return False
                
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON format")
            return False
        except Exception as e:
            st.error(f"❌ Error importing questions: {str(e)}")
            return False

def get_user_collections(username: str, all_collections: List[str]) -> List[str]:
    """Filter collections that belong to the current user"""
    if not username:
        return []
    
    # Super admin can see all collections
    if st.session_state.current_user and st.session_state.current_user.get("role") == "super_admin":
        return all_collections
    
    # Regular users can only see their own collections
    prefix = get_user_collection_prefix(username)
    return [col for col in all_collections if col.startswith(prefix)]

def get_user_collection_prefix(username: str) -> str:
    """Get collection prefix for user isolation"""
    return f"user_{username}_"

# Auto-save configuration function
def auto_save_config_if_enabled():
    """Auto-save configuration if enabled"""
    if st.session_state.get("auto_save_config", True) and st.session_state.current_user:
        save_user_session_config()

def save_user_session_config():
    """Save current session state to user configuration"""
    if not st.session_state.current_user:
        return False
    
    username = st.session_state.current_user["username"]
    
    config_updates = {
        "qdrant": {
            "host": st.session_state.get("qdrant_host", "......"),
            "port": st.session_state.get("qdrant_port", 6333),
            "auto_connect": st.session_state.get("qdrant_auto_connect", True)
        },
        "webhook": {
            "url": st.session_state.get("webhook_url", ""),
            "timeout": st.session_state.get("webhook_timeout", 30),
            "auto_test": st.session_state.get("webhook_auto_test", False)
        },
        "openai": {
            "api_key": st.session_state.get("openai_api_key", ""),
            "model": st.session_state.get("openai_model", "text-embedding-3-small"),
            "chat_model": st.session_state.get("openai_chat_model", "gpt-3.5-turbo"),
            "auto_load": st.session_state.get("openai_auto_load", True)
        },
        "preferences": {
            "debug_mode": st.session_state.get("debug_mode", False),
            "auto_save": st.session_state.get("auto_save_config", True)
        }
    }
    
    return UserConfigManager.save_user_config(username, config_updates)
class DataVisualization:
    """Handles all data visualization and analytics"""
    
    def __init__(self):
        self.colors = CHART_COLORS
        self.theme = PLOTLY_THEME
    
    def create_collections_overview(self, username: str, all_collections: List[str], qdrant_client) -> Dict[str, Any]:
        """Create overview data for collections visualization"""
        try:
            user_collections = get_user_collections(username, all_collections)
            collections_data = []
            
            for collection_name in user_collections:
                try:
                    collection_info = qdrant_client.get_collection(collection_name)
                    
                    # Get sample points to analyze content
                    search_filter = None
                    if st.session_state.current_user and st.session_state.current_user.get("role") != "super_admin":
                        search_filter = Filter(
                            must=[
                                FieldCondition(
                                    key="metadata.uploaded_by",
                                    match=MatchValue(value=username)
                                )
                            ]
                        )
                    
                    if search_filter:
                        points, _ = qdrant_client.scroll(
                            collection_name=collection_name,
                            scroll_filter=search_filter,
                            limit=100,
                            with_payload=True
                        )
                    else:
                        points, _ = qdrant_client.scroll(
                            collection_name=collection_name,
                            limit=100,
                            with_payload=True
                        )
                    
                    # Analyze collection data
                    total_documents = len(points)
                    file_types = Counter()
                    text_lengths = []
                    chunk_types = Counter()
                    upload_dates = []
                    uploaders = Counter()
                    
                    for point in points:
                        metadata = point.payload.get('metadata', {})
                        
                        # File types
                        file_type = metadata.get('file_type', 'unknown')
                        file_types[file_type] += 1
                        
                        # Text lengths
                        text = point.payload.get('pageContent') or point.payload.get('text', '')
                        if text:
                            text_lengths.append(len(str(text)))
                        
                        # Chunk types
                        chunk_type = metadata.get('chunk_type', 'unknown')
                        chunk_types[chunk_type] += 1
                        
                        # Upload dates
                        upload_time = metadata.get('upload_timestamp', '')
                        if upload_time:
                            try:
                                upload_date = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
                                upload_dates.append(upload_date.date())
                            except:
                                pass
                        
                        # Uploaders (for super admin)
                        if st.session_state.current_user and st.session_state.current_user.get("role") == "super_admin":
                            uploader = metadata.get('uploaded_by', 'unknown')
                            uploaders[uploader] += 1
                    
                    collections_data.append({
                        'name': collection_name,
                        'total_documents': total_documents,
                        'vector_size': collection_info.config.params.vectors.size,
                        'file_types': dict(file_types),
                        'chunk_types': dict(chunk_types),
                        'text_lengths': text_lengths,
                        'upload_dates': upload_dates,
                        'uploaders': dict(uploaders),
                        'avg_text_length': np.mean(text_lengths) if text_lengths else 0,
                        'total_characters': sum(text_lengths) if text_lengths else 0
                    })
                    
                except Exception as e:
                    st.warning(f"Could not analyze collection {collection_name}: {str(e)}")
                    continue
            
            return {
                'collections': collections_data,
                'total_collections': len(user_collections),
                'total_documents': sum(c['total_documents'] for c in collections_data),
                'total_characters': sum(c['total_characters'] for c in collections_data)
            }
            
        except Exception as e:
            st.error(f"Error creating collections overview: {str(e)}")
            return {'collections': [], 'total_collections': 0, 'total_documents': 0, 'total_characters': 0}
    
    def create_chat_analytics(self, chat_history: List[Dict]) -> Dict[str, Any]:
        """Analyze chat history for visualization"""
        try:
            if not chat_history:
                return {'total_messages': 0, 'conversations': 0}
            
            # Basic metrics
            total_messages = len(chat_history)
            user_messages = [msg for msg in chat_history if msg.get('type') == 'user']
            assistant_messages = [msg for msg in chat_history if msg.get('type') == 'assistant']
            
            # Conversation analysis
            conversations = []
            current_conversation = {'questions': [], 'responses': [], 'start_time': None, 'end_time': None}
            
            for msg in chat_history:
                if msg.get('type') == 'user':
                    if current_conversation['questions']:
                        # Save previous conversation
                        conversations.append(current_conversation.copy())
                        current_conversation = {'questions': [], 'responses': [], 'start_time': None, 'end_time': None}
                    
                    current_conversation['questions'].append(msg)
                    if not current_conversation['start_time']:
                        current_conversation['start_time'] = msg.get('timestamp', '')
                
                elif msg.get('type') == 'assistant':
                    current_conversation['responses'].append(msg)
                    current_conversation['end_time'] = msg.get('timestamp', '')
            
            # Add last conversation
            if current_conversation['questions']:
                conversations.append(current_conversation)
            
            # Time analysis
            hourly_activity = Counter()
            daily_activity = Counter()
            
            for msg in chat_history:
                timestamp_str = msg.get('timestamp', '')
                if timestamp_str:
                    try:
                        # Parse time (assuming format HH:MM:SS)
                        hour = int(timestamp_str.split(':')[0])
                        hourly_activity[hour] += 1
                        
                        # For daily activity, we'll use today as default since we only have time
                        daily_activity[datetime.now().strftime('%Y-%m-%d')] += 1
                    except:
                        pass
            
            # Message length analysis
            question_lengths = []
            response_lengths = []
            
            for msg in user_messages:
                question_lengths.append(len(msg.get('message', '')))
            
            for msg in assistant_messages:
                response_text = msg.get('message', '')
                if isinstance(response_text, str):
                    try:
                        data = json.loads(response_text)
                        response_text = data.get('response', str(data))
                    except:
                        pass
                response_lengths.append(len(str(response_text)))
            
            return {
                'total_messages': total_messages,
                'user_messages': len(user_messages),
                'assistant_messages': len(assistant_messages),
                'conversations': len(conversations),
                'conversation_details': conversations,
                'hourly_activity': dict(hourly_activity),
                'daily_activity': dict(daily_activity),
                'question_lengths': question_lengths,
                'response_lengths': response_lengths,
                'avg_question_length': np.mean(question_lengths) if question_lengths else 0,
                'avg_response_length': np.mean(response_lengths) if response_lengths else 0
            }
            
        except Exception as e:
            st.error(f"Error analyzing chat data: {str(e)}")
            return {'total_messages': 0, 'conversations': 0}
    
    def create_questions_analytics(self, username: str) -> Dict[str, Any]:
        """Analyze saved questions for visualization"""
        try:
            questions = QuestionManager.load_user_questions(username)
            
            if not questions:
                return {'total_questions': 0, 'categories': {}, 'usage_stats': {}}
            
            # Basic metrics
            total_questions = len(questions)
            favorites = len([q for q in questions if q.get('favorite', False)])
            
            # Category analysis
            categories = Counter()
            for q in questions:
                category = q.get('category', 'General')
                categories[category] += 1
            
            # Usage analysis
            usage_stats = {
                'never_used': len([q for q in questions if q.get('used_count', 0) == 0]),
                'low_usage': len([q for q in questions if 1 <= q.get('used_count', 0) <= 3]),
                'medium_usage': len([q for q in questions if 4 <= q.get('used_count', 0) <= 10]),
                'high_usage': len([q for q in questions if q.get('used_count', 0) > 10])
            }
            
            # Tags analysis
            all_tags = []
            for q in questions:
                tags = q.get('tags', [])
                all_tags.extend(tags)
            
            tag_frequency = Counter(all_tags)
            
            # Time analysis
            creation_dates = []
            for q in questions:
                created_at = q.get('created_at', '')
                if created_at:
                    try:
                        date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        creation_dates.append(date.date())
                    except:
                        pass
            
            daily_creation = Counter(creation_dates)
            
            # Question length analysis
            question_lengths = [len(q.get('question', '')) for q in questions]
            
            return {
                'total_questions': total_questions,
                'favorites': favorites,
                'categories': dict(categories),
                'usage_stats': usage_stats,
                'tag_frequency': dict(tag_frequency),
                'daily_creation': dict(daily_creation),
                'question_lengths': question_lengths,
                'avg_question_length': np.mean(question_lengths) if question_lengths else 0,
                'most_used_questions': sorted(questions, key=lambda x: x.get('used_count', 0), reverse=True)[:5]
            }
            
        except Exception as e:
            st.error(f"Error analyzing questions data: {str(e)}")
            return {'total_questions': 0, 'categories': {}, 'usage_stats': {}}
    
    def plot_collections_overview(self, collections_data: Dict[str, Any]) -> List[go.Figure]:
        """Create collection overview charts"""
        charts = []
        
        if not collections_data['collections']:
            return charts
        
        collections = collections_data['collections']
        
        # 1. Documents per Collection (Bar Chart)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=[c['name'] for c in collections],
            y=[c['total_documents'] for c in collections],
            marker_color=self.colors['primary'],
            name='Documents'
        ))
        fig1.update_layout(
            title='📊 Documents per Collection',
            xaxis_title='Collections',
            yaxis_title='Number of Documents',
            template=self.theme,
            height=CHART_HEIGHT
        )
        charts.append(fig1)
        
        # 2. File Types Distribution (Pie Chart)
        all_file_types = Counter()
        for c in collections:
            for file_type, count in c['file_types'].items():
                all_file_types[file_type] += count
        
        if all_file_types:
            fig2 = go.Figure()
            fig2.add_trace(go.Pie(
                labels=list(all_file_types.keys()),
                values=list(all_file_types.values()),
                hole=0.3,
                marker_colors=px.colors.qualitative.Set3
            ))
            fig2.update_layout(
                title='📄 File Types Distribution',
                template=self.theme,
                height=CHART_HEIGHT
            )
            charts.append(fig2)
        
        # 3. Text Length Distribution (Histogram)
        all_text_lengths = []
        for c in collections:
            all_text_lengths.extend(c['text_lengths'])
        
        if all_text_lengths:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=all_text_lengths,
                nbinsx=30,
                marker_color=self.colors['success'],
                name='Text Length'
            ))
            fig3.update_layout(
                title='📏 Document Text Length Distribution',
                xaxis_title='Characters per Document',
                yaxis_title='Frequency',
                template=self.theme,
                height=CHART_HEIGHT
            )
            charts.append(fig3)
        
        # 4. Collection Metrics Comparison (Multi-bar Chart)
        fig4 = go.Figure()
        
        collection_names = [c['name'] for c in collections]
        documents = [c['total_documents'] for c in collections]
        avg_lengths = [c['avg_text_length'] for c in collections]
        
        fig4.add_trace(go.Bar(
            name='Documents',
            x=collection_names,
            y=documents,
            yaxis='y',
            marker_color=self.colors['primary']
        ))
        
        fig4.add_trace(go.Scatter(
            name='Avg Text Length',
            x=collection_names,
            y=avg_lengths,
            yaxis='y2',
            mode='lines+markers',
            marker_color=self.colors['danger']
        ))
        
        fig4.update_layout(
            title='📈 Collection Metrics Comparison',
            xaxis_title='Collections',
            yaxis=dict(title='Number of Documents', side='left'),
            yaxis2=dict(title='Average Text Length', side='right', overlaying='y'),
            template=self.theme,
            height=CHART_HEIGHT
        )
        charts.append(fig4)
        
        return charts
    
    def plot_chat_analytics(self, chat_data: Dict[str, Any]) -> List[go.Figure]:
        """Create chat analytics charts"""
        charts = []
        
        if chat_data['total_messages'] == 0:
            return charts
        
        # 1. Message Distribution (Pie Chart)
        fig1 = go.Figure()
        fig1.add_trace(go.Pie(
            labels=['User Messages', 'Assistant Responses'],
            values=[chat_data['user_messages'], chat_data['assistant_messages']],
            hole=0.3,
            marker_colors=[self.colors['primary'], self.colors['success']]
        ))
        fig1.update_layout(
            title='💬 Message Distribution',
            template=self.theme,
            height=CHART_HEIGHT
        )
        charts.append(fig1)
        
        # 2. Hourly Activity (Bar Chart)
        if chat_data['hourly_activity']:
            hours = list(range(24))
            activity = [chat_data['hourly_activity'].get(h, 0) for h in hours]
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=hours,
                y=activity,
                marker_color=self.colors['info'],
                name='Messages'
            ))
            fig2.update_layout(
                title='🕐 Chat Activity by Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Number of Messages',
                template=self.theme,
                height=CHART_HEIGHT
            )
            charts.append(fig2)
        
        # 3. Message Length Comparison (Box Plot)
        if chat_data['question_lengths'] and chat_data['response_lengths']:
            fig3 = go.Figure()
            
            fig3.add_trace(go.Box(
                y=chat_data['question_lengths'],
                name='Questions',
                marker_color=self.colors['primary']
            ))
            
            fig3.add_trace(go.Box(
                y=chat_data['response_lengths'],
                name='Responses',
                marker_color=self.colors['success']
            ))
            
            fig3.update_layout(
                title='📏 Message Length Distribution',
                yaxis_title='Character Count',
                template=self.theme,
                height=CHART_HEIGHT
            )
            charts.append(fig3)
        
        # 4. Conversation Flow (Sankey Diagram)
        if len(chat_data.get('conversation_details', [])) > 1:
            conversations = chat_data['conversation_details']
            conversation_lengths = [len(c['questions']) for c in conversations]
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=list(range(1, len(conversation_lengths) + 1)),
                y=conversation_lengths,
                mode='lines+markers',
                marker_color=self.colors['warning'],
                name='Questions per Conversation'
            ))
            
            fig4.update_layout(
                title='🔄 Conversation Flow',
                xaxis_title='Conversation Number',
                yaxis_title='Questions per Conversation',
                template=self.theme,
                height=CHART_HEIGHT
            )
            charts.append(fig4)
        
        return charts
    
    def plot_questions_analytics(self, questions_data: Dict[str, Any]) -> List[go.Figure]:
        """Create questions analytics charts"""
        charts = []
        
        if questions_data['total_questions'] == 0:
            return charts
        
        # 1. Questions by Category (Bar Chart)
        if questions_data['categories']:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=list(questions_data['categories'].keys()),
                y=list(questions_data['categories'].values()),
                marker_color=self.colors['secondary'],
                name='Questions'
            ))
            fig1.update_layout(
                title='📂 Questions by Category',
                xaxis_title='Categories',
                yaxis_title='Number of Questions',
                template=self.theme,
                height=CHART_HEIGHT
            )
            charts.append(fig1)
        
        # 2. Usage Statistics (Donut Chart)
        usage_stats = questions_data['usage_stats']
        fig2 = go.Figure()
        fig2.add_trace(go.Pie(
            labels=['Never Used', 'Low Usage (1-3)', 'Medium Usage (4-10)', 'High Usage (10+)'],
            values=[usage_stats['never_used'], usage_stats['low_usage'], 
                   usage_stats['medium_usage'], usage_stats['high_usage']],
            hole=0.4,
            marker_colors=px.colors.qualitative.Pastel
        ))
        fig2.update_layout(
            title='📊 Question Usage Distribution',
            template=self.theme,
            height=CHART_HEIGHT
        )
        charts.append(fig2)
        
        # 3. Tag Cloud (Bar Chart)
        if questions_data['tag_frequency']:
            top_tags = dict(sorted(questions_data['tag_frequency'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10])
            
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=list(top_tags.values()),
                y=list(top_tags.keys()),
                orientation='h',
                marker_color=self.colors['info'],
                name='Frequency'
            ))
            fig3.update_layout(
                title='🏷️ Top 10 Question Tags',
                xaxis_title='Frequency',
                yaxis_title='Tags',
                template=self.theme,
                height=CHART_HEIGHT
            )
            charts.append(fig3)
        
        # 4. Question Length Distribution (Histogram)
        if questions_data['question_lengths']:
            fig4 = go.Figure()
            fig4.add_trace(go.Histogram(
                x=questions_data['question_lengths'],
                nbinsx=20,
                marker_color=self.colors['warning'],
                name='Question Length'
            ))
            fig4.update_layout(
                title='📏 Question Length Distribution',
                xaxis_title='Characters per Question',
                yaxis_title='Frequency',
                template=self.theme,
                height=CHART_HEIGHT
            )
            charts.append(fig4)
        
        return charts
    
    def create_custom_chart(self, data: pd.DataFrame, chart_type: str, x_col: str, y_col: str, 
                           title: str = "", color_col: str = None) -> go.Figure:
        """Create custom charts based on user input"""
        try:
            fig = go.Figure()
            
            if chart_type == "line":
                if color_col and color_col in data.columns:
                    for group in data[color_col].unique():
                        group_data = data[data[color_col] == group]
                        fig.add_trace(go.Scatter(
                            x=group_data[x_col],
                            y=group_data[y_col],
                            mode='lines+markers',
                            name=str(group)
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=data[x_col],
                        y=data[y_col],
                        mode='lines+markers',
                        marker_color=self.colors['primary']
                    ))
            
            elif chart_type == "bar":
                if color_col and color_col in data.columns:
                    fig = px.bar(data, x=x_col, y=y_col, color=color_col, title=title)
                else:
                    fig.add_trace(go.Bar(
                        x=data[x_col],
                        y=data[y_col],
                        marker_color=self.colors['primary']
                    ))
            
            elif chart_type == "scatter":
                if color_col and color_col in data.columns:
                    fig = px.scatter(data, x=x_col, y=y_col, color=color_col, title=title)
                else:
                    fig.add_trace(go.Scatter(
                        x=data[x_col],
                        y=data[y_col],
                        mode='markers',
                        marker_color=self.colors['primary']
                    ))
            
            elif chart_type == "histogram":
                fig.add_trace(go.Histogram(
                    x=data[x_col],
                    marker_color=self.colors['primary']
                ))
            
            elif chart_type == "box":
                if color_col and color_col in data.columns:
                    for group in data[color_col].unique():
                        group_data = data[data[color_col] == group]
                        fig.add_trace(go.Box(
                            y=group_data[y_col],
                            name=str(group)
                        ))
                else:
                    fig.add_trace(go.Box(
                        y=data[y_col],
                        marker_color=self.colors['primary']
                    ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
                template=self.theme,
                height=CHART_HEIGHT
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating custom chart: {str(e)}")
            return go.Figure()
    
    def export_chart(self, fig: go.Figure, filename: str, format: str = "png") -> bytes:
        """Export chart to various formats"""
        try:
            if format == "png":
                img_bytes = fig.to_image(format="png", width=CHART_WIDTH, height=CHART_HEIGHT)
            elif format == "jpg":
                img_bytes = fig.to_image(format="jpg", width=CHART_WIDTH, height=CHART_HEIGHT)
            elif format == "pdf":
                img_bytes = fig.to_image(format="pdf", width=CHART_WIDTH, height=CHART_HEIGHT)
            elif format == "svg":
                img_bytes = fig.to_image(format="svg", width=CHART_WIDTH, height=CHART_HEIGHT)
            elif format == "html":
                html_str = fig.to_html(include_plotlyjs='cdn')
                img_bytes = html_str.encode('utf-8')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return img_bytes
            
        except Exception as e:
            st.error(f"Error exporting chart: {str(e)}")
            return b""

def render_data_visualization_tab():
    """Render the main data visualization tab"""
    st.header("📊 Data Visualization & Analytics")
    
    if not st.session_state.current_user:
        st.error("❌ Please log in to access data visualization")
        return
    
    # Initialize visualization system
    viz = DataVisualization()
    username = st.session_state.current_user["username"]
    
    # Sub-tabs for different visualization categories
    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
        "📊 Overview", 
        "📚 Collections", 
        "💬 Chat Analytics", 
        "💾 Questions", 
        "🎨 Custom Charts"
    ])
    
    with viz_tab1:
        render_overview_dashboard(viz, username)
    
    with viz_tab2:
        render_collections_analytics(viz, username)
    
    with viz_tab3:
        render_chat_analytics(viz, username)
    
    with viz_tab4:
        render_questions_analytics(viz, username)
    
    with viz_tab5:
        render_custom_charts(viz, username)


class EmbeddingManager:
    """Handles embedding operations"""
    
    @staticmethod
    @st.cache_resource
    def load_sentence_transformer(model_name: str) -> SentenceTransformer:
        """Load sentence transformer model for embeddings"""
        return SentenceTransformer(model_name)
    
    @staticmethod
    def get_openai_embedding(text: str, api_key: str, model: str = "text-embedding-ada-002") -> Optional[List[float]]:
        """Get embedding from OpenAI API with proper error handling"""
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except openai.AuthenticationError:
            st.error("❌ Invalid OpenAI API key")
        except openai.RateLimitError:
            st.error("❌ OpenAI rate limit exceeded")
        except openai.APIError as e:
            st.error(f"❌ OpenAI API error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
        return None
    
    @staticmethod
    def encode_text(text: str, model, model_type: str, api_key: Optional[str] = None) -> Optional[List[float]]:
        """Encode text using either SentenceTransformers or OpenAI API"""
        if model_type == "openai":
            if not api_key:
                st.error("❌ OpenAI API key required")
                return None
            return EmbeddingManager.get_openai_embedding(text, api_key, model)
        else:
            return model.encode(text).tolist()
    
    @staticmethod
    def get_vector_size(model, model_type: str, api_key: Optional[str] = None) -> int:
        """Get the vector size from the loaded model"""
        try:
            if model_type == "openai":
                # OpenAI embedding models and their dimensions
                model_sizes = {
                    "text-embedding-ada-002": 1536,
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072
                }
                return model_sizes.get(model, 1536)
            else:
                # Test with a simple text to get the actual vector size
                test_vector = model.encode("test")
                return len(test_vector)
        except Exception as e:
            st.error(f"❌ Failed to get model vector size: {str(e)}")
            return 1536  # Default fallback

def render_header():
    """Render the application header with user info and logout"""
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        st.title("🔍 Qdrant File Reader - Multi-User")
    
    with col2:
        if st.session_state.current_user:
            user = st.session_state.current_user
            role_icon = "👑" if user.get("role") == "super_admin" else "👤"
            st.write(f"**Welcome, {role_icon} {user.get('username')}**")
            st.caption(f"Role: {user.get('role', 'user')}")
    
    with col3:
        if st.button("🚪 Logout", key="logout_btn"):
            # Clear authentication state
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.chat_history = []
            st.session_state.user_collections = []
            st.success("👋 Logged out successfully!")
            st.rerun()

def render_sidebar():
    """Render sidebar configuration"""
    st.sidebar.header("⚙️ Configuration")
    
    # Show current user info
    if st.session_state.current_user:
        with st.sidebar.container():
            user = st.session_state.current_user
            role_icon = "👑" if user.get("role") == "super_admin" else "👤"
            st.sidebar.success(f"{role_icon} **{user.get('username')}**")
            st.sidebar.caption(f"Role: {user.get('role', 'user')}")
            
            # Quick save button
            if st.sidebar.button("💾 Save Config", key="quick_save_config"):
                if save_user_session_config():
                    st.sidebar.success("✅ Config saved!")
                else:
                    st.sidebar.error("❌ Save failed!")
    
    # Qdrant connection settings
    with st.sidebar.expander("🔗 Qdrant Connection", expanded=True):
        host = st.text_input("Host", value="......", key="qdrant_host")
        port = st.number_input("Port", value=6333, min_value=1, max_value=65535, key="qdrant_port")
        
        if st.button("Connect to Qdrant", key="connect_qdrant"):
            client = QdrantManager.connect(host, port)
            if client:
                st.session_state.qdrant_client = client
                st.session_state.connection_status = True
                st.success("✅ Connected to Qdrant!")
                
                # Load collections with user filtering
                try:
                    collections = client.get_collections().collections
                    all_collections = [col.name for col in collections]
                    st.session_state.collections = all_collections
                    
                    # Filter collections for current user
                    if st.session_state.current_user:
                        username = st.session_state.current_user["username"]
                        st.session_state.user_collections = get_user_collections(username, all_collections)
                except Exception as e:
                    st.error(f"❌ Failed to get collections: {str(e)}")
    
    # Model loading
    with st.sidebar.expander("🤖 Embedding Model", expanded=True):
        model_type = st.selectbox("Model Type", ["openai", "sentence-transformers"], 
                                 index=0, key="model_type_select")
        
        if model_type == "openai":
            # OpenAI API configuration
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                value=st.session_state.openai_api_key,
                help="Your OpenAI API key (pre-filled with default)",
                key="openai_key_input"
            )
            if api_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = api_key
            
            # Set default model index to text-embedding-3-small
            default_model_index = 1 if "text-embedding-3-small" in OPENAI_MODELS else 0
            selected_model = st.selectbox("Select OpenAI Model", OPENAI_MODELS, 
                                        index=default_model_index, key="openai_model_select")
            
            # Auto-load the default model on startup
            if not st.session_state.model_status:
                st.info("🔄 Auto-loading default OpenAI model...")
                if st.button("🚀 Load Default Model", key="auto_load_openai"):
                    current_api_key = get_api_key()
                    if current_api_key:
                        try:
                            # Test the API key with a simple embedding
                            test_embedding = EmbeddingManager.get_openai_embedding("test", current_api_key, selected_model)
                            if test_embedding:
                                st.session_state.model = selected_model
                                st.session_state.model_type = "openai"
                                st.session_state.model_status = True
                                st.success(f"✅ Auto-loaded OpenAI model: {selected_model}")
                                st.info(f"Vector size: {len(test_embedding)}D")
                                st.rerun()
                            else:
                                st.error("❌ Failed to load OpenAI model - check your API key")
                        except Exception as e:
                            st.error(f"❌ Error loading OpenAI model: {str(e)}")
            
            if st.button("🔄 Reload OpenAI Model", key="load_openai_model"):
                current_api_key = get_api_key()
                if current_api_key:
                    try:
                        # Test the API key with a simple embedding
                        test_embedding = EmbeddingManager.get_openai_embedding("test", current_api_key, selected_model)
                        if test_embedding:
                            st.session_state.model = selected_model
                            st.session_state.model_type = "openai"
                            st.session_state.model_status = True
                            st.success(f"✅ Loaded OpenAI model: {selected_model}")
                            st.info(f"Vector size: {len(test_embedding)}D")
                        else:
                            st.error("❌ Failed to load OpenAI model - check your API key")
                    except Exception as e:
                        st.error(f"❌ Error loading OpenAI model: {str(e)}")
                else:
                    st.error("❌ Please enter your OpenAI API key")
        
        else:
            # Sentence Transformers
            selected_model = st.selectbox("Select Model", SENTENCE_TRANSFORMER_MODELS, key="st_model_select")
            
            if st.button("Load Model", key="load_st_model"):
                with st.spinner("Loading embedding model..."):
                    try:
                        model = EmbeddingManager.load_sentence_transformer(selected_model)
                        st.session_state.model = model
                        st.session_state.model_type = "sentence-transformers"
                        st.session_state.model_status = True
                        vector_size = EmbeddingManager.get_vector_size(model, "sentence-transformers")
                        st.success(f"✅ Loaded model: {selected_model}")
                        st.info(f"Vector size: {vector_size}D")
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
        
        # Display current model info
        if st.session_state.model_status and st.session_state.model:
            st.success(f"🤖 Current: {st.session_state.model} ({st.session_state.model_type})")
        else:
            st.warning("⚠️ No model loaded")

def render_overview_dashboard(viz: DataVisualization, username: str):
    """Render overview dashboard with key metrics"""
    st.subheader("📈 Dashboard Overview")
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Collections metrics
    if st.session_state.qdrant_client and st.session_state.collections:
        collections_data = viz.create_collections_overview(
            username, st.session_state.collections, st.session_state.qdrant_client
        )
        
        with col1:
            st.metric(
                label="📚 Collections",
                value=collections_data['total_collections'],
                delta=None
            )
        
        with col2:
            st.metric(
                label="📄 Documents", 
                value=collections_data['total_documents'],
                delta=None
            )
    else:
        with col1:
            st.metric("📚 Collections", "0")
        with col2:
            st.metric("📄 Documents", "0")
    
    # Chat metrics
    chat_data = viz.create_chat_analytics(st.session_state.chat_history)
    with col3:
        st.metric(
            label="💬 Chat Messages",
            value=chat_data['total_messages'],
            delta=None
        )
    
    # Questions metrics
    questions_data = viz.create_questions_analytics(username)
    with col4:
        st.metric(
            label="💾 Saved Questions",
            value=questions_data['total_questions'],
            delta=None
        )
    
    st.divider()
    
    # Quick insights
    st.subheader("🔍 Quick Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        if chat_data['total_messages'] > 0:
            avg_response_time = "N/A"  # You can calculate this from webhook metadata
            st.info(f"""
            **💬 Chat Activity:**
            - Total conversations: {chat_data.get('conversations', 0)}
            - Average question length: {chat_data.get('avg_question_length', 0):.0f} characters
            - Average response length: {chat_data.get('avg_response_length', 0):.0f} characters
            """)
        else:
            st.info("💬 **No chat activity yet**\nStart chatting to see analytics!")
    
    with insight_col2:
        if questions_data['total_questions'] > 0:
            most_used_category = max(questions_data['categories'].items(), key=lambda x: x[1])[0] if questions_data['categories'] else "None"
            st.info(f"""
            **💾 Question Insights:**
            - Most popular category: {most_used_category}
            - Favorite questions: {questions_data.get('favorites', 0)}
            - Average question length: {questions_data.get('avg_question_length', 0):.0f} characters
            """)
        else:
            st.info("💾 **No saved questions yet**\nSave your first question to see analytics!")

def render_collections_analytics(viz: DataVisualization, username: str):
    """Render collections analytics and visualizations"""
    st.subheader("📚 Collections Analytics")
    
    if not st.session_state.qdrant_client:
        st.warning("⚠️ Please connect to Qdrant to view collection analytics")
        return
    
    if not st.session_state.collections:
        st.info("📝 No collections found. Upload some documents first!")
        return
    
    # Get collections data
    with st.spinner("📊 Analyzing collections data..."):
        collections_data = viz.create_collections_overview(
            username, st.session_state.collections, st.session_state.qdrant_client
        )
    
    if not collections_data['collections']:
        st.info("📝 No accessible collections found.")
        return
    
    # Generate charts
    charts = viz.plot_collections_overview(collections_data)
    
    # Display charts
    for i, chart in enumerate(charts):
        st.plotly_chart(chart, use_container_width=True, key=f"collection_chart_{i}")
        
        # Export options
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(f"📥 PNG", key=f"export_png_col_{i}"):
                img_bytes = viz.export_chart(chart, f"collection_chart_{i}", "png")
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name=f"collection_chart_{i}.png",
                    mime="image/png",
                    key=f"download_png_col_{i}"
                )
        with col2:
            if st.button(f"📄 PDF", key=f"export_pdf_col_{i}"):
                img_bytes = viz.export_chart(chart, f"collection_chart_{i}", "pdf")
                st.download_button(
                    label="Download PDF",
                    data=img_bytes,
                    file_name=f"collection_chart_{i}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_col_{i}"
                )
    
    # Detailed statistics
    st.subheader("📋 Detailed Statistics")
    
    # Create summary table
    summary_data = []
    for collection in collections_data['collections']:
        summary_data.append({
            'Collection': collection['name'],
            'Documents': collection['total_documents'],
            'Vector Size': f"{collection['vector_size']}D",
            'Avg Text Length': f"{collection['avg_text_length']:.0f}",
            'Total Characters': f"{collection['total_characters']:,}",
            'File Types': len(collection['file_types'])
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Export data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Export Statistics CSV",
            data=csv,
            file_name=f"collections_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def render_chat_analytics(viz: DataVisualization, username: str):
    """Render chat analytics and visualizations"""
    st.subheader("💬 Chat Analytics")
    
    if not st.session_state.chat_history:
        st.info("💬 No chat history available. Start chatting to see analytics!")
        return
    
    # Get chat data
    with st.spinner("📊 Analyzing chat data..."):
        chat_data = viz.create_chat_analytics(st.session_state.chat_history)
    
    # Generate charts
    charts = viz.plot_chat_analytics(chat_data)
    
    # Display summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", chat_data['total_messages'])
    with col2:
        st.metric("Conversations", chat_data['conversations'])
    with col3:
        avg_msg_length = (chat_data['avg_question_length'] + chat_data['avg_response_length']) / 2
        st.metric("Avg Message Length", f"{avg_msg_length:.0f} chars")
    
    # Display charts
    for i, chart in enumerate(charts):
        st.plotly_chart(chart, use_container_width=True, key=f"chat_chart_{i}")
        
        # Export options
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(f"📥 PNG", key=f"export_png_chat_{i}"):
                img_bytes = viz.export_chart(chart, f"chat_chart_{i}", "png")
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name=f"chat_chart_{i}.png",
                    mime="image/png",
                    key=f"download_png_chat_{i}"
                )
        with col2:
            if st.button(f"📄 PDF", key=f"export_pdf_chat_{i}"):
                img_bytes = viz.export_chart(chart, f"chat_chart_{i}", "pdf")
                st.download_button(
                    label="Download PDF",
                    data=img_bytes,
                    file_name=f"chat_chart_{i}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_chat_{i}"
                )
    
    # Chat insights
    st.subheader("🔍 Chat Insights")
    
    if chat_data['hourly_activity']:
        busiest_hour = max(chat_data['hourly_activity'].items(), key=lambda x: x[1])
        st.info(f"🕐 **Busiest Hour:** {busiest_hour[0]}:00 with {busiest_hour[1]} messages")
    
    if chat_data['conversation_details']:
        longest_conversation = max(chat_data['conversation_details'], key=lambda x: len(x['questions']))
        st.info(f"💬 **Longest Conversation:** {len(longest_conversation['questions'])} questions")

def render_questions_analytics(viz: DataVisualization, username: str):
    """Render questions analytics and visualizations"""
    st.subheader("💾 Questions Analytics")
    
    # Get questions data
    with st.spinner("📊 Analyzing questions data..."):
        questions_data = viz.create_questions_analytics(username)
    
    if questions_data['total_questions'] == 0:
        st.info("💾 No saved questions available. Save some questions to see analytics!")
        return
    
    # Generate charts
    charts = viz.plot_questions_analytics(questions_data)
    
    # Display summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", questions_data['total_questions'])
    with col2:
        st.metric("Categories", len(questions_data['categories']))
    with col3:
        st.metric("Favorites", questions_data['favorites'])
    with col4:
        never_used = questions_data['usage_stats']['never_used']
        st.metric("Never Used", never_used)
    
    # Display charts
    for i, chart in enumerate(charts):
        st.plotly_chart(chart, use_container_width=True, key=f"question_chart_{i}")
        
        # Export options
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(f"📥 PNG", key=f"export_png_q_{i}"):
                img_bytes = viz.export_chart(chart, f"question_chart_{i}", "png")
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name=f"question_chart_{i}.png",
                    mime="image/png",
                    key=f"download_png_q_{i}"
                )
        with col2:
            if st.button(f"📄 PDF", key=f"export_pdf_q_{i}"):
                img_bytes = viz.export_chart(chart, f"question_chart_{i}", "pdf")
                st.download_button(
                    label="Download PDF",
                    data=img_bytes,
                    file_name=f"question_chart_{i}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_q_{i}"
                )
    
    # Most used questions
    st.subheader("🔥 Most Used Questions")
    most_used = questions_data.get('most_used_questions', [])
    if most_used:
        for i, question in enumerate(most_used[:5], 1):
            usage_count = question.get('used_count', 0)
            question_text = question.get('question', '')
            if len(question_text) > 100:
                question_text = question_text[:100] + "..."
            
            st.write(f"**{i}.** {question_text}")
            st.caption(f"Used {usage_count} times | Category: {question.get('category', 'General')}")

def render_custom_charts(viz: DataVisualization, username: str):
    """Render custom chart creation interface"""
    st.subheader("🎨 Custom Chart Creator")
    
    # Data source selection
    data_source = st.selectbox(
        "Select Data Source",
        ["Upload CSV", "Collections Data", "Chat Data", "Questions Data", "Sample Data"],
        key="custom_chart_data_source"
    )
    
    df = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file to create custom visualizations",
            key="custom_chart_csv"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
                
                # Show data preview
                with st.expander("👀 Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    elif data_source == "Collections Data":
        if st.session_state.qdrant_client and st.session_state.collections:
            with st.spinner("Loading collections data..."):
                collections_data = viz.create_collections_overview(
                    username, st.session_state.collections, st.session_state.qdrant_client
                )
                
                if collections_data['collections']:
                    # Convert to DataFrame
                    data_list = []
                    for collection in collections_data['collections']:
                        data_list.append({
                            'Collection': collection['name'],
                            'Documents': collection['total_documents'],
                            'Vector_Size': collection['vector_size'],
                            'Avg_Text_Length': collection['avg_text_length'],
                            'Total_Characters': collection['total_characters']
                        })
                    
                    df = pd.DataFrame(data_list)
                    st.success(f"✅ Loaded collections data: {len(df)} collections")
        else:
            st.warning("⚠️ No collections data available")
    
    elif data_source == "Chat Data":
        if st.session_state.chat_history:
            # Convert chat history to DataFrame
            chat_list = []
            for msg in st.session_state.chat_history:
                chat_list.append({
                    'Type': msg.get('type', ''),
                    'Message_Length': len(msg.get('message', '')),
                    'Timestamp': msg.get('timestamp', ''),
                    'Hour': int(msg.get('timestamp', '0:0:0').split(':')[0]) if ':' in msg.get('timestamp', '') else 0
                })
            
            df = pd.DataFrame(chat_list)
            st.success(f"✅ Loaded chat data: {len(df)} messages")
        else:
            st.warning("⚠️ No chat data available")
    
    elif data_source == "Questions Data":
        questions = QuestionManager.load_user_questions(username)
        if questions:
            # Convert questions to DataFrame
            questions_list = []
            for q in questions:
                questions_list.append({
                    'Category': q.get('category', 'General'),
                    'Question_Length': len(q.get('question', '')),
                    'Used_Count': q.get('used_count', 0),
                    'Favorite': q.get('favorite', False),
                    'Tag_Count': len(q.get('tags', []))
                })
            
            df = pd.DataFrame(questions_list)
            st.success(f"✅ Loaded questions data: {len(df)} questions")
        else:
            st.warning("⚠️ No questions data available")
    
    elif data_source == "Sample Data":
        # Generate sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'Value1': np.random.normal(50, 15, 100),
            'Value2': np.random.normal(30, 10, 100),
            'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Count': np.random.poisson(5, 100)
        })
        st.success("✅ Generated sample data for demonstration")
    
    if df is not None and not df.empty:
        st.divider()
        
        # Chart configuration
        st.subheader("⚙️ Chart Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["line", "bar", "scatter", "histogram", "box"],
                key="custom_chart_type"
            )
            
            x_column = st.selectbox(
                "X-axis Column",
                df.columns.tolist(),
                key="custom_chart_x"
            )
        
        with col2:
            if chart_type != "histogram":
                y_column = st.selectbox(
                    "Y-axis Column",
                    [col for col in df.columns if col != x_column],
                    key="custom_chart_y"
                )
            else:
                y_column = None
            
            color_column = st.selectbox(
                "Color By (Optional)",
                ["None"] + [col for col in df.columns if col not in [x_column, y_column]],
                key="custom_chart_color"
            )
        
        chart_title = st.text_input(
            "Chart Title",
            placeholder="Enter chart title...",
            key="custom_chart_title"
        )
        
        # Create and display chart
        if st.button("🎨 Create Chart", key="create_custom_chart"):
            try:
                color_col = color_column if color_column != "None" else None
                
                with st.spinner("Creating chart..."):
                    custom_chart = viz.create_custom_chart(
                        df, chart_type, x_column, y_column or x_column, 
                        chart_title, color_col
                    )
                
                if custom_chart:
                    st.plotly_chart(custom_chart, use_container_width=True)
                    
                    # Export options
                    st.subheader("📥 Export Options")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("📥 PNG", key="export_custom_png"):
                            img_bytes = viz.export_chart(custom_chart, "custom_chart", "png")
                            st.download_button(
                                label="Download PNG",
                                data=img_bytes,
                                file_name="custom_chart.png",
                                mime="image/png",
                                key="download_custom_png"
                            )
                    
                    with col2:
                        if st.button("📄 PDF", key="export_custom_pdf"):
                            img_bytes = viz.export_chart(custom_chart, "custom_chart", "pdf")
                            st.download_button(
                                label="Download PDF",
                                data=img_bytes,
                                file_name="custom_chart.pdf",
                                mime="application/pdf",
                                key="download_custom_pdf"
                            )
                    
                    with col3:
                        if st.button("🌐 HTML", key="export_custom_html"):
                            html_bytes = viz.export_chart(custom_chart, "custom_chart", "html")
                            st.download_button(
                                label="Download HTML",
                                data=html_bytes,
                                file_name="custom_chart.html",
                                mime="text/html",
                                key="download_custom_html"
                            )
                    
                    with col4:
                        if st.button("🖼️ SVG", key="export_custom_svg"):
                            svg_bytes = viz.export_chart(custom_chart, "custom_chart", "svg")
                            st.download_button(
                                label="Download SVG",
                                data=svg_bytes,
                                file_name="custom_chart.svg",
                                mime="image/svg+xml",
                                key="download_custom_svg"
                            )
                
            except Exception as e:
                st.error(f"❌ Error creating chart: {str(e)}")
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"- Rows: {len(df)}")
            st.write(f"- Columns: {len(df.columns)}")
            st.write(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        with col2:
            st.write("**Column Types:**")
            for col, dtype in df.dtypes.items():
                st.write(f"- {col}: {dtype}")
        
        # Data preview
        with st.expander("📋 Full Data Preview"):
            st.dataframe(df, use_container_width=True)
            
            # Export data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export Data CSV",
                data=csv,
                file_name="chart_data.csv",
                mime="text/csv"
            )
class FileProcessor:
    @staticmethod
    def validate_file(uploaded_file) -> bool:
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
            return False
        return True
    
    @staticmethod
    def extract_text_pypdf2(pdf_file) -> Tuple[str, List[Dict]]:
        """Extract text from PDF using PyPDF2"""
        try:
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            pages_info = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    pages_info.append({
                        "page_number": page_num + 1,
                        "text_length": len(page_text),
                        "text": page_text
                    })
                except Exception as e:
                    st.warning(f"⚠️ Error reading page {page_num + 1}: {str(e)}")
                    continue
            
            return text, pages_info
        except Exception as e:
            st.error(f"❌ Error extracting text with PyPDF2: {str(e)}")
            return "", []
    
    @staticmethod
    def extract_text_pymupdf(pdf_file) -> Tuple[str, List[Dict]]:
        """Extract text from PDF using PyMuPDF (more robust)"""
        try:
            pdf_file.seek(0)  # Reset file pointer
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            pages_info = []
            
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    text += page_text + "\n"
                    
                    # Get page dimensions
                    rect = page.rect
                    pages_info.append({
                        "page_number": page_num + 1,
                        "text_length": len(page_text),
                        "text": page_text,
                        "width": rect.width,
                        "height": rect.height
                    })
                except Exception as e:
                    st.warning(f"⚠️ Error reading page {page_num + 1}: {str(e)}")
                    continue
            
            pdf_document.close()
            return text, pages_info
        except Exception as e:
            st.error(f"❌ Error extracting text with PyMuPDF: {str(e)}")
            return "", []
        
    @staticmethod
    def process_pdf_file(pdf_file, extraction_method: str = "pymupdf", 
                        chunk_by: str = "pages", chunk_size: int = 1000, 
                        overlap_size: int = 100) -> List[Dict]:
        """Process PDF file and return chunks with metadata"""
        
        # Extract text based on method
        if extraction_method == "pypdf2":
            full_text, pages_info = FileProcessor.extract_text_pypdf2(pdf_file)
        else:  # pymupdf
            full_text, pages_info = FileProcessor.extract_text_pymupdf(pdf_file)
        
        if not full_text.strip():
            st.warning("⚠️ No text could be extracted from the PDF")
            return []
        
        chunks = []
        
        if chunk_by == "pages":
            # Chunk by pages (no overlap for page-based chunking)
            for page_info in pages_info:
                if page_info["text"].strip():  # Only include pages with text
                    chunks.append({
                        "text": page_info["text"],
                        "page_number": page_info["page_number"],
                        "chunk_type": "page",
                        "text_length": page_info["text_length"]
                    })
        
        elif chunk_by == "paragraphs":
            # Chunk by paragraphs across all pages (no overlap for paragraph-based chunking)
            paragraphs = re.split(r'\n\s*\n', full_text)
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    chunks.append({
                        "text": paragraph.strip(),
                        "chunk_id": i + 1,
                        "chunk_type": "paragraph",
                        "text_length": len(paragraph)
                    })
        
        else:  # fixed_size
            # Fixed-size chunking with custom overlap
            words = full_text.split()
            
            # Ensure overlap doesn't exceed chunk size
            actual_overlap = min(overlap_size, chunk_size // 2)
            
            for i in range(0, len(words), chunk_size - actual_overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "chunk_id": i // (chunk_size - actual_overlap) + 1,
                        "chunk_type": "fixed_size",
                        "word_start": i,
                        "word_end": min(i + chunk_size, len(words)),
                        "text_length": len(chunk_text),
                        "overlap_size": actual_overlap if i > 0 else 0  # First chunk has no overlap
                    })
        
        return chunks
    
    @staticmethod
    def process_text_file(file_content: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                         overlap_size: int = 50) -> List[Dict]:
        """Split text file into chunks with custom overlap"""
        words = file_content.split()
        chunks = []
        
        # Ensure overlap doesn't exceed chunk size
        actual_overlap = min(overlap_size, chunk_size // 2)
        
        for i in range(0, len(words), chunk_size - actual_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "chunk_id": i // (chunk_size - actual_overlap) + 1,
                    "chunk_type": "text",
                    "text_length": len(chunk),
                    "word_start": i,
                    "word_end": min(i + chunk_size, len(words)),
                    "overlap_size": actual_overlap if i > 0 else 0  # First chunk has no overlap
                })
        return chunks
    
    @staticmethod
    def process_csv_file(file_content: str) -> List[Dict]:
        """Convert CSV rows to text chunks"""
        try:
            df = pd.read_csv(io.StringIO(file_content))
            chunks = []
            for idx, row in df.iterrows():
                chunk = ' | '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                if chunk.strip():
                    chunks.append({
                        "text": chunk,
                        "chunk_id": idx + 1,
                        "chunk_type": "csv_row",
                        "text_length": len(chunk)
                    })
            return chunks
        except Exception as e:
            st.error(f"❌ Error processing CSV: {str(e)}")
            return []
    
    @staticmethod
    def process_json_file(file_content: str) -> List[Dict]:
        """Convert JSON data to text chunks"""
        try:
            data = json.loads(file_content)
            chunks = []
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        chunk = ' | '.join([f"{k}: {v}" for k, v in item.items() if v is not None])
                    else:
                        chunk = str(item)
                    
                    if chunk.strip():
                        chunks.append({
                            "text": chunk,
                            "chunk_id": i + 1,
                            "chunk_type": "json_item",
                            "text_length": len(chunk)
                        })
            elif isinstance(data, dict):
                chunk = ' | '.join([f"{k}: {v}" for k, v in data.items() if v is not None])
                if chunk.strip():
                    chunks.append({
                        "text": chunk,
                        "chunk_id": 1,
                        "chunk_type": "json_object",
                        "text_length": len(chunk)
                    })
            else:
                chunk_text = str(data)
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "chunk_id": 1,
                        "chunk_type": "json_value",
                        "text_length": len(chunk_text)
                    })
            
            return chunks
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {str(e)}")
            return []

class DataUploader:
    """Handles data upload to Qdrant with user isolation"""
    
    @staticmethod
    def upload_to_qdrant(client: QdrantClient, collection_name: str, chunks: List[Dict], 
                        model, model_type: str, metadata: Optional[Dict[str, Any]] = None, 
                        api_key: Optional[str] = None) -> bool:
        """Upload text chunks to Qdrant with embeddings and user metadata"""
        if not chunks:
            st.error("❌ No chunks to upload")
            return False
        
        # Add user information to metadata
        if st.session_state.current_user:
            user_metadata = {
                "uploaded_by": st.session_state.current_user["username"],
                "user_role": st.session_state.current_user["role"]
            }
            if metadata:
                metadata.update(user_metadata)
            else:
                metadata = user_metadata
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Get model vector size
            model_vector_size = EmbeddingManager.get_vector_size(model, model_type, api_key)
            status_text.text(f'Model vector size: {model_vector_size}')
            
            # Check if collection exists and is compatible
            status_text.text('Checking collection compatibility...')
            try:
                collection_info = client.get_collection(collection_name)
                actual_size = collection_info.config.params.vectors.size
                if actual_size != model_vector_size:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Vector dimension mismatch!")
                    st.error(f"Collection '{collection_name}' expects {actual_size}D vectors")
                    st.error(f"Your model produces {model_vector_size}D vectors")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Delete Collection & Recreate"):
                            try:
                                client.delete_collection(collection_name)
                                if QdrantManager.create_collection(client, collection_name, model_vector_size):
                                    st.success("✅ Collection recreated with correct dimensions!")
                                    # Update user collections
                                    if st.session_state.current_user:
                                        username = st.session_state.current_user["username"]
                                        all_collections = [col.name for col in client.get_collections().collections]
                                        st.session_state.user_collections = get_user_collections(username, all_collections)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to recreate collection: {str(e)}")
                    
                    with col2:
                        st.info("💡 Or use a different collection name")
                    
                    return False
            except Exception:
                # Collection doesn't exist, create it
                status_text.text(f'Creating collection: {collection_name}...')
                if not QdrantManager.create_collection(client, collection_name, model_vector_size):
                    progress_bar.empty()
                    status_text.empty()
                    return False
                
                # Update user collections list
                if st.session_state.current_user:
                    username = st.session_state.current_user["username"]
                    all_collections = [col.name for col in client.get_collections().collections]
                    st.session_state.user_collections = get_user_collections(username, all_collections)
            
            points = []
            failed_chunks = 0
            
            for i, chunk_data in enumerate(chunks):
                # Update status
                status_text.text(f'Processing chunk {i + 1} of {len(chunks)}...')
                
                # Generate embedding
                embedding = EmbeddingManager.encode_text(chunk_data["text"], model, model_type, api_key)
                
                if embedding is None:
                    failed_chunks += 1
                    st.warning(f"⚠️ Failed to generate embedding for chunk {i + 1}")
                    continue
                
                # Verify embedding size
                if len(embedding) != model_vector_size:
                    st.error(f"❌ Embedding size mismatch: expected {model_vector_size}, got {len(embedding)}")
                    failed_chunks += 1
                    continue
                
                # Create point with user-specific metadata
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "pageContent": chunk_data["text"],  # n8n expects this at root level
                        "text": chunk_data["text"],         # Streamlit uses this
                        "content": chunk_data["text"],      # Alternative field name
                        "metadata": {
                            "chunk_index": i,
                            "source": metadata.get("filename", "unknown") if metadata else "unknown",
                            **{k: v for k, v in chunk_data.items() if k != "text"},
                            **(metadata or {})
                        }
                    }
                )
                points.append(point)
                
                # Update progress
                progress_bar.progress((i + 1) / len(chunks))
            
            if not points:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ No valid embeddings generated")
                return False
            
            # Upload to Qdrant in batches
            batch_size = 100
            status_text.text('Uploading to Qdrant...')
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                client.upsert(collection_name=collection_name, points=batch)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if failed_chunks > 0:
                st.warning(f"⚠️ {failed_chunks} chunks failed to process")
            
            st.success(f"✅ Successfully uploaded {len(points)} chunks to Qdrant!")
            return True
            
        except Exception as e:
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Failed to upload to Qdrant: {str(e)}")
            return False


def render_upload_tab():
    """Render file upload tab with user isolation"""
    st.header("📁 Upload and Process Files")
    
    # Collection management
    st.subheader("Collection Management")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.session_state.user_collections:
            collection_options = st.session_state.user_collections + ["Create New..."]
            selected_option = st.selectbox("Select or Create Collection", collection_options)
            
            if selected_option == "Create New...":
                base_name = st.text_input("New Collection Name", value="documents")
                # Create user-specific collection name
                if st.session_state.current_user:
                    username = st.session_state.current_user["username"]
                    collection_name = create_user_collection_name(username, base_name)
                    if base_name and collection_name != base_name:
                        st.info(f"Collection will be created as: `{collection_name}`")
                else:
                    collection_name = base_name
            else:
                collection_name = selected_option
        else:
            base_name = st.text_input("Collection Name", value="documents")
            if st.session_state.current_user:
                username = st.session_state.current_user["username"]
                collection_name = create_user_collection_name(username, base_name)
                if base_name and collection_name != base_name:
                    st.info(f"Collection will be created as: `{collection_name}`")
            else:
                collection_name = base_name
    
    with col2:
        if st.button("Create Collection", key="create_collection_btn"):
            if collection_name and collection_name not in st.session_state.user_collections:
                if st.session_state.model:
                    vector_size = EmbeddingManager.get_vector_size(
                        st.session_state.model, 
                        st.session_state.model_type,
                        get_api_key()
                    )
                    st.info(f"Creating collection with vector size: {vector_size}")
                else:
                    vector_size = 384  # Default size
                    st.warning("No model loaded, using default vector size: 384")
                
                if QdrantManager.create_collection(st.session_state.qdrant_client, collection_name, vector_size):
                    # Update user collections
                    if st.session_state.current_user:
                        username = st.session_state.current_user["username"]
                        all_collections = [col.name for col in st.session_state.qdrant_client.get_collections().collections]
                        st.session_state.user_collections = get_user_collections(username, all_collections)
                        st.session_state.collections = all_collections
                    st.rerun()
            elif collection_name in st.session_state.user_collections:
                st.warning("⚠️ Collection already exists")
            else:
                st.error("❌ Please enter a collection name")
    
    with col3:
        if st.button("Refresh Collections", key="refresh_collections_btn"):
            try:
                if st.session_state.qdrant_client and st.session_state.current_user:
                    collections = st.session_state.qdrant_client.get_collections().collections
                    all_collections = [col.name for col in collections]
                    st.session_state.collections = all_collections
                    username = st.session_state.current_user["username"]
                    st.session_state.user_collections = get_user_collections(username, all_collections)
                    st.success("✅ Collections refreshed!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to refresh: {str(e)}")
    
    # Show selected collection info
    if collection_name and st.session_state.qdrant_client:
        try:
            if collection_name in st.session_state.user_collections:
                collection_info = st.session_state.qdrant_client.get_collection(collection_name)
                st.info(f"📊 Collection '{collection_name}' has {collection_info.points_count} documents | Vector size: {collection_info.config.params.vectors.size}D")
            elif collection_name:
                st.warning(f"⚠️ Collection '{collection_name}' doesn't exist yet")
        except Exception as e:
            st.warning(f"Could not get collection info: {str(e)}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'csv', 'json', 'pdf'],
        help="Supported formats: TXT, CSV, JSON, PDF"
    )
    
    if uploaded_file is not None:
        # Validate file
        if not FileProcessor.validate_file(uploaded_file):
            return
        
        # File details
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size:,} bytes ({uploaded_file.size / (1024*1024):.2f} MB)")
        st.write(f"**Type:** {uploaded_file.type}")
        
        # Processing options based on file type
        if uploaded_file.name.endswith('.pdf'):
            st.subheader("PDF Processing Options")
            
            col1, col2 = st.columns(2)
            with col1:
                extraction_method = st.selectbox(
                    "Extraction Method",
                    ["pymupdf", "pypdf2"],
                    help="PyMuPDF is generally more robust",
                    key="pdf_extraction_method"
                )
            
            with col2:
                chunk_by = st.selectbox(
                    "Chunking Strategy",
                    ["pages", "paragraphs", "fixed_size"],
                    help="How to split the PDF content",
                    key="pdf_chunk_by"
                )
            
            if chunk_by == "fixed_size":
                # Chunking parameters for fixed size
                st.subheader("Chunking Parameters")
                chunk_col1, chunk_col2 = st.columns(2)
                
                with chunk_col1:
                    chunk_size = st.number_input(
                        "Chunk Size (words)", 
                        min_value=100, 
                        max_value=MAX_CHUNK_SIZE, 
                        value=1000,
                        step=50,
                        help="Number of words per chunk",
                        key="pdf_chunk_size"
                    )
                
                with chunk_col2:
                    overlap_size = st.number_input(
                        "Overlap Size (words)", 
                        min_value=0, 
                        max_value=min(chunk_size // 2, 500), 
                        value=min(chunk_size // 4, 100),
                        step=10,
                        help="Number of overlapping words between chunks",
                        key="pdf_overlap_size"
                    )
                
                # Show chunk info
                st.info(f"📊 Each chunk: {chunk_size} words | Overlap: {overlap_size} words | Effective new content per chunk: {chunk_size - overlap_size} words")
            else:
                chunk_size = 1000  # Default for other methods
                overlap_size = 0   # No overlap for page/paragraph chunking
        
        elif uploaded_file.name.endswith('.txt'):
            st.subheader("Text Processing Options")
            
            # Chunking parameters for text files
            chunk_col1, chunk_col2 = st.columns(2)
            
            with chunk_col1:
                chunk_size = st.number_input(
                    "Chunk Size (words)", 
                    min_value=50, 
                    max_value=2000, 
                    value=DEFAULT_CHUNK_SIZE,
                    step=25,
                    help="Number of words per chunk",
                    key="txt_chunk_size"
                )
            
            with chunk_col2:
                overlap_size = st.number_input(
                    "Overlap Size (words)", 
                    min_value=0, 
                    max_value=min(chunk_size // 2, 200), 
                    value=min(chunk_size // 4, 50),
                    step=5,
                    help="Number of overlapping words between chunks",
                    key="txt_overlap_size"
                )
            
            # Show chunk info
            st.info(f"📊 Each chunk: {chunk_size} words | Overlap: {overlap_size} words | Effective new content per chunk: {chunk_size - overlap_size} words")
        
        else:
            # For CSV and JSON, no chunking parameters needed
            chunk_size = DEFAULT_CHUNK_SIZE
            overlap_size = 0
        
        if st.button("Process and Upload", key="process_upload_btn"):
            if not collection_name:
                st.error("❌ Please select or create a collection first")
                return
            
            # Read and process based on file type
            try:
                if uploaded_file.name.endswith('.pdf'):
                    with st.spinner("Extracting text from PDF..."):
                        chunks = FileProcessor.process_pdf_file(
                            uploaded_file, 
                            extraction_method, 
                            chunk_by, 
                            chunk_size,
                            overlap_size
                        )
                elif uploaded_file.name.endswith('.txt'):
                    file_content = str(uploaded_file.read(), "utf-8")
                    chunks = FileProcessor.process_text_file(file_content, chunk_size, overlap_size)
                elif uploaded_file.name.endswith('.csv'):
                    file_content = str(uploaded_file.read(), "utf-8")
                    chunks = FileProcessor.process_csv_file(file_content)
                elif uploaded_file.name.endswith('.json'):
                    file_content = str(uploaded_file.read(), "utf-8")
                    chunks = FileProcessor.process_json_file(file_content)
                else:
                    st.error("❌ Unsupported file type")
                    return
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                return
            
            if chunks:
                st.write(f"✅ Created {len(chunks)} chunks")
                
                # Show chunk statistics
                avg_length = sum(chunk.get("text_length", 0) for chunk in chunks) / len(chunks)
                st.write(f"📊 Average chunk length: {avg_length:.0f} characters")
                
                # Show effective overlap info for chunked files
                if overlap_size > 0:
                    st.write(f"🔗 Overlap: {overlap_size} words between consecutive chunks")
                
                # Show sample chunk
                with st.expander("📄 Sample Chunk"):
                    sample_chunk = chunks[0]
                    st.write("**Text:**")
                    sample_text = sample_chunk["text"]
                    if len(sample_text) > 500:
                        st.write(sample_text[:500] + "...")
                        if st.checkbox("Show full sample text"):
                            st.write(sample_text)
                    else:
                        st.write(sample_text)
                    st.write("**Metadata:**")
                    st.json({k: v for k, v in sample_chunk.items() if k != "text"})
                
                # Metadata with user information
                metadata = {
                    "filename": uploaded_file.name,
                    "file_type": uploaded_file.type if hasattr(uploaded_file, 'type') else uploaded_file.name.split('.')[-1],
                    "file_size": uploaded_file.size,
                    "total_chunks": len(chunks),
                    "chunk_size": chunk_size,
                    "overlap_size": overlap_size,
                    "upload_timestamp": datetime.now().isoformat()
                }
                
                # Upload to Qdrant
                with st.spinner("Uploading to Qdrant..."):
                    success = DataUploader.upload_to_qdrant(
                        st.session_state.qdrant_client,
                        collection_name,
                        chunks,
                        st.session_state.model,
                        st.session_state.model_type,
                        metadata,
                        get_api_key()
                    )
                
                if success:
                    st.balloons()
            else:
                st.error("❌ No chunks were created from the file")

def render_search_tab():
    """Render search tab with user isolation"""
    st.header("🔍 Semantic Search")
    
    # Add cache clear button
    if st.button("🔄 Clear Cache", key="clear_search_cache"):
        st.cache_data.clear()
        st.success("✅ Search cache cleared!")
    
    # Collection selection for search
    if st.session_state.user_collections:
        search_collection = st.selectbox(
            "Select Collection to Search",
            st.session_state.user_collections,
            key="search_collection_select"
        )
        
        # Show current model info
        if st.session_state.model:
            st.info(f"🤖 Using {st.session_state.model_type}: {st.session_state.model}")
        
        # Show user access info
        if st.session_state.current_user:
            user_role = st.session_state.current_user.get("role")
            if user_role == "super_admin":
                st.info("👑 Super Admin: You can search across all collections and see all documents")
            else:
                st.info("👤 User: You can only search your own documents")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter search query",
                placeholder="What topics are covered in the documents?",
                key="search_query_input"
            )
        with col2:
            limit = st.slider("Results", 1, 20, 5, key="search_limit")
        
        # Search functionality with better state management
        search_triggered = False
        
        # Search button
        if st.button("🔍 Search", key="search_btn"):
            search_triggered = True
        
        # Auto-search when query changes
        if query and query != st.session_state.last_query:
            search_triggered = True
            
        if search_triggered and query.strip():
            st.session_state.last_query = query
            
            # Clear any cached results by using a unique key
            search_key = f"{search_collection}_{query}_{limit}_{hash(str(datetime.now()))}"
            
            with st.spinner("🔍 Searching..."):
                results = SearchManager.search_qdrant(
                    st.session_state.qdrant_client,
                    search_collection,
                    query,
                    st.session_state.model,
                    st.session_state.model_type,
                    limit,
                    get_api_key()
                )
            
            if results:
                st.subheader(f"🎯 Search Results ({len(results)} found)")
                
                # Add AI Summary section
                if st.button("🤖 Generate AI Summary", key=f"summary_btn_{hash(query)}"):
                    current_api_key = get_api_key()
                    if current_api_key:
                        with st.spinner("🤖 Generating AI summary..."):
                            summary = SummaryManager.summarize_search_results(results, query, current_api_key)
                            if summary:
                                st.success("**🎯 AI Summary:**")
                                st.write(summary)
                                st.divider()
                            else:
                                st.error("❌ Failed to generate summary")
                    else:
                        st.error("❌ OpenAI API key required for AI summaries")
                
                # Enhanced results display with user metadata
                for i, result in enumerate(results):
                    with st.expander(f"📄 Result {i+1} - Score: {result.score:.4f}", expanded=(i < 3)):
                        # Get text content
                        text = result.payload.get("text", "") or result.payload.get("pageContent", "No text available")
                        
                        # Show user ownership info if super admin
                        metadata = result.payload.get("metadata", {})
                        if st.session_state.current_user and st.session_state.current_user.get("role") == "super_admin":
                            uploaded_by = metadata.get("uploaded_by", "Unknown")
                            st.caption(f"👤 Uploaded by: {uploaded_by}")
                        
                        st.write("**Content:**")
                        if len(text) > 1000:
                            # Show preview first
                            st.write(text[:1000] + "...")
                            
                            # Use session state to track if full text should be shown
                            show_full_key = f"show_full_{i}_{hash(query)}_{search_collection}"
                            if show_full_key not in st.session_state:
                                st.session_state[show_full_key] = False
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Show full text" if not st.session_state[show_full_key] else f"Hide full text", 
                                           key=f"toggle_full_{i}_{hash(query)}_{search_collection}"):
                                    st.session_state[show_full_key] = not st.session_state[show_full_key]
                                    st.rerun()
                            
                            with col2:
                                # Add individual document summary
                                if st.button(f"📝 Summarize", key=f"summarize_doc_{i}_{hash(query)}"):
                                    current_api_key = get_api_key()
                                    if current_api_key:
                                        with st.spinner("📝 Summarizing document..."):
                                            doc_messages = [
                                                {
                                                    "role": "system",
                                                    "content": "You are an AI assistant that creates clear, concise summaries of documents. Focus on key points and make the content easy to understand."
                                                },
                                                {
                                                    "role": "user",
                                                    "content": f"Please provide a clear, concise summary of this document excerpt:\n\n{text[:2000]}"
                                                }
                                            ]
                                            doc_summary = SummaryManager.get_openai_chat_response(doc_messages, current_api_key)
                                            if doc_summary:
                                                st.info(f"**📝 Document Summary:**\n{doc_summary}")
                            
                            # Show full text if toggled on
                            if st.session_state[show_full_key]:
                                st.write("**Full Content:**")
                                st.write(text)
                        else:
                            # For shorter texts, still offer summarization
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(text)
                            with col2:
                                if st.button(f"📝 Summarize", key=f"summarize_short_{i}_{hash(query)}"):
                                    current_api_key = get_api_key()
                                    if current_api_key:
                                        with st.spinner("📝 Summarizing..."):
                                            doc_messages = [
                                                {
                                                    "role": "system",
                                                    "content": "You are an AI assistant that creates clear, concise summaries. Focus on key points and make the content easy to understand."
                                                },
                                                {
                                                    "role": "user",
                                                    "content": f"Please provide a clear, concise summary of this text:\n\n{text}"
                                                }
                                            ]
                                            doc_summary = SummaryManager.get_openai_chat_response(doc_messages, current_api_key)
                                            if doc_summary:
                                                st.info(f"**📝 Summary:**\n{doc_summary}")
                        
                        # Show metadata
                        if metadata:
                            st.write("**Metadata:**")
                            # Format metadata nicely, excluding user info for regular users
                            for key, value in metadata.items():
                                if key not in ["text", "pageContent"]:
                                    # Hide user info from non-super admin users
                                    if key in ["uploaded_by", "user_role"] and st.session_state.current_user.get("role") != "super_admin":
                                        continue
                                    st.write(f"- **{key}:** {value}")
            else:
                st.info("🔍 No results found for your query")
        elif query.strip() and not search_triggered:
            st.info("👆 Click Search button or press Enter to search")
        elif not query.strip() and search_triggered:
            st.warning("⚠️ Please enter a search query")
    else:
        st.info("📚 No collections available. Please create a collection and upload files first.")


def main():
    """Main application function with auto-connection"""
    # Initialize session state
    init_session_state()

    # Check required packages
    required_viz_packages = ['plotly', 'kaleido']
    missing_viz_packages = []
    
    for package in required_viz_packages:
        try:
            __import__(package)
        except ImportError:
            missing_viz_packages.append(package)
    
    if missing_viz_packages:
        st.error("📦 Missing required packages for CSV visualization!")
        st.markdown(f"""
        **Please install the following packages:**
        ```bash
        pip install {' '.join(missing_viz_packages)}
        ```
        """)
        return
    
    # Check authentication
    if not st.session_state.authenticated:
        render_login_page()
        return
    
    # Auto-connect services after login
    auto_connect_services()
    
    # Render authenticated app
    render_header()
    
    # Render sidebar with question management (use fixed version)
    try:
        render_sidebar_with_questions()
    except Exception as e:
        st.error(f"Sidebar error: {str(e)}")
        # Fallback to basic sidebar
        render_sidebar()
    
    # Render status indicators
    render_status_indicators()
    
    # Check prerequisites (more lenient - allow app to work without connections)
    if not st.session_state.connection_status:
        st.warning("⚠️ Qdrant not connected. Some features may be limited.")
    
    if not st.session_state.model_status:
        st.warning("⚠️ No embedding model loaded. Some features may be limited.")
    
    # Show question manager if requested
    if st.session_state.get("show_question_manager", False):
        try:
            render_question_manager()
        except Exception as e:
            st.error(f"Question manager error: {str(e)}")
            st.session_state.show_question_manager = False
        
        if st.button("🔙 Back to Main App", key="back_to_main"):
            st.session_state.show_question_manager = False
            st.rerun()
        return
    
    # Main content tabs - add user management for super admin
    if st.session_state.current_user and st.session_state.current_user.get("role") == "super_admin":
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📁 Upload Files", 
            "🔍 Search", 
            "💬 Webhook Chat", 
            "📊 Collections", 
            "👥 User Management",
            "💾 Questions",
            "📊 Visualization"
        ])
        with tab5:
            render_user_management_tab()
        with tab6:
            try:
                render_question_manager()
            except Exception as e:
                st.error(f"Question manager tab error: {str(e)}")
        with tab7:
            render_data_visualization_tab()
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📁 Upload Files", 
            "🔍 Search", 
            "💬 Webhook Chat", 
            "📊 Collections",
            "💾 Questions",
            "📊 Visualization"
        ])
        with tab5:
            try:
                render_question_manager_fixed()
            except Exception as e:
                st.error(f"Question manager tab error: {str(e)}")
        with tab6:
            render_data_visualization_tab()
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_search_tab()
    
    with tab3:
        render_chat_tab_with_questions()
    
    with tab4:
        render_collections_tab()

# Check required packages installation
def check_visualization_packages():
    """Check if required visualization packages are installed"""
    required_packages = {
        'plotly': 'plotly',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'seaborn': 'seaborn',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        st.error("📦 Missing required packages for visualization!")
        st.markdown(f"""
        **Please install the following packages:**
        ```bash
        pip install {' '.join(missing_packages)}
        ```
        
        **For complete visualization support:**
        ```bash
        pip install plotly pandas numpy seaborn matplotlib kaleido
        ```
        """)
        return False
    
    return True

def render_sidebar_with_questions():
    """Enhanced sidebar with question management and sentence transformer defaults"""
    st.sidebar.header("⚙️ Configuration")
    
    # Show current user info
    if st.session_state.current_user:
        with st.sidebar.container():
            user = st.session_state.current_user
            role_icon = "👑" if user.get("role") == "super_admin" else "👤"
            st.sidebar.success(f"{role_icon} **{user.get('username')}**")
            st.sidebar.caption(f"Role: {user.get('role', 'user')}")
            
            # Quick save button
            if st.sidebar.button("💾 Save Config", key="quick_save_config"):
                try:
                    if save_user_session_config():
                        st.sidebar.success("✅ Config saved!")
                    else:
                        st.sidebar.error("❌ Save failed!")
                except Exception as e:
                    st.sidebar.error(f"Save error: {str(e)}")
            
            st.sidebar.divider()
    
    # Question management in sidebar
    try:
        selected_question = render_question_manager_sidebar_fixed()
        if selected_question:
            st.session_state.selected_question = selected_question
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Question sidebar error: {str(e)}")
    
    # Qdrant connection settings
    with st.sidebar.expander("🔗 Qdrant Connection", expanded=True):
        host = st.text_input("Host", value=st.session_state.get("qdrant_host", "......"), key="qdrant_host")
        port = st.number_input("Port", value=st.session_state.get("qdrant_port", 6333), min_value=1, max_value=65535, key="qdrant_port")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", key="connect_qdrant"):
                client = QdrantManager.connect(host, port)
                if client:
                    st.session_state.qdrant_client = client
                    st.session_state.connection_status = True
                    st.success("✅ Connected!")
                    
                    # Load collections with user filtering
                    try:
                        collections = client.get_collections().collections
                        all_collections = [col.name for col in collections]
                        st.session_state.collections = all_collections
                        
                        # Filter collections for current user
                        if st.session_state.current_user:
                            username = st.session_state.current_user["username"]
                            st.session_state.user_collections = get_user_collections(username, all_collections)
                    except Exception as e:
                        st.error(f"❌ Failed to get collections: {str(e)}")
                    
                    # Auto-save config
                    auto_save_config_if_enabled()
        
        with col2:
            auto_connect = st.checkbox("Auto", value=st.session_state.get("qdrant_auto_connect", True), key="qdrant_auto_connect")
    
    # Model loading with sentence transformer as default
    with st.sidebar.expander("🤖 Embedding Model", expanded=True):
        model_type = st.selectbox("Model Type", ["sentence-transformers", "openai"], 
                                 index=0, key="model_type_select")  # sentence-transformers first
        
        if model_type == "sentence-transformers":
            # Sentence Transformers with all-mpnet-base-v2 as default
            default_model = st.session_state.get('default_sentence_model', 'all-mpnet-base-v2')
            try:
                default_index = SENTENCE_TRANSFORMER_MODELS.index(default_model)
            except ValueError:
                default_index = 1  # fallback to all-mpnet-base-v2 if it's in the list
            
            selected_model = st.selectbox("Select Model", SENTENCE_TRANSFORMER_MODELS, 
                                        index=default_index, key="st_model_select")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Model", key="load_st_model"):
                    with st.spinner("Loading model..."):
                        try:
                            model = EmbeddingManager.load_sentence_transformer(selected_model)
                            st.session_state.model = model
                            st.session_state.model_type = "sentence-transformers"
                            st.session_state.model_status = True
                            vector_size = EmbeddingManager.get_vector_size(model, "sentence-transformers")
                            st.success(f"✅ Loaded: {selected_model}")
                            st.info(f"Vector size: {vector_size}D")
                            auto_save_config_if_enabled()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            with col2:
                auto_load = st.checkbox("Auto", value=st.session_state.get("sentence_model_auto_load", True), key="sentence_auto_load")
        
        else:
            # OpenAI configuration (same as before)
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                value=st.session_state.get("openai_api_key", ""),
                help="Your OpenAI API key",
                key="openai_key_input"
            )
            if api_key != st.session_state.get("openai_api_key", ""):
                st.session_state.openai_api_key = api_key
            
            selected_model = st.selectbox("Select OpenAI Model", OPENAI_MODELS, 
                                        index=1, key="openai_model_select")  # text-embedding-3-small
            
            if st.button("🔄 Load OpenAI Model", key="load_openai_model"):
                current_api_key = get_api_key()
                if current_api_key:
                    try:
                        test_embedding = EmbeddingManager.get_openai_embedding("test", current_api_key, selected_model)
                        if test_embedding:
                            st.session_state.model = selected_model
                            st.session_state.model_type = "openai"
                            st.session_state.model_status = True
                            st.success(f"✅ Loaded: {selected_model}")
                            st.info(f"Vector size: {len(test_embedding)}D")
                            auto_save_config_if_enabled()
                        else:
                            st.error("❌ Failed - check API key")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                else:
                    st.error("❌ Please enter API key")
        
        # Display current model info
        if st.session_state.model_status and st.session_state.model:
            st.success(f"🤖 Current: {st.session_state.model} ({st.session_state.model_type})")
        else:
            st.warning("⚠️ No model loaded")

# Quick access widget for questions
def render_quick_questions_widget():
    """Render a compact questions widget for quick access"""
    if not st.session_state.current_user:
        return
    
    username = st.session_state.current_user["username"]
    questions = QuestionManager.load_user_questions(username)
    
    if questions:
        # Show top 3 most used questions
        top_questions = sorted(questions, key=lambda x: x.get("used_count", 0), reverse=True)[:3]
        
        if top_questions:
            st.write("🔥 **Quick Questions:**")
            for q in top_questions:
                question_text = q["question"]
                if len(question_text) > 40:
                    question_text = question_text[:40] + "..."
                
                if st.button(f"⚡ {question_text}", key=f"quick_{q['id']}", help=q["question"]):
                    QuestionManager.update_question_usage(username, q["id"])
                    st.session_state.selected_question = q["question"]
                    st.rerun()

# Add this new class for CSV data processing and visualization
class CSVVisualizationManager:
    """Handles CSV data visualization requests from n8n AI"""
    
    @staticmethod
    def debug_data_types(obj, path="root"):
        """Debug helper to identify non-serializable objects"""
        try:
            json.dumps(obj)
            return True
        except TypeError as e:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if not CSVVisualizationManager.debug_data_types(value, f"{path}.{key}"):
                        print(f"Non-serializable object at {path}.{key}: {type(value).__name__} = {value}")
                        return False
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if not CSVVisualizationManager.debug_data_types(item, f"{path}[{i}]"):
                        return False
            else:
                print(f"Non-serializable object at {path}: {type(obj).__name__} = {repr(obj)}")
                return False
            return True
    
    @staticmethod
    def make_json_serializable(obj):
        """Convert pandas/numpy objects to JSON-serializable Python types"""
        if isinstance(obj, dict):
            return {key: CSVVisualizationManager.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [CSVVisualizationManager.make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'dtype'):  # pandas/numpy objects
            if pd.isna(obj):
                return None
            elif obj.dtype.kind in 'fc':  # float or complex
                return float(obj)
            elif obj.dtype.kind in 'iu':  # integer or unsigned integer
                return int(obj)
            elif obj.dtype.kind == 'b':  # boolean
                return bool(obj)
            else:
                return str(obj)
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    @staticmethod
    def extract_csv_data_from_collection(client: QdrantClient, collection_name: str, 
                                       username: str = None, limit: int = 1000) -> pd.DataFrame:
        """Extract CSV data from Qdrant collection and reconstruct as DataFrame"""
        try:
            # Create user filter for search if not super admin
            search_filter = None
            if username and st.session_state.current_user and st.session_state.current_user.get("role") != "super_admin":
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.uploaded_by",
                            match=MatchValue(value=username)
                        ),
                        FieldCondition(
                            key="metadata.file_type", 
                            match=MatchValue(value="text/csv")
                        )
                    ]
                )
            else:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.file_type",
                            match=MatchValue(value="text/csv")
                        )
                    ]
                )
            
            # Get CSV chunks from collection
            points, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            if not points:
                return pd.DataFrame()
            
            # Reconstruct CSV data from chunks
            csv_rows = []
            for point in points:
                text_content = (
                    point.payload.get("pageContent") or 
                    point.payload.get("text") or 
                    point.payload.get("content") or ""
                )
                
                # Parse the CSV row format: "col1: val1 | col2: val2 | ..."
                if " | " in text_content:
                    row_data = {}
                    pairs = text_content.split(" | ")
                    for pair in pairs:
                        if ": " in pair:
                            key, value = pair.split(": ", 1)
                            # Try to convert to numeric if possible
                            try:
                                if '.' in value:
                                    row_data[key.strip()] = float(value.strip())
                                elif value.strip().isdigit() or (value.strip().startswith('-') and value.strip()[1:].isdigit()):
                                    row_data[key.strip()] = int(value.strip())
                                else:
                                    row_data[key.strip()] = value.strip()
                            except ValueError:
                                row_data[key.strip()] = value.strip()
                    
                    if row_data:
                        csv_rows.append(row_data)
            
            if csv_rows:
                df = pd.DataFrame(csv_rows)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error extracting CSV data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def create_visualization_from_query(df: pd.DataFrame, query: str, api_key: str) -> dict:
        """Use OpenAI to interpret the query and create appropriate visualization"""
        if df.empty:
            return {"success": False, "error": "No data available"}
        
        try:
            # Create data summary for AI context with JSON-safe types
            data_info = {
                "columns": df.columns.tolist(),
                "shape": df.shape,
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                "sample_data": CSVVisualizationManager.make_json_serializable(df.head(3).to_dict('records')),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
            }
            
            # Create AI prompt for visualization recommendation
            messages = [
                {
                    "role": "system",
                    "content": """You are a data visualization expert. Based on the user's query and the CSV data provided, recommend the best visualization and provide the exact parameters needed.

Your response must be a JSON object with this exact structure:
{
    "chart_type": "line|bar|scatter|pie|histogram|box|heatmap",
    "x_column": "column_name",
    "y_column": "column_name", 
    "color_column": "column_name_or_null",
    "title": "Chart Title",
    "description": "Brief description of what the chart shows",
    "aggregation": "sum|mean|count|none"
}

Chart type guidelines:
- line: for trends over time or continuous data
- bar: for comparing categories or counts
- scatter: for relationships between two numeric variables
- pie: for showing parts of a whole (use with aggregated data)
- histogram: for distribution of a single numeric variable
- box: for showing distribution and outliers
- heatmap: for correlation between numeric variables

Only use columns that exist in the data. If aggregation is needed, specify how to aggregate the data."""
                },
                {
                    "role": "user",
                    "content": f"""Data Information:
- Shape: {data_info['shape'][0]} rows, {data_info['shape'][1]} columns
- Columns: {', '.join(data_info['columns'])}
- Numeric columns: {', '.join(data_info['numeric_columns'])}
- Categorical columns: {', '.join(data_info['categorical_columns'])}
- Data types: {data_info['dtypes']}
- Sample data: {data_info['sample_data']}

User Query: "{query}"

Based on this data and query, what visualization would be most appropriate? Respond with only the JSON object."""
                }
            ]
            
            # Get AI recommendation
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content.strip()
            
            # Clean up the response (remove markdown formatting if present)
            if "```json" in ai_response:
                ai_response = ai_response.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_response:
                ai_response = ai_response.split("```")[1].split("```")[0].strip()
            
            try:
                viz_config = json.loads(ai_response)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                viz_config = {
                    "chart_type": "bar",
                    "x_column": data_info['columns'][0] if data_info['columns'] else None,
                    "y_column": data_info['numeric_columns'][0] if data_info['numeric_columns'] else data_info['columns'][1] if len(data_info['columns']) > 1 else None,
                    "color_column": None,
                    "title": f"Analysis of {query}",
                    "description": "Automatically generated visualization",
                    "aggregation": "none"
                }
            
            # Convert all data to JSON-serializable format
            viz_config = CSVVisualizationManager.make_json_serializable(viz_config)
            
            # Ensure all data is JSON serializable
            json_safe_summary = {
                "rows": len(df),
                "columns": df.columns.tolist(),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            }
            
            return {"success": True, "config": viz_config, "data_summary": json_safe_summary}
            
        except Exception as e:
            return {"success": False, "error": f"AI visualization error: {str(e)}"}
    
    @staticmethod
    def generate_chart(df: pd.DataFrame, config: dict) -> go.Figure:
        """Generate Plotly chart with comprehensive data type handling"""
        try:
            chart_type = config.get("chart_type", "bar")
            x_col = config.get("x_column")
            y_col = config.get("y_column") 
            color_col = config.get("color_column")
            title = str(config.get("title", "Data Visualization"))  # Force string
            aggregation = config.get("aggregation", "none")
            
            # STEP 1: Validate and fix column names
            if x_col and x_col not in df.columns:
                x_col = df.columns[0] if len(df.columns) > 0 else None
            if y_col and y_col not in df.columns:
                y_col = df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            
            # STEP 2: Create a clean copy of the dataframe
            plot_df = df.copy()
            
            # STEP 3: COMPREHENSIVE DATA TYPE CLEANING
            # Clean ALL columns to ensure consistent data types
            for col in plot_df.columns:
                try:
                    # Convert to string first, then handle each column appropriately
                    plot_df[col] = plot_df[col].astype(str)
                    
                    # Try to convert back to numeric if it looks like numbers
                    if col == y_col:  # Y-axis should be numeric
                        # Remove any currency symbols, commas, etc.
                        plot_df[col] = plot_df[col].str.replace('$', '').str.replace(',', '').str.replace('%', '')
                        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                        plot_df[col] = plot_df[col].fillna(0)  # Replace NaN with 0
                    elif col in [x_col, color_col]:  # Categorical columns stay as strings
                        plot_df[col] = plot_df[col].astype(str)
                        # Replace 'nan' strings with 'Unknown'
                        plot_df[col] = plot_df[col].replace(['nan', 'None', 'NaN'], 'Unknown')
                except Exception as e:
                    # If any conversion fails, keep as string
                    plot_df[col] = plot_df[col].astype(str).replace(['nan', 'None', 'NaN'], 'Unknown')
            
            # STEP 4: Apply aggregation with safe operations
            if aggregation != "none" and x_col and y_col:
                try:
                    if aggregation == "sum":
                        plot_df = plot_df.groupby(x_col, as_index=False)[y_col].sum()
                    elif aggregation == "mean":
                        plot_df = plot_df.groupby(x_col, as_index=False)[y_col].mean()
                    elif aggregation == "count":
                        plot_df = plot_df.groupby(x_col, as_index=False).size().reset_index(name='count')
                        y_col = 'count'
                except Exception as e:
                    # If aggregation fails, use original data
                    pass
            
            # STEP 5: Final data validation
            if x_col not in plot_df.columns or y_col not in plot_df.columns:
                raise ValueError(f"Required columns not found: x_col={x_col}, y_col={y_col}")
            
            if len(plot_df) == 0:
                raise ValueError("No data available for chart generation")
            
            # STEP 6: Create the chart with safe data
            fig = go.Figure()
            
            try:
                if chart_type == "bar":
                    # Ensure all data is properly typed for bar chart
                    x_data = plot_df[x_col].astype(str).tolist()
                    y_data = pd.to_numeric(plot_df[y_col], errors='coerce').fillna(0).tolist()
                    
                    if color_col and color_col in plot_df.columns:
                        # Grouped bar chart
                        color_data = plot_df[color_col].astype(str).tolist()
                        unique_colors = list(set(color_data))
                        
                        for color in unique_colors:
                            mask = plot_df[color_col].astype(str) == color
                            fig.add_trace(go.Bar(
                                name=str(color),
                                x=plot_df[mask][x_col].astype(str).tolist(),
                                y=pd.to_numeric(plot_df[mask][y_col], errors='coerce').fillna(0).tolist(),
                                text=[f'{val:.1f}' if val > 0 else '0' for val in pd.to_numeric(plot_df[mask][y_col], errors='coerce').fillna(0)],
                                textposition='auto'
                            ))
                    else:
                        # Simple bar chart
                        fig.add_trace(go.Bar(
                            x=x_data,
                            y=y_data,
                            text=[f'{val:.1f}' if val > 0 else '0' for val in y_data],
                            textposition='auto',
                            marker_color='#1f77b4'
                        ))
                
                elif chart_type == "line":
                    x_data = plot_df[x_col].astype(str).tolist()
                    y_data = pd.to_numeric(plot_df[y_col], errors='coerce').fillna(0).tolist()
                    
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='lines+markers',
                        name='Line Chart'
                    ))
                
                elif chart_type == "pie":
                    # For pie charts, use value counts
                    value_counts = plot_df[x_col].value_counts()
                    fig.add_trace(go.Pie(
                        labels=[str(label) for label in value_counts.index],
                        values=value_counts.values.tolist(),
                        textinfo='label+percent'
                    ))
                
                else:  # Default to bar chart
                    x_data = plot_df[x_col].astype(str).tolist()
                    y_data = pd.to_numeric(plot_df[y_col], errors='coerce').fillna(0).tolist()
                    
                    fig.add_trace(go.Bar(
                        x=x_data,
                        y=y_data,
                        text=[f'{val:.1f}' for val in y_data],
                        textposition='auto'
                    ))
            
            except Exception as e:
                # Create error chart
                fig.add_annotation(
                    text=f"Chart creation error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
            
            # STEP 7: Update layout with safe string conversion
            try:
                fig.update_layout(
                    title=str(title),
                    xaxis_title=str(x_col) if x_col else "Categories",
                    yaxis_title=str(y_col) if y_col else "Values",
                    template='plotly_white',
                    height=500,
                    width=800,
                    showlegend=True if color_col else False
                )
            except Exception as e:
                # Minimal layout if update fails
                fig.update_layout(title="Chart", height=400)
            
            return fig
            
        except Exception as e:
            # Create fallback error chart
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="red")
            )
            error_fig.update_layout(
                title="Chart Generation Error",
                height=400,
                template='plotly_white'
            )
            return error_fig
    
@staticmethod
def generate_chart(df: pd.DataFrame, config: dict) -> go.Figure:
    """Generate Plotly chart based on configuration"""
    try:
        chart_type = config.get("chart_type", "bar")
        x_col = config.get("x_column")
        y_col = config.get("y_column") 
        color_col = config.get("color_column")
        title = config.get("title", "Data Visualization")
        aggregation = config.get("aggregation", "none")
        
        # Validate columns exist
        if x_col and x_col not in df.columns:
            x_col = df.columns[0]
        if y_col and y_col not in df.columns:
            y_col = df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else df.columns[-1]
        
        # Apply aggregation if needed
        plot_df = df.copy()
        if aggregation != "none" and x_col and y_col:
            if aggregation == "sum":
                plot_df = df.groupby(x_col)[y_col].sum().reset_index()
            elif aggregation == "mean":
                plot_df = df.groupby(x_col)[y_col].mean().reset_index()
            elif aggregation == "count":
                plot_df = df.groupby(x_col).size().reset_index(name='count')
                y_col = 'count'
        
        # Create chart based on type
        fig = go.Figure()
        
        if chart_type == "line":
            if color_col and color_col in plot_df.columns:
                for group in plot_df[color_col].unique():
                    group_data = plot_df[plot_df[color_col] == group]
                    fig.add_trace(go.Scatter(
                        x=group_data[x_col],
                        y=group_data[y_col],
                        mode='lines+markers',
                        name=str(group)
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col],
                    y=plot_df[y_col],
                    mode='lines+markers'
                ))
        
        elif chart_type == "bar":
            if color_col and color_col in plot_df.columns:
                fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, title=title)
            else:
                fig.add_trace(go.Bar(
                    x=plot_df[x_col],
                    y=plot_df[y_col]
                ))
        
        elif chart_type == "scatter":
            if color_col and color_col in plot_df.columns:
                fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, title=title)
            else:
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col],
                    y=plot_df[y_col],
                    mode='markers'
                ))
        
        elif chart_type == "pie":
            # For pie charts, use value counts or aggregated data
            if aggregation == "none":
                pie_data = plot_df[x_col].value_counts()
                fig.add_trace(go.Pie(
                    labels=pie_data.index,
                    values=pie_data.values
                ))
            else:
                fig.add_trace(go.Pie(
                    labels=plot_df[x_col],
                    values=plot_df[y_col]
                ))
        
        elif chart_type == "histogram":
            fig.add_trace(go.Histogram(
                x=plot_df[x_col],
                nbinsx=20
            ))
        
        elif chart_type == "box":
            if color_col and color_col in plot_df.columns:
                for group in plot_df[color_col].unique():
                    group_data = plot_df[plot_df[color_col] == group]
                    fig.add_trace(go.Box(
                        y=group_data[y_col],
                        name=str(group)
                    ))
            else:
                fig.add_trace(go.Box(y=plot_df[y_col]))
        
        elif chart_type == "heatmap":
            # Create correlation heatmap for numeric columns
            numeric_df = plot_df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                fig.add_trace(go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu'
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white',
            height=500,
            width=800
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(title="Visualization Error")
        return fig
        
def create_guaranteed_customer_chart():
    """Create a chart that will definitely work"""
    import plotly.graph_objects as go
    
    # Simple, clean data
    customers = ['Customer A', 'Customer B', 'Customer C', 'Customer D']
    sales = [1500, 1200, 1800, 900]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=customers,
        y=sales,
        text=[f'${s:,}' for s in sales],
        textposition='auto',
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ))
    
    fig.update_layout(
        title='Customer Sales Summary',
        xaxis_title='Customers',
        yaxis_title='Sales Amount ($)',
        template='plotly_white',
        height=500
    )
    
    return fig

# Enhanced WebhookManager with CSV visualization support
class EnhancedWebhookManager(WebhookManager):
    """Enhanced webhook manager with CSV visualization capabilities"""
    
    @staticmethod
    def process_csv_visualization_request(webhook_url: str, message: str, collection_name: str, 
                                        timeout: int = 180) -> Dict[str, Any]:
        """Process CSV visualization request and return chart data"""
        try:
            # Check if this is a visualization request
            viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'show', 'display', 'analyze']
            is_viz_request = any(keyword in message.lower() for keyword in viz_keywords)
            
            if not is_viz_request:
                # Use normal webhook processing
                return WebhookManager.send_message(webhook_url, message, timeout)
            
            # Extract CSV data from collection
            username = st.session_state.current_user.get("username") if st.session_state.current_user else None
            df = CSVVisualizationManager.extract_csv_data_from_collection(
                st.session_state.qdrant_client, 
                collection_name, 
                username
            )
            
            if df.empty:
                return {
                    "success": True,
                    "response": "No CSV data found in the collection. Please upload CSV files first.",
                    "chart_data": None
                }
            
            # Get OpenAI API key
            api_key = get_api_key()
            if not api_key:
                return {
                    "success": True,
                    "response": "OpenAI API key required for CSV visualization analysis.",
                    "chart_data": None
                }
            
            # Generate visualization configuration using AI
            viz_result = CSVVisualizationManager.create_visualization_from_query(df, message, api_key)
            
            if not viz_result["success"]:
                return {
                    "success": True,
                    "response": f"Could not create visualization: {viz_result.get('error', 'Unknown error')}",
                    "chart_data": None
                }
            
            # Generate the chart
            config = viz_result["config"]
            chart = CSVVisualizationManager.generate_chart(df, config)
            
            # Convert chart to JSON for frontend
            chart_json = chart.to_json()
            
            # Create text response
            response_text = f"""📊 **Data Visualization Generated**

**Chart Type:** {config.get('chart_type', 'Unknown').title()}
**Title:** {config.get('title', 'Data Analysis')}
**Description:** {config.get('description', 'Visualization based on your query')}

**Data Summary:**
- Total records: {len(df)}
- Columns analyzed: {', '.join(df.columns.tolist())}

The visualization has been generated and displayed above. You can export this as PDF using the export options."""
            
            return {
                "success": True,
                "response": response_text,
                "chart_data": {
                    "chart_json": chart_json,
                    "config": config,
                    "data_summary": {
                        "rows": len(df),
                        "columns": df.columns.tolist(),
                        "data_types": df.dtypes.to_dict()
                    }
                },
                "response_time": 1.0  # Placeholder
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"CSV visualization error: {str(e)}",
                "chart_data": None
            }
        
class ThaiPDFGenerator:
    """Generate PDF reports with Thai-English support using ReportLab"""
    
    def __init__(self):
        self.setup_fonts()
        self.setup_styles()
    
    def setup_fonts(self):
        """Setup Thai-English font support"""
        try:
            # Try to register common Thai fonts
            thai_fonts = [
                # System fonts (Windows)
                "C:/Windows/Fonts/tahoma.ttf",
                "C:/Windows/Fonts/tahomabd.ttf",
                # System fonts (macOS)
                "/System/Library/Fonts/Thonburi.ttc",
                "/Library/Fonts/Thonburi.ttc",
                # Linux fonts
                "/usr/share/fonts/truetype/thai/Loma.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                # Google Fonts alternatives
                "./fonts/NotoSansThai-Regular.ttf",
                "./fonts/Sarabun-Regular.ttf"
            ]
            
            self.font_registered = False
            
            for font_path in thai_fonts:
                try:
                    if os.path.exists(font_path):
                        if "tahoma" in font_path.lower():
                            pdfmetrics.registerFont(TTFont('ThaiFont', font_path))
                            if "tahomabd" in font_path.lower():
                                pdfmetrics.registerFont(TTFont('ThaiFontBold', font_path))
                            else:
                                pdfmetrics.registerFont(TTFont('ThaiFontBold', font_path))
                        else:
                            pdfmetrics.registerFont(TTFont('ThaiFont', font_path))
                            pdfmetrics.registerFont(TTFont('ThaiFontBold', font_path))
                        
                        self.font_registered = True
                        self.thai_font = 'ThaiFont'
                        self.thai_font_bold = 'ThaiFontBold'
                        break
                except Exception as e:
                    continue
            
            if not self.font_registered:
                # Fallback to DejaVu fonts (usually available)
                try:
                    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
                    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
                    self.thai_font = 'STSong-Light'
                    self.thai_font_bold = 'STSong-Light'
                    self.font_registered = True
                except:
                    # Ultimate fallback to Helvetica (no Thai support)
                    self.thai_font = 'Helvetica'
                    self.thai_font_bold = 'Helvetica-Bold'
                    self.font_registered = False
                    
        except Exception as e:
            # Fallback fonts
            self.thai_font = 'Helvetica'
            self.thai_font_bold = 'Helvetica-Bold'
            self.font_registered = False
    
    def setup_styles(self):
        """Setup paragraph styles for different elements"""
        styles = getSampleStyleSheet()
        
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontName=self.thai_font_bold,
            fontSize=18,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontName=self.thai_font_bold,
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkgreen
        )
        
        # Question style
        self.question_style = ParagraphStyle(
            'CustomQuestion',
            parent=styles['Normal'],
            fontName=self.thai_font_bold,
            fontSize=12,
            spaceAfter=8,
            spaceBefore=15,
            textColor=colors.darkblue,
            leftIndent=20
        )
        
        # Answer style
        self.answer_style = ParagraphStyle(
            'CustomAnswer',
            parent=styles['Normal'],
            fontName=self.thai_font,
            fontSize=11,
            spaceAfter=12,
            spaceBefore=5,
            alignment=TA_JUSTIFY,
            leftIndent=20,
            rightIndent=20
        )
        
        # Metadata style
        self.meta_style = ParagraphStyle(
            'CustomMeta',
            parent=styles['Normal'],
            fontName=self.thai_font,
            fontSize=9,
            textColor=colors.grey,
            spaceAfter=5
        )
        
        # Header/Footer style
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Normal'],
            fontName=self.thai_font,
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
    
    def clean_text(self, text):
        """Clean and prepare text for PDF rendering"""
        if not text:
            return ""
        
        # Convert to string and handle encoding
        text = str(text)
        
        # Replace problematic characters
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '\r\n': '<br/>',
            '\n': '<br/>',
            '\r': '<br/>',
            '\t': '    '
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def create_header_footer(self, canvas, doc):
        """Create header and footer for each page"""
        canvas.saveState()
        
        # Header
        header_text = f"Chat Export - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        canvas.setFont(self.thai_font, 10)
        canvas.setFillColor(colors.grey)
        canvas.drawString(50, A4[1] - 50, header_text)
        
        # Footer
        footer_text = f"Page {doc.page}"
        canvas.drawRightString(A4[0] - 50, 50, footer_text)
        
        canvas.restoreState()
    
    def generate_chat_pdf(self, chat_history, user_info=None):
        """Generate PDF from chat history with Thai-English support"""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.close()
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                temp_file.name,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=3*cm,
                bottomMargin=2*cm
            )
            
            # Build content
            story = []
            
            # Title
            title_text = "💬 Chat Conversation Export"
            story.append(Paragraph(self.clean_text(title_text), self.title_style))
            story.append(Spacer(1, 20))
            
            # User info and metadata
            if user_info:
                user_text = f"👤 User: {user_info.get('username', 'Unknown')} | Role: {user_info.get('role', 'User')}"
                story.append(Paragraph(self.clean_text(user_text), self.meta_style))
            
            export_time = f"📅 Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(self.clean_text(export_time), self.meta_style))
            story.append(Spacer(1, 20))
            
            # Warning about font support
            if not self.font_registered:
                warning_text = "⚠️ Thai font not available - some characters may not display correctly"
                warning_style = ParagraphStyle(
                    'Warning',
                    parent=self.meta_style,
                    textColor=colors.red,
                    fontSize=10
                )
                story.append(Paragraph(warning_text, warning_style))
                story.append(Spacer(1, 10))
            
            # Process chat history
            qa_pairs = self.extract_qa_pairs(chat_history)
            
            if not qa_pairs:
                no_data_text = "📝 No conversation data available"
                story.append(Paragraph(self.clean_text(no_data_text), self.answer_style))
            else:
                # Add summary
                summary_text = f"📊 Total Conversations: {len(qa_pairs)}"
                story.append(Paragraph(self.clean_text(summary_text), self.subtitle_style))
                story.append(Spacer(1, 15))
                
                # Add Q&A pairs
                for i, qa in enumerate(qa_pairs, 1):
                    # Question
                    question_text = f"❓ Q{i}: {qa.get('question', 'No question')}"
                    story.append(Paragraph(self.clean_text(question_text), self.question_style))
                    
                    # Answer
                    answer_text = f"💡 A{i}: {qa.get('response', 'No response')}"
                    story.append(Paragraph(self.clean_text(answer_text), self.answer_style))
                    
                    # Timestamp if available
                    if qa.get('timestamp'):
                        time_text = f"🕒 Time: {qa['timestamp']}"
                        story.append(Paragraph(self.clean_text(time_text), self.meta_style))
                    
                    # Add separator between Q&A pairs
                    if i < len(qa_pairs):
                        story.append(Spacer(1, 15))
                        # Add a subtle line separator
                        line_data = [[''] * 3]
                        line_table = Table(line_data, colWidths=[A4[0]-4*cm])
                        line_table.setStyle(TableStyle([
                            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.lightgrey),
                        ]))
                        story.append(line_table)
                        story.append(Spacer(1, 15))
            
            # Footer info
            story.append(Spacer(1, 30))
            footer_info = f"📋 Generated by Qdrant File Reader | Total Pages: [PAGE_COUNT]"
            story.append(Paragraph(self.clean_text(footer_info), self.meta_style))
            
            # Build PDF
            doc.build(story, onFirstPage=self.create_header_footer, onLaterPages=self.create_header_footer)
            
            # Read the generated PDF
            with open(temp_file.name, 'rb') as f:
                pdf_data = f.read()
            
            return pdf_data
            
        except Exception as e:
            raise Exception(f"Error generating PDF: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def extract_qa_pairs(self, chat_history):
        """Extract Q&A pairs from chat history"""
        qa_pairs = []
        current_question = None
        
        for chat in chat_history:
            if chat.get("type") == "user":
                current_question = {
                    "question": chat.get("message", ""),
                    "timestamp": chat.get("timestamp", ""),
                    "response": None
                }
            elif chat.get("type") == "assistant" and current_question:
                # Handle different response formats
                message_content = chat.get("message", "")
                if isinstance(message_content, str):
                    try:
                        # Try to parse as JSON
                        import json
                        data = json.loads(message_content)
                        response_text = (
                            data.get('response') or 
                            data.get('text') or 
                            data.get('answer') or
                            str(data)
                        )
                    except json.JSONDecodeError:
                        response_text = message_content
                else:
                    response_text = str(message_content)
                
                current_question["response"] = response_text
                qa_pairs.append(current_question)
                current_question = None
        
        return qa_pairs


# Enhanced PDF generator with chart support
class EnhancedThaiPDFGenerator(ThaiPDFGenerator):
    """Enhanced PDF generator with chart visualization support"""
    
    def add_chart_to_story(self, story: list, chart_fig: go.Figure, title: str = "Chart"):
        """Add Plotly chart to PDF story"""
        try:
            # Convert Plotly figure to image
            img_bytes = pio.to_image(chart_fig, format="png", width=800, height=500)
            
            # Create temporary file for the image
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes)
                tmp_file.flush()
                
                # Add chart title
                chart_title = ParagraphStyle(
                    'ChartTitle',
                    parent=self.subtitle_style,
                    fontSize=12,
                    spaceAfter=10,
                    textColor=colors.darkblue
                )
                story.append(Paragraph(self.clean_text(title), chart_title))
                
                # Add chart image
                img = ReportLabImage(tmp_file.name, width=6*inch, height=3.75*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                
                # Clean up temporary file
                import os
                os.unlink(tmp_file.name)
                
        except Exception as e:
            # Add error message if chart cannot be added
            error_text = f"Error adding chart to PDF: {str(e)}"
            story.append(Paragraph(self.clean_text(error_text), self.meta_style))
            story.append(Spacer(1, 10))
    
    def generate_chat_pdf_with_charts(self, chat_history, user_info=None):
        """Generate PDF from chat history including charts"""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.close()
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                temp_file.name,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=3*cm,
                bottomMargin=2*cm
            )
            
            # Build content
            story = []
            
            # Title
            title_text = "💬 Chat Conversation Export with Visualizations"
            story.append(Paragraph(self.clean_text(title_text), self.title_style))
            story.append(Spacer(1, 20))
            
            # User info and metadata
            if user_info:
                user_text = f"👤 User: {user_info.get('username', 'Unknown')} | Role: {user_info.get('role', 'User')}"
                story.append(Paragraph(self.clean_text(user_text), self.meta_style))
            
            export_time = f"📅 Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(self.clean_text(export_time), self.meta_style))
            story.append(Spacer(1, 20))
            
            # Warning about font support
            if not self.font_registered:
                warning_text = "⚠️ Thai font not available - some characters may not display correctly"
                warning_style = ParagraphStyle(
                    'Warning',
                    parent=self.meta_style,
                    textColor=colors.red,
                    fontSize=10
                )
                story.append(Paragraph(warning_text, warning_style))
                story.append(Spacer(1, 10))
            
            # Process chat history with chart support
            qa_pairs = self.extract_qa_pairs_with_charts(chat_history)
            
            if not qa_pairs:
                no_data_text = "📝 No conversation data available"
                story.append(Paragraph(self.clean_text(no_data_text), self.answer_style))
            else:
                # Add summary
                charts_count = sum(1 for qa in qa_pairs if qa.get('chart_data'))
                summary_text = f"📊 Total Conversations: {len(qa_pairs)} | Charts Generated: {charts_count}"
                story.append(Paragraph(self.clean_text(summary_text), self.subtitle_style))
                story.append(Spacer(1, 15))
                
                # Add Q&A pairs with charts
                for i, qa in enumerate(qa_pairs, 1):
                    # Question
                    question_text = f"❓ Q{i}: {qa.get('question', 'No question')}"
                    story.append(Paragraph(self.clean_text(question_text), self.question_style))
                    
                    # Answer
                    answer_text = f"💡 A{i}: {qa.get('response', 'No response')}"
                    story.append(Paragraph(self.clean_text(answer_text), self.answer_style))
                    
                    # Add chart if available
                    if qa.get('chart_data'):
                        try:
                            chart_config = qa['chart_data'].get('config', {})
                            chart_title = chart_config.get('title', f'Chart {i}')
                            
                            # Recreate chart from stored data
                            chart_json = qa['chart_data'].get('chart_json')
                            if chart_json:
                                chart_fig = pio.from_json(chart_json)
                                self.add_chart_to_story(story, chart_fig, chart_title)
                        except Exception as e:
                            error_text = f"Could not include chart: {str(e)}"
                            story.append(Paragraph(self.clean_text(error_text), self.meta_style))
                    
                    # Timestamp if available
                    if qa.get('timestamp'):
                        time_text = f"🕒 Time: {qa['timestamp']}"
                        story.append(Paragraph(self.clean_text(time_text), self.meta_style))
                    
                    # Add separator between Q&A pairs
                    if i < len(qa_pairs):
                        story.append(Spacer(1, 15))
                        # Add a subtle line separator
                        line_data = [[''] * 3]
                        line_table = Table(line_data, colWidths=[A4[0]-4*cm])
                        line_table.setStyle(TableStyle([
                            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.lightgrey),
                        ]))
                        story.append(line_table)
                        story.append(Spacer(1, 15))
            
            # Footer info
            story.append(Spacer(1, 30))
            footer_info = f"📋 Generated by Qdrant File Reader with CSV Visualization Support"
            story.append(Paragraph(self.clean_text(footer_info), self.meta_style))
            
            # Build PDF
            doc.build(story, onFirstPage=self.create_header_footer, onLaterPages=self.create_header_footer)
            
            # Read the generated PDF
            with open(temp_file.name, 'rb') as f:
                pdf_data = f.read()
            
            return pdf_data
            
        except Exception as e:
            raise Exception(f"Error generating PDF with charts: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def extract_qa_pairs_with_charts(self, chat_history):
        """Extract Q&A pairs from chat history including chart data"""
        qa_pairs = []
        current_question = None
        
        for chat in chat_history:
            if chat.get("type") == "user":
                current_question = {
                    "question": chat.get("message", ""),
                    "timestamp": chat.get("timestamp", ""),
                    "response": None,
                    "chart_data": None
                }
            elif chat.get("type") == "assistant" and current_question:
                # Handle different response formats
                message_content = chat.get("message", "")
                if isinstance(message_content, str):
                    try:
                        # Try to parse as JSON
                        data = json.loads(message_content)
                        response_text = (
                            data.get('response') or 
                            data.get('text') or 
                            data.get('answer') or
                            str(data)
                        )
                        # Check for chart data
                        chart_data = data.get('chart_data')
                        if chart_data:
                            current_question["chart_data"] = chart_data
                    except json.JSONDecodeError:
                        response_text = message_content
                else:
                    response_text = str(message_content)
                
                current_question["response"] = response_text
                qa_pairs.append(current_question)
                current_question = None
        
        return qa_pairs        
    
# Add CSV data preview functionality
def render_csv_data_preview():
    """Render CSV data preview for the current collection"""
    st.subheader("📊 CSV Data Preview")
    
    if not st.session_state.user_collections:
        st.info("No collections available")
        return
    
    selected_collection = st.selectbox(
        "Select Collection",
        st.session_state.user_collections,
        key="csv_preview_collection"
    )
    
    if st.button("📋 Preview CSV Data", key="preview_csv_data"):
        username = st.session_state.current_user.get("username") if st.session_state.current_user else None
        df = CSVVisualizationManager.extract_csv_data_from_collection(
            st.session_state.qdrant_client,
            selected_collection,
            username
        )
        
        if not df.empty:
            st.success(f"✅ Found {len(df)} CSV records")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_cols)
            
            # Show column info
            st.write("**Column Information:**")
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                col_info.append({
                    "Column": col,
                    "Type": dtype,
                    "Null Values": null_count,
                    "Unique Values": unique_count
                })
            
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
            
            # Show data preview
            st.write("**Data Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show data statistics for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.write("**Numeric Statistics:**")
                st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No CSV data found in the selected collection")

# Installation check function
def check_enhanced_requirements():
    """Check if all required packages for enhanced functionality are installed"""
    required_packages = {
        'plotly': 'plotly',
        'kaleido': 'kaleido',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        st.error("📦 Missing required packages for enhanced CSV visualization!")
        st.markdown(f"""
        **Please install the following packages:**
        ```bash
        pip install {' '.join(missing_packages)}
        ```
        
        **For complete functionality, install all at once:**
        ```bash
        pip install plotly kaleido pandas numpy reportlab
        ```
        """)
        return False
    
    return True

if __name__ == "__main__":
    main()