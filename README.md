# Thrive AI - Intelligent Data Analysis Platform

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.43+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Thrive AI is an intelligent data analysis platform that combines natural language processing with SQL generation, advanced analytics, and interactive visualizations. Built with Streamlit, it provides an intuitive chat interface for database interactions, comprehensive data analysis, and machine learning insights.

## 🚀 Features

### AI-Powered SQL Generation
- **Natural Language to SQL**: Convert plain English questions into SQL queries using Vanna AI
- **Multiple LLM Support**: Integration with Anthropic Claude, OpenAI, and local Ollama models
- **Smart Query Optimization**: Automatic query refinement and error handling

### Advanced Analytics & Visualizations
- **Magic Commands**: 20+ specialized commands for statistical analysis, data profiling, and ML
- **Interactive Charts**: Plotly-powered visualizations with statistical annotations
- **Machine Learning**: Built-in clustering, PCA, and outlier detection
- **Data Quality**: Comprehensive profiling, missing data analysis, and duplicate detection

### Security & Authentication
- **Secure Authentication**: Cookie-based session management with PBKDF2 password hashing
- **Ethical Guardrails**: Content filtering and safety measures
- **Role-Based Access**: Admin and Doctor user roles with appropriate permissions
- **Data Privacy**: Configurable data visibility controls

### Flexible Architecture
- **Multi-Database Support**: PostgreSQL and SQLite compatibility
- **Hybrid Deployment**: Cloud, hybrid, or fully local configurations
- **Vector Storage**: ChromaDB integration for embeddings and similarity search
- **Voice Interaction**: Speech-to-text and text-to-speech capabilities

## 📋 Prerequisites

- **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
- **Python 3.13** - Required for the application
- **UV Package Manager** - Fast Python package management

## 🛠️ Installation

### 1. Install UV Package Manager

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup Project

```bash
git clone <repository-url>
cd thrive_ui
uv sync
```

### 3. Configure Application

Copy the configuration template:
```bash
cp .streamlit/secrets_example.toml .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` with your configuration:

#### Cloud Configuration
```toml
[ai_keys]
vanna_api = "your_vanna_api_key"
vanna_model = "your_model_name"
# Optional: Add Anthropic for enhanced responses
anthropic_api = "your_anthropic_key"
anthropic_model = "claude-3-5-sonnet-latest"
```

#### Hybrid Configuration
```toml
[ai_keys]
anthropic_api = "your_anthropic_key"
anthropic_model = "claude-3-5-sonnet-latest"

[rag_model]
chroma_path = "./chromadb"
```

#### Local Configuration
```toml
[ai_keys]
ollama_model = "llama3.2"

[rag_model]
chroma_path = "./chromadb"
```

### 4. Start Database

```bash
docker compose up -d
```

### 5. Run Application

```bash
uv run streamlit run app.py
```

Access the application at `http://localhost:8501`

## 📊 Sample Data

The application includes sample datasets for immediate testing:
- **Penguins Dataset** - Species classification data
- **Titanic Dataset** - Historical passenger data
- **WNY Health Dataset** - Regional health statistics

## 🔧 Configuration Files

### Training Data (`utils/config/training_data.json`)
Configure custom training examples for your database schema. The application automatically reads DDL and populates schema information.

### Forbidden References (`utils/config/forbidden_references.json`)
Specify tables or columns to exclude from queries and training data for security compliance.

### Environment Variables
Configure application behavior using environment variables or `.streamlit/secrets.toml`:

- **`MAX_DISPLAY_ROWS`** - Maximum rows to display in DataFrames (default: 1000)
- **`MAX_SESSION_MESSAGES`** - Maximum messages to keep in session state for performance (default: 20)

## 🎯 Magic Commands

Thrive AI includes powerful magic commands for advanced analysis:

### Statistical Analysis
- `/describe <table>` - Comprehensive descriptive statistics
- `/distribution <table>.<column>` - Distribution analysis with tests
- `/correlation <table>.<column1>.<column2>` - Detailed correlation analysis
- `/outliers <table>.<column>` - Multi-method outlier detection

### Data Quality & Profiling
- `/profile <table>` - Comprehensive data profiling
- `/missing <table>` - Missing data analysis
- `/duplicates <table>` - Duplicate detection and analysis

### Visualizations
- `/boxplot <table>.<column>` - Statistical box plots
- `/heatmap <table>` - Correlation heatmaps
- `/wordcloud <table>` - Text visualization

### Machine Learning
- `/clusters <table>` - K-means clustering analysis
- `/pca <table>` - Principal Component Analysis

### Reporting
- `/report <table>` - Comprehensive analysis report
- `/summary <table>` - Executive summary

## 🏗️ Architecture

```
thrive_ui/
├── app.py                 # Main Streamlit application
├── views/                 # UI components
│   ├── chat_bot.py       # Chat interface
│   └── user.py           # User management
├── utils/                 # Core utilities
│   ├── vanna_calls.py    # AI/ML integration
│   ├── magic_functions.py # Advanced analytics
│   ├── auth.py           # Authentication
│   └── security.py       # Security measures
├── orm/                   # Database layer
│   ├── models.py         # SQLAlchemy models
│   └── functions.py      # Database operations
└── tests/                # Comprehensive test suite
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
uv run python -m pytest

# Specific test categories
uv run python -m pytest -m unit
uv run python -m pytest -m integration
uv run python -m pytest -m slow

# With coverage
uv run python -m pytest --cov=utils --cov-report=html

# Verbose debugging
uv run python -m pytest -vs
```

### Milvus tests

Milvus Lite hybrid search tests are marked `@pytest.mark.milvus` and require `pymilvus`.

- Run only Milvus tests:
```bash
uv run pytest -m milvus
```
- Skip Milvus tests:
```bash
uv run pytest -m "not milvus"
```

Comparison tests report top-1 accuracy and MRR@k; Milvus should meet or exceed Chroma on the sample corpus.

## 🔍 Code Quality

```bash
# Linting
uv run ruff check

# Formatting
uv run ruff format

# Test cleanup
python scripts/cleanup_test_artifacts.py
```

## 👥 Default Users

The application creates secure default accounts on first run:

| Username | Role | Password |
|----------|------|----------|
| thriveai-kr | Admin | password |
| thriveai-je | Admin | password |
| thriveai-as | Admin | password |
| thriveai-fm | Admin | password |
| thriveai-dr | Doctor | password |
| thriveai-re | Admin | password |

**⚠️ Security Note**: Change default passwords immediately after first login. All passwords use PBKDF2 hashing with 100,000 iterations.

## 🆘 Support

For issues and questions:
- Check the [troubleshooting guide](docs/TROUBLESHOOTING.md)
- Review test examples in the `tests/` directory
- Open an issue on GitHub

## 🔗 Dependencies

Key technologies powering Thrive AI:
- **Streamlit** - Web application framework
- **Vanna AI** - SQL generation and NLP
- **Anthropic Claude** - Advanced language model
- **ChromaDB** - Vector database for embeddings
- **PostgreSQL** - Primary database
- **Plotly** - Interactive visualizations
- **scikit-learn** - Machine learning algorithms