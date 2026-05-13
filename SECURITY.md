# Security Notice

## API Key Security

This project requires a Materials Project API key to access the Materials Project database.

### ⚠️ IMPORTANT: Do NOT commit API keys to Git

Never commit your actual API key to version control. The repository includes security measures to prevent this:

1. The `.env` file is in `.gitignore` and will not be committed
2. A `.env.example` file is provided as a template
3. The `config.py` reads from environment variables first

### Setting Your API Key (Secure Methods)

#### Method 1: Environment Variable (Recommended)
```bash
export MP_API_KEY="your_api_key_here"
```

#### Method 2: .env File
1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
2. Edit `.env` and replace `your_api_key_here` with your actual key

#### Method 3: Direct Edit (Less Secure)
Edit `config.py` and replace the default value, but **do not commit this change**.

### Getting Your API Key

Get your free API key from: https://materialsproject.org/open

### In Google Colab

Use the provided `@param` cell in the notebook to enter your key securely:
```python
MP_API_KEY = "YOUR_KEY_HERE"  # @param {type:"string"}
import os
os.environ["MP_API_KEY"] = MP_API_KEY
```

## Data Privacy

This project processes materials data from the Materials Project database. No personal or sensitive data is collected or stored by this application.

## Reporting Security Issues

If you discover any security vulnerabilities, please report them responsibly.
