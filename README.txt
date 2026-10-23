Medical MCQ Bot
Files: bot.py, requirements.txt, Dockerfile, fly.toml (Fly.io), Procfile (Railway)
Secrets to set on the host (never put them in files): TELEGRAM_TOKEN, GEMINI_API_KEYS (comma separated)
Run locally: pip install -r requirements.txt ; create .env from .env.example ; python bot.py
