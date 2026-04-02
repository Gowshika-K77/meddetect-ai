timeout = 120
workers = 1
threads = 1
```

---

### Step 2 — Go to Render dashboard

1. Open **render.com**
2. Click your **meddetect-ai** service
3. Click **Settings**
4. Find **Start Command**
5. Change it to:
```
gunicorn app:app --config gunicorn.conf.py