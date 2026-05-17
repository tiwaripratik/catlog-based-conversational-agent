"""
Render Entrypoint — Instant Startup & Lazy-Loading ASGI Wrapper
==============================================================
Exposes a lightweight ASGI interface:
  - GET /health           → Responds in < 1ms with {"status": "ok"}
  - lifespan              → Completes startup instantly, skipping heavy imports
  - Other (e.g. /chat)    → Lazy-loads `app.main:app` on the first request and delegates
"""

import os
import sys

# Ensure current directory is in the Python search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file manually if present (zero-dependency env loading for local testing)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                try:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
                except Exception as e:
                    print(f"[LazyLoad] Warning: Failed to parse line in .env: '{line}' - {e}")

_actual_app = None

async def app(scope, receive, send):
    global _actual_app
    
    if scope["type"] == "lifespan":
        # Handle lifespan events for startup instantly to make Render's health checks pass immediately
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if _actual_app is not None:
                    try:
                        from app.retrieval import get_retriever
                        get_retriever().close()
                    except Exception as e:
                        print(f"[LazyLoad] Error closing retriever: {e}")
                await send({"type": "lifespan.shutdown.complete"})
                break
                
    elif scope["type"] == "http" and scope["path"] == "/health":
        # Instantly handle the health check without triggering any heavy imports or database connections
        headers = [(b"content-type", b"application/json")]
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": headers,
        })
        await send({
            "type": "http.response.body",
            "body": b'{"status": "ok"}',
        })
        
    else:
        # Lazy load the actual app on the first request to /chat or other non-health endpoints
        if _actual_app is None:
            print("[LazyLoad] Request received to non-health endpoint. Importing actual app...")
            try:
                from app.main import app as actual_app
                _actual_app = actual_app
                print("[LazyLoad] Actual app loaded successfully!")
            except Exception as e:
                import traceback
                print(f"[LazyLoad] ERROR loading actual app:\n{traceback.format_exc()}")
                # Return a graceful 500 error if import fails
                headers = [(b"content-type", b"application/json")]
                await send({
                    "type": "http.response.start",
                    "status": 500,
                    "headers": headers,
                })
                await send({
                    "type": "http.response.body",
                    "body": f'{{"error": "Failed to load application: {str(e)}"}}'.encode('utf-8'),
                })
                return
                
        # Delegate the request to the actual FastAPI app
        await _actual_app(scope, receive, send)
