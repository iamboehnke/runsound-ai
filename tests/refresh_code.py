import os, requests, base64
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = "http://127.0.0.1:8888/"
code = "AQClwE76oiclrFwWZf75MykupUTqo5tt5lleJh1Z_fZaxwtlk-wVmfS3VmLFd3UIhp6Tns1wmcp5_qRQU_b2UmcwuwkXGKHEYg24fvqnt3K_vzG8LgJ5GZ8-6Ye-Bw_USXaIYxdbH9BWnxoEaNHTZ7g7bJQ5n1NAt0F68tZIgoZTV5Mxxo_QAkmHfEZfmd8ow-Y7HUr5K-9yGWrs1GaOLBr4wvMwIpVanF3r2VTtzhxJ1PZEL0_0ybgpI-5I3eDj3Ck0WzxzJYdKfirH4g"

auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
r = requests.post(
    "https://accounts.spotify.com/api/token",
    headers={"Authorization": f"Basic {auth_header}"},
    data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
    },
    timeout=15,
)
print(r.status_code, r.text)
# Save r.json()["refresh_token"] into your .env as SPOTIPY_REFRESH_TOKEN