import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL") or ""
key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or ""
supabase: Client = create_client(url, key)