from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def format_timestamp_value(value):
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(str(value))
    except Exception:
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(IST)
    else:
        dt = dt.astimezone(IST)
    return dt.strftime("%b %d, %Y %I:%M %p IST")

def now_ist():
    return datetime.now(IST)
