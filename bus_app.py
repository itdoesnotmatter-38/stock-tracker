import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone
import json
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SG Bus Arrival", page_icon="🚌")

FAV_FILE = "bus_favourites.json"
LTA_URL = "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival"

# Human-readable lookups for LTA bus-arrival codes
LOAD_MAP = {
    "SEA": ("🟢", "Seats available"),
    "SDA": ("🟡", "Standing only"),
    "LSD": ("🔴", "Limited standing"),
}
TYPE_MAP = {
    "SD": "Single deck",
    "DD": "Double deck",
    "BD": "Bendy",
}


# --- API KEY ---
def get_api_key():
    """Read the LTA DataMall AccountKey from secrets, env, or the sidebar."""
    key = ""
    try:
        key = st.secrets.get("LTA_ACCOUNT_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("LTA_ACCOUNT_KEY", "")
    return key


# --- FAVOURITES STORAGE ---
def load_favs():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_favs(favs):
    with open(FAV_FILE, "w") as f:
        json.dump(favs, f, indent=2)


# --- DATA FETCH ---
@st.cache_data(ttl=20, show_spinner=False)
def get_arrivals(stop_code, key):
    """Call the LTA Bus Arrival API for a single bus stop."""
    headers = {"AccountKey": key, "accept": "application/json"}
    params = {"BusStopCode": str(stop_code).strip()}
    resp = requests.get(LTA_URL, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def minutes_until(iso_ts):
    """Return whole minutes from now until an ISO-8601 arrival timestamp."""
    if not iso_ts:
        return None
    try:
        arrival = datetime.fromisoformat(iso_ts)
        if arrival.tzinfo is None:
            arrival = arrival.replace(tzinfo=timezone.utc)
        delta = (arrival - datetime.now(timezone.utc)).total_seconds() / 60
        return int(round(delta))
    except Exception:
        return None


def fmt_eta(mins):
    if mins is None:
        return "—"
    if mins <= 0:
        return "Arr"
    return f"{mins} min"


def bus_row(service):
    """Flatten one LTA 'Services' entry into a display dict."""
    no = service.get("ServiceNo", "?")
    etas, loads, feats, types = [], [], [], []
    for slot in ("NextBus", "NextBus2", "NextBus3"):
        nb = service.get(slot, {}) or {}
        etas.append(fmt_eta(minutes_until(nb.get("EstimatedArrival"))))
        load_icon = LOAD_MAP.get(nb.get("Load", ""), ("", ""))[0]
        loads.append(load_icon)
        feats.append("♿" if nb.get("Feature") == "WAB" else "")
        types.append(TYPE_MAP.get(nb.get("Type", ""), ""))
    first = service.get("NextBus", {}) or {}
    return {
        "Bus": no,
        "Next": etas[0],
        "After": etas[1],
        "3rd": etas[2],
        "Load": LOAD_MAP.get(first.get("Load", ""), ("", "—"))[1],
        "Type": TYPE_MAP.get(first.get("Type", ""), "—"),
        "♿": feats[0],
    }


try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# --- MAIN ---
st.title("🚌 Singapore Bus Arrival")

api_key = get_api_key()

with st.sidebar:
    st.header("⚙️ Setup")

    if not api_key:
        st.warning("Enter your LTA DataMall API key to start.")
        api_key = st.text_input("LTA AccountKey", type="password")
        with st.expander("How to get a free key"):
            st.markdown(
                "1. Register at **datamall.lta.gov.sg** → *Request API Access*.\n"
                "2. You'll receive an **AccountKey** by email (free, instant).\n"
                "3. Paste it above, or add it to `.streamlit/secrets.toml` as "
                "`LTA_ACCOUNT_KEY = \"...\"` so you don't re-enter it."
            )
    else:
        st.success("API key loaded ✅")

    st.divider()
    st.subheader("⭐ Add a favourite")
    with st.form("add_fav", clear_on_submit=True):
        label = st.text_input("Label (e.g. Home, Office)")
        stop_code = st.text_input("Bus stop code (5 digits)")
        services = st.text_input("Bus numbers (comma separated, blank = all)")
        if st.form_submit_button("Save", use_container_width=True):
            if stop_code.strip():
                favs = load_favs()
                favs.append({
                    "label": label.strip() or f"Stop {stop_code.strip()}",
                    "stop_code": stop_code.strip(),
                    "services": [s.strip() for s in services.split(",") if s.strip()],
                })
                save_favs(favs)
                st.success("Saved!")
                st.rerun()
            else:
                st.error("Bus stop code is required.")

    st.divider()
    if st.button("🔄 Refresh now", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    auto = st.checkbox("Auto-refresh every 20s", value=False)

# Auto-refresh (uses the streamlit-autorefresh component when installed)
if auto:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=20000, key="bus_autorefresh")
        st.cache_data.clear()
    else:
        st.caption("⏱️ Install `streamlit-autorefresh` to enable live auto-refresh.")

favs = load_favs()

if not api_key:
    st.info("👈 Add your LTA DataMall API key in the sidebar to see live arrivals.")
    st.stop()

if not favs:
    st.info("👈 Add a favourite bus stop in the sidebar. "
            "Tip: the 5-digit stop code is printed on every bus-stop pole.")
    st.stop()

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} · "
           "🟢 seats · 🟡 standing · 🔴 packed · ♿ wheelchair accessible")


def render_stop(fav):
    st.subheader(f"📍 {fav['label']}  ·  Stop {fav['stop_code']}")
    try:
        data = get_arrivals(fav["stop_code"], api_key)
    except requests.HTTPError as e:
        st.error(f"LTA API error ({e.response.status_code}). Check your API key / stop code.")
        return
    except Exception as e:
        st.error(f"Could not fetch arrivals: {e}")
        return

    all_services = data.get("Services", [])
    wanted = set(fav.get("services", []))
    if wanted:
        all_services = [s for s in all_services if s.get("ServiceNo") in wanted]
        missing = wanted - {s.get("ServiceNo") for s in all_services}
        if missing:
            st.caption(f"No live data right now for: {', '.join(sorted(missing))}")

    if not all_services:
        st.warning("No buses currently in service for this stop.")
        return

    rows = [bus_row(s) for s in all_services]
    rows.sort(key=lambda r: int(r["Bus"]) if r["Bus"].isdigit() else 9999)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# Render each favourite, with a delete control
for i, fav in enumerate(favs):
    cols = st.columns([10, 1])
    with cols[0]:
        render_stop(fav)
    with cols[1]:
        if st.button("🗑️", key=f"del_{i}", help="Remove this stop"):
            favs.pop(i)
            save_favs(favs)
            st.rerun()
    st.divider()
