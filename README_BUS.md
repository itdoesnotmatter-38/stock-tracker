# 🚌 Singapore Bus Arrival App

A simple live bus-arrival tracker for Singapore. Save the bus stops and bus
numbers you care about, and see real-time "minutes to arrival" — including how
crowded the bus is (🟢/🟡/🔴) and whether it's wheelchair accessible (♿).

Powered by the free **LTA DataMall** Bus Arrival API.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run bus_app.py
```

## Get a free API key (one-time)

1. Go to **https://datamall.lta.gov.sg** → *Request API Access*.
2. Fill in the short form — you'll receive an **AccountKey** by email instantly.
3. Either paste the key into the sidebar when the app opens, **or** save it so
   you never re-enter it by creating `.streamlit/secrets.toml`:

   ```toml
   LTA_ACCOUNT_KEY = "your-account-key-here"
   ```

## How to use

- In the sidebar, **Add a favourite**: give it a label (e.g. *Home*), the
  5-digit **bus stop code** (printed on every bus-stop pole), and optionally the
  **bus numbers** you care about (e.g. `61, 154`). Leave bus numbers blank to
  show every bus at that stop.
- Each saved stop shows the next 3 arrivals per bus.
- Tap **🔄 Refresh now**, or tick **Auto-refresh every 20s** for hands-free live updates.

## Share it with your family 👨‍👩‍👧‍👦

The easiest way to give your family a link they can open on any phone is
**Streamlit Community Cloud** (free):

1. Push this repo to GitHub (this branch already has the app).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. **Create app** → pick this repo/branch → set the main file to `bus_app.py`.
4. Under **Advanced settings → Secrets**, paste your key so nobody has to type it:

   ```toml
   LTA_ACCOUNT_KEY = "your-account-key-here"
   ```

5. Deploy. You'll get a public URL like `https://your-app.streamlit.app` —
   send that to your family and they can open it straight from their phone's
   browser (and "Add to Home Screen" for an app-like icon).

> **Note:** favourite stops are stored in `bus_favourites.json` on the server.
> On Streamlit Cloud this resets if the app restarts, so to ship a permanent
> default set of family stops, edit/commit `bus_favourites.json` in the repo.
