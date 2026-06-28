// Serverless proxy to LTA DataMall Bus Arrival API.
// The AccountKey is read from an environment variable, or from a local
// git-ignored file (api/_key.js) so the key never reaches the public repo.
let INLINE_KEY = "";
try { INLINE_KEY = require("./_key").KEY; } catch (e) { /* no local key file */ }
const KEY = process.env.LTA_ACCOUNT_KEY || INLINE_KEY;

export default async function handler(req, res) {
  const stop = String(req.query.stop || "").replace(/[^0-9]/g, "");
  if (!stop) {
    res.status(400).send("Missing ?stop=<bus stop code>");
    return;
  }
  if (!KEY) {
    res.status(500).send("Server is missing the LTA API key.");
    return;
  }
  try {
    const url =
      "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival?BusStopCode=" +
      stop;
    const r = await fetch(url, {
      headers: { AccountKey: KEY, accept: "application/json" },
    });
    if (!r.ok) {
      res.status(502).send("LTA API error " + r.status);
      return;
    }
    const data = await r.json();
    res.setHeader("Cache-Control", "s-maxage=15, stale-while-revalidate=30");
    res.status(200).json(data);
  } catch (e) {
    res.status(500).send("Fetch failed: " + e.message);
  }
}
