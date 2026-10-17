# MT5 bar export for the ORB test

Put the CSV files exported from your MetaTrader 5 **demo** terminal here, on branch `claude/new-project-folder-vjimke`.

**Easiest way, in the browser:** open
https://github.com/TnkT59/test/tree/claude/new-project-folder-vjimke/fusion-strategy/data/mt5
and use **Add file → Upload files**. Drag the CSVs in, keep "Commit directly to the `claude/new-project-folder-vjimke` branch" selected, and click **Commit changes**. The browser accepts files up to 25 MB each.

## How to export (free, any MT5 broker)

1. In MT5 open **View → Symbols** (Ctrl+U), then the **Bars** tab.
2. Pick the symbol (XAUUSD may be called GOLD at your broker; suffixes like `.m` are fine), timeframe **M15** (M1 or M5 also work; use the same one for all four symbols), and dates **2023-01-01 → today**. Click **Request**.
   - If your broker has less M1 history than that, note the first date shown. The test runs on whatever is there.
   - If fewer bars appear than expected, try setting **Tools → Options → Charts → Max bars in chart** to Unlimited and requesting again. I haven't verified that this setting affects the Bars tab.
3. Click **Export Bars** and save as `SYMBOL_YEAR.csv`. Use one file per symbol per year (for example `EURUSD_2023.csv`) if you want. With M15, one file per symbol (for example `EURUSD.csv`) is fine, because it is only a few MB.
4. Repeat for XAUUSD, EURUSD, GBPUSD and USDJPY.

The files must contain the `<SPREAD>` column, which the default export includes. Leave the format as MT5 writes it (UTF-16, tab-separated is fine).

## Also tell me (if you know)

- Your broker's **server timezone**, shown as the time in Market Watch versus your local time. Most brokers use GMT+2 in winter and GMT+3 in summer. The script detects this from the FX weekly open and prints what it found.
- Whether the account's commission matches the plan ($7 per round-turn lot). The test uses the plan's costs either way.

## Run it yourself (optional)

```
python -m venv .venv && .venv\\Scripts\\pip install pandas numpy pyarrow matplotlib
cd research
..\\.venv\\Scripts\\python orb_sessions.py --data ..\\data\\mt5 --out ..\\results\\orb_sessions
```
