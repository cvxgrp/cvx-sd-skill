---
name: trend-seasonal-confound-record-length
description: Gotcha — a trend and a low-frequency periodic term fight over the same signal; over short records they trade off arbitrarily. Plus the corrected record-length rule for when to include yearly-periodic vs. trend.
metadata:
  type: reference
---

**Trend ↔ low-frequency-periodic confound (gotcha).**
A trend and a yearly seasonal term fight over the same low-frequency signal;
over a short record they trade off arbitrarily — half a sine over one year looks
like a trend, so a smooth trend can eat a large chunk of the seasonal swing and
the decomposition splits seasonality between the two in a non-identifiable way.
Same *family* as the DC-offset collision the periodic component already resolves
by construction (dropping the DC column so the constant offset lives in the
trend intercept) — just at the seasonal frequency instead of at zero. Footgun:
don't put a trend and a yearly seasonal term in the same short-record model
without expecting them to trade off.
→ `gotchas.md` + `periodic-and-time.md`.

**Corrected record-length rule (seasonal vs. trend):**
- **< ~1 year:** cannot fit a yearly cycle (fitting a full-period harmonic to a
  fraction of one cycle is overfitting) → drop the yearly periodic term, use a
  slow smooth trend to absorb the partial seasonal drift you can see.
- **1–2 years:** **keep the yearly periodic term, drop the trend.** You have
  enough to identify the seasonal shape, and a trend would confound with it;
  there also isn't enough baseline to justify a separate long-term trend. (This
  corrected an earlier in-conversation answer that said drop-yearly/use-trend at
  this range — wrong; keep-yearly/drop-trend is right.)
- **2+ years:** afford both — a yearly periodic term *and* a slow trend separate
  cleanly, because there is enough data to distinguish "the shape that repeats
  each year" from "the level drifting across years."
→ `periodic-and-time.md`, illustrated via the hourly-load example
([[hourly-load-worked-example]]).
