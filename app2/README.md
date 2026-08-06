# Jet Engine Hospital — Streamlit

This directory contains the Streamlit Community Cloud version of the Jet Engine
Hospital dashboard. It uses the bundled, precomputed NASA C-MAPSS inference
artifacts, so deployment does not require model training, external services, or
secrets.

## Run locally

Use Python 3.12, then install and start the app:

```bash
python -m pip install -r app2/requirements.txt
streamlit run app2/app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at <https://share.streamlit.io> and choose **Create app**.
3. Select the repository and branch.
4. Set **Main file path** to `app2/app.py`.
5. In **Advanced settings**, select Python 3.12. No secrets are required.
6. Select **Deploy**.

`app2/requirements.txt` is intentionally next to the entrypoint so Community
Cloud discovers the correct dependencies when this repository contains multiple
applications.

## Included functionality

- FD001, FD003, and FD004 dataset selection
- Engine, cycle, and sensor exploration
- Sensor and remaining-useful-life timelines with a 90% interval
- Calibrated 10/20/30-cycle failure risks
- Normalized anomaly evidence and evidence-family agreement
- CONTINUE, INSPECT, or STOP maintenance recommendations

