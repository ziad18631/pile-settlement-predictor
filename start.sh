#!/bin/bash
# Railway deployment start script
streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
