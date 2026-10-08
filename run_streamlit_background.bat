@echo off
cd /d "%~dp0"
"%USERPROFILE%\anaconda3\python.exe" -m streamlit run streamlit_app.py --server.port 8501 --server.headless true > streamlit_background.log 2> streamlit_background.err.log
