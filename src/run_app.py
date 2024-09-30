import streamlit
import sys
import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    app_path = os.path.join(script_dir, "app_streamlit.py")
    sys.argv = ["streamlit", "run", app_path]
    streamlit.cli.main()