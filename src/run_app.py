import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)

    app_path = os.path.join(script_dir, "app_streamlit.py")  # Update this to run the desired app
    sys.argv = ["streamlit", "run", app_path]
    stcli.main()