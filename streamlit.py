import streamlit as st
import subprocess

def main():
    st.write("Running Bot...")
    try:
        subprocess.run(["python", "Bounty/app.py"])
    except FileNotFoundError:
        st.error("File 'Bounty/app.py' not found.")

if __name__ == "__main__":
    main()
