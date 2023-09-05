import streamlit as st
import fastf1 as ff1
from fastf1 import plotting
from tabs import tabs_options
st.set_page_config(layout="wide")


# Enable cache and setup plotting
plotting.setup_mpl()
ff1.Cache.enable_cache('cache')


tabs = st.tabs(tabs_options.keys())

for tab, func in zip(tabs, tabs_options.values()):
    with tab:
        func()