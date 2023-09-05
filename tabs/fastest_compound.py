import fastf1.core
import pandas as pd
import numpy as np
import fastf1 as ff1
from fastf1 import plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import streamlit as st


def build_fastest_compound():
    st.header("F1 Telemetry")

    # Load the session
    if "fp1" not in st.session_state:
        with st.spinner("Loading session data..."):
            race = ff1.get_session(2023, 'Zandvoort', 'R')
            race.load(laps=True, telemetry=True, weather=False, messages=False)
            st.session_state['fp1'] = race
            st.success("Loaded session data")

    with st.spinner("Processing data..."):
        race = st.session_state['fp1']
        lap_number = st.selectbox(label="Select lap", options=list(range(race.total_laps)))
        laps = race.laps.pick_lap(lap_number)
        # drivers = pd.unique(laps['Driver'])
        compounds_options = pd.unique(laps['Compound'])
        compounds = st.multiselect(label="Select compounds", options=compounds_options, default=compounds_options)
        # driver_laps: dict[str, fastf1.core.Lap] = {driver: laps.pick_driver(driver) for driver in drivers}
        compound_laps: dict[str, fastf1.core.Lap] = {compound: laps.pick_tyre(compound) for compound in compounds}

        fastest_compound_laps = {compound: laps.pick_fastest(only_by_time=True) for compound, laps in
                                 compound_laps.items()}
        compound_telemetry = {compound: lap.get_car_data().add_distance() for compound, lap in
                              fastest_compound_laps.items()}

        for compound, telemetry in compound_telemetry.items():
            telemetry['Driver'] = fastest_compound_laps[compound]['Driver']

        for compound, telemetry in compound_telemetry.items():
            telemetry['Brake'] = telemetry['Brake'].astype(float)

        metrics_options = ['Speed', 'Throttle', 'Brake', 'nGear']
        metrics = st.multiselect(label="Select metrics", options=metrics_options, default=metrics_options)
        fig, ax = plt.subplots(len(metrics), figsize=(20, 10))

        title = "Compound Telemetry - "
        for compound in compounds:
            title += f"{fastest_compound_laps[compound]['Driver']} - "
        fig.suptitle(title)

        for i in range(len(metrics)):
            ax[i].set_title(f"{metrics[i]}")
            for compound in compounds:
                compound_telemetry[compound].plot(x='Distance', y=metrics[i], ax=ax[i], label=compound)
            ax[i].legend()

        for a in ax.flat:
            a.label_outer()

        st.pyplot(fig)