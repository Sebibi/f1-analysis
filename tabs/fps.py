import copy
from typing import Iterable
from fastf1.core import Laps, Lap, Telemetry, Session
import fastf1
import numpy as np
import pandas as pd
import streamlit as st
import fastf1 as ff1
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def mix_colors(color1, color2):
    rgb1 = colors.to_rgb(color1)
    rgb2 = colors.to_rgb(color2)
    blended_rgb = tuple((c1 + c2) / 2 for c1, c2 in zip(rgb1, rgb2))
    blended_hex = colors.to_hex(blended_rgb)
    return blended_hex


def create_driver_compounds_laps(session: Session, drivers: list[str], quick_laps_threshold: float = 1.10):
    laps = session.laps

    # Split by driver
    driver_laps = {driver: laps.pick_driver(driver).pick_wo_box().pick_quicklaps(quick_laps_threshold) for driver in
                   drivers}

    # Split by compound
    driver_compounds = {driver: driver_laps[driver]['Compound'].unique() for driver in drivers}
    driver_compound_laps = {
        driver: {compound: driver_laps[driver].pick_tyre(compound) for compound in driver_compounds[driver]} for driver
        in drivers}

    # Split by stint
    driver_compound_stints = {
        driver: {compound: driver_compound_laps[driver][compound]['Stint'].unique() for compound in
                 driver_compounds[driver]} for driver in drivers}

    driver_compound_stint_laps = {driver: {compound: {
        stint: driver_compound_laps[driver][compound][driver_compound_laps[driver][compound]['Stint'] == stint] for
        stint in driver_compound_stints[driver][compound]} for compound in driver_compounds[driver]} for driver in
        drivers}
    return driver_compound_stint_laps


def plot_driver_stint_laps(name: str, driver_compound_stint_laps: dict, stint_deg: dict = None):
    fig = plt.figure(figsize=(10, 4))
    plt.title(name)
    for driver in driver_compound_stint_laps.keys():
        driver_color = fastf1.plotting.DRIVER_COLORS[fastf1.plotting.DRIVER_TRANSLATE[driver]]
        for compound in driver_compound_stint_laps[driver].keys():
            compound_color = fastf1.plotting.COMPOUND_COLORS[compound]
            for stint in driver_compound_stint_laps[driver][compound].keys():
                laps = driver_compound_stint_laps[driver][compound][stint]
                color = mix_colors(compound_color, driver_color)
                slope = ""
                if stint_deg is not None and name == "R":
                    deg = stint_deg[driver][compound][stint]
                    slope = f"{np.around(deg[0] * 100, 2)}%"
                    x = np.linspace(laps['LapNumber'].min(), laps['LapNumber'].max(), 100)
                    y = np.polyval(deg, x)
                    plt.plot(x, y, color=color)
                plt.plot(laps['LapNumber'], laps['LapTime'], label=f"{driver} {int(stint)} {compound} {slope}",
                         marker='o', color=color)
    plt.legend()
    with st.expander(label="Lap times" + name):
        st.pyplot(fig)


def create_telemetry(lap1: Lap, lap2: Lap) -> tuple[Telemetry, Telemetry]:
    driver1_telemetry = lap1.get_car_data().add_distance()
    driver2_telemetry = lap2.get_car_data().add_distance()

    driver1_telemetry['Time'] = driver1_telemetry['Time'].dt.total_seconds()
    driver2_telemetry['Time'] = driver2_telemetry['Time'].dt.total_seconds()

    delta_distance = driver1_telemetry['Distance'] - driver2_telemetry['Distance']
    delta_time = driver2_telemetry['Time'] - driver1_telemetry['Time']

    driver1_telemetry['Delta_Distance'] = -1 * delta_distance
    driver2_telemetry['Delta_Distance'] = delta_distance
    driver1_telemetry['Delta_Time'] = delta_time
    driver2_telemetry['Delta_Time'] = -1 * delta_time
    driver1_telemetry['Power'] = driver1_telemetry['RPM'] * driver1_telemetry['nGear']
    driver2_telemetry['Power'] = driver2_telemetry['RPM'] * driver2_telemetry['nGear']
    driver1_telemetry['Brake'] = driver1_telemetry['Brake'].astype(float)
    driver2_telemetry['Brake'] = driver2_telemetry['Brake'].astype(float)
    return driver1_telemetry, driver2_telemetry


def remove_outliers(laps: Laps, threshold: float = 3.5) -> Laps:
    if len(laps) == 1:
        return laps
    lap_times = laps['LapTime'].to_numpy()
    z_scores = (lap_times - np.mean(lap_times)) / np.std(lap_times)
    mask = np.vectorize(lambda x: x < threshold)(z_scores)
    return laps[mask]


def build_fps():
    st.header("FP ANALYSIS")
    circuit_options = ['Monza', 'Zandvoort', 'Spain', 'Canada', 'Austria', 'Bahrain', 'Australia', 'Sahkir']
    circuit_name = st.selectbox(
        label="Select circuit",
        options=circuit_options,
        index=0,
        on_change=lambda: st.session_state.pop('fps', None)
    )

    fps_options = ['FP1', 'FP2', 'FP3', 'Q', 'R']
    fps_name = st.multiselect(
        label="Select FP",
        options=fps_options,
        default=["FP1", "FP2"],
        on_change=lambda: st.session_state.pop('fps', None)
    )

    quick_laps_threshold = st.number_input(
        label="Quick laps threshold",
        min_value=1.0,
        max_value=2.0,
        value=1.10,
    )

    if 'fps' not in st.session_state:
        with st.spinner("Loading session data..."):
            fps = {}
            for name in fps_name:
                fp = ff1.get_session(2023, circuit_name, name)
                fp.load(laps=True, telemetry=True, weather=False, messages=False)
                fp.laps['LapTime'] = fp.laps['LapTime'].dt.total_seconds()
                fps[name] = fp
            st.session_state['fps'] = fps

    fps = copy.deepcopy(st.session_state['fps'])
    driver_options = fps[fps_name[0]].laps['Driver'].unique()
    drivers = st.multiselect(label="Select drivers", options=driver_options, default=["VER", "HAM"])

    if 'Q' in fps_name:
        tyre_adjustment = st.checkbox(label="Tyre quali adjustment", value=False)
        if tyre_adjustment:
            qualif = fps['Q']
            qs: Iterable[fastf1.core.Laps] = qualif.laps.split_qualifying_sessions()
            qs = [q.pick_wo_box().pick_quicklaps(quick_laps_threshold) for q in qs]
            q3_drivers = qs[-1]['Driver'].unique()
            qs = [q.pick_drivers(q3_drivers) for q in qs]
            time_qs = [laps.groupby('Driver').agg('LapTime').min().mean() for laps in qs]
            time_delta = [t - time_qs[-1] for t in time_qs]
            tyre_adjustment = {"HARD": time_delta[0], "MEDIUM": time_delta[1], "SOFT": time_delta[2]}
            st.warning(f"Qualifying sessions: {tyre_adjustment}")
            st.session_state['tyre_adjustment'] = tyre_adjustment
            qualif.laps['LapTime'] -= qualif.laps['Compound'].map(tyre_adjustment)

    if 'R' in fps_name:
        fuel_adjustment = st.checkbox(label="Fuel race adjustment", value=False)
        tyre_adjustment = st.checkbox(label="Tyre race adjustment", value=False)
        if fuel_adjustment:
            race = fps['R']
            total_fuel = 110
            total_laps = race.total_laps
            fuel_per_lap = total_fuel / total_laps
            fuel_time_per_lap = fuel_per_lap * 0.2 / 10
            race.laps['LapTime'] -= (total_laps - race.laps['LapNumber']) * fuel_time_per_lap

        if tyre_adjustment and "tyre_adjustment" in st.session_state:
            race = fps['R']
            race.laps['LapTime'] -= race.laps['Compound'].map(st.session_state['tyre_adjustment'])

    stints = {name: create_driver_compounds_laps(fps[name], drivers, quick_laps_threshold) for name in fps_name}
    tire_deg = st.checkbox(label="Tire deg analysis", value=False)

    stints_deg = {}
    outlier_threshold = st.number_input(label="Outlier threshold", min_value=-5.0, max_value=5.0, value=1.5)
    if tire_deg:  # Perform linear regression on the stint lap times
        for name in fps_name:
            stints_deg[name] = {}
            for driver in stints[name].keys():
                stints_deg[name][driver] = {}
                for compound in stints[name][driver].keys():
                    stints_deg[name][driver][compound] = {}
                    for stint, laps in stints[name][driver][compound].items():
                        fastest_lap_number = laps.pick_quicklaps(1.0001)['LapNumber'].iloc[0]
                        laps = laps[laps['LapNumber'] >= fastest_lap_number]
                        laps = remove_outliers(laps, threshold=outlier_threshold)
                        x = laps['LapNumber'].to_numpy()
                        y = laps['LapTime'].to_numpy()
                        stints_deg[name][driver][compound][stint] = np.polyfit(x, y, 1)

    for name, stint in stints.items():
        plot_driver_stint_laps(name, stint, stints_deg.get(name, None))

    # Compare 2 laps
    fp = st.selectbox(label="Select FP to compare", options=fps_name)
    laps = fps[fp].laps
    drivers = laps['Driver'].unique()

    cols = st.columns(2)
    with cols[0]:
        driver1 = st.selectbox(label="Select driver 1", options=drivers)
        driver1_laps = laps.pick_driver(driver1)
        lap1 = st.selectbox(label="Select lap 1", options=driver1_laps['LapNumber'])

    with cols[1]:
        driver2 = st.selectbox(label="Select driver 2", options=drivers)
        driver2_laps = laps.pick_driver(driver2)
        lap2 = st.selectbox(label="Select lap 2", options=driver2_laps['LapNumber'])

    driver1_lap = driver1_laps.pick_lap(lap1)
    driver2_lap = driver2_laps.pick_lap(lap2)

    driver1_telemetry, driver2_telemetry = create_telemetry(driver1_lap, driver2_lap)

    metrics_options = driver1_telemetry.columns.tolist()
    metrics = st.multiselect(label="Select metrics", options=metrics_options, default=["Speed", "Throttle"],
                             key='metrics2')

    index = st.selectbox(label="Select index", options=metrics_options, index=len(metrics_options) - 4)

    fig, ax = plt.subplots(len(metrics), figsize=(20, 10))
    fig.suptitle(f"{driver1} {driver1_lap['LapTime'].iloc[0]} vs {driver2} - {driver2_lap['LapTime'].iloc[0]} - {fp}")

    for i in range(len(metrics)):
        ax[i].set_title(f"{metrics[i]}")
        driver1_telemetry.plot(x=index, y=metrics[i], ax=ax[i], label=driver1, color='red')
        driver2_telemetry.plot(x=index, y=metrics[i], ax=ax[i], label=driver2, color='green')
        ax[i].legend()

    for a in ax.flat:
        a.label_outer()

    st.pyplot(fig)
