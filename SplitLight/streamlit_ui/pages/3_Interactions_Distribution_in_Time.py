"""
Page with time distribution of interactions.
"""

import warnings

import pandas as pd
import streamlit as st

from src.stats.plots import plot_inters_px
from src.stats.temporal import inters_per_period
from streamlit_ui.utils import load_data, page_config


warnings.filterwarnings('ignore')
page_config()
pd.set_option('display.max_columns', None)

if not st.session_state.data_loaded:
    st.warning("Please load data to access this page")
    st.stop()


def main():

    st.title('Interactions Distribution Over Time')
    st.markdown(f'**Dataset:** {st.session_state.selected_dataset}' + (f' | **Split:** {st.session_state.selected_split}' if st.session_state.selected_split else ''))

    st.write("Use this page to to access the interactions intensity and uniformity over time for selected subsets. \
    Plot multiple subsets to sanity-check the splitting results."                                                                                                                                                                                                                                                                                                                                                                                                      )

    st.markdown("##### Select Subsets")
    available_subsets = list(st.session_state.df_paths.keys())
    selected_subsets = st.multiselect(
        "Select subsets:",
        available_subsets,
        default=st.session_state.default_subset,
        key="target",
        label_visibility="collapsed",
        )

    if not selected_subsets:
        st.warning("Please select at least one subset to analyze.")
        st.stop()

    selected_df = {subset: load_data(subset) for subset in selected_subsets}

    for name, df in selected_df.items():
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        selected_df[name] = df

    min_date = min(df['timestamp'].min() for df in selected_df.values()).date()
    max_date = max(df['timestamp'].max() for df in selected_df.values()).date()

    st.markdown("##### Select Date Range and Time Granularity")

    st.write(f"**Data available from:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date",
                                 value=min_date,
                                 min_value=min_date,
                                 max_value=max_date)
    with col2:
        end_date = st.date_input("End Date",
                               value=max_date,
                               min_value=min_date,
                               max_value=max_date)

    frequency_map = {
        'Minute': 'min',
        'Hour': 'H',
        'Day': 'D',
        'Week': 'W',
        'Month': 'M'
    }
    granularity = st.selectbox(
        "Choose the time unit for aggregation:",
        frequency_map.keys(),
        index=4,
    )
    selected_frequency = frequency_map[granularity]

    with st.spinner(text="Calculating interaction counts...", show_time=False, width="content"):
        fig = plot_inters_px(
            selected_df,
            start_date=start_date,
            end_date=end_date,
            granularity=selected_frequency,
            title=f'Number of Interactions per {granularity}'
            )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Interactions',
        legend_title='Dataset'
    )

    st.plotly_chart(fig, config={"width":"stretch"})

    st.markdown(f"##### Total number of interactions from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    total_interactions_cols = st.columns(len(selected_df.keys()))
    inters_dict = {}

    with st.spinner(text="Calculating interactuions table...", show_time=False, width="content"):
        for i, (name, df) in enumerate(selected_df.items()):
            inters_df = inters_per_period(df, start_date, end_date, selected_frequency)
            inters_dict[name] = inters_df.rename(columns={"n_inters": name}).set_index("timestamp")

            n_inters = inters_df["n_inters"].sum()
            total_interactions_cols[i].metric(label=name, value=f'{n_inters:,}')


if __name__ == "__main__":
    main()
