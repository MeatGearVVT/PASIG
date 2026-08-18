"""
Time deltas page.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import ks_2samp

from streamlit_ui.utils import (load_data, load_data_split, page_config,
                                select_test_and_reference_subsets)


warnings.filterwarnings('ignore')
page_config()
pd.set_option('display.max_columns', None)

if not st.session_state.data_loaded:
    st.warning("Please load data to access this page")
    st.stop()


def main():

    st.title("Comparison of Different Splits: Distribution of time gaps between interactions")
    st.markdown(f'**Dataset:** {st.session_state.selected_dataset}')

    st.write("This page analyzes the time gaps between consecutive interactions \
             in the dataset across different splits.")
    st.write("The goal is to examine the time delta between each **target** interaction \
             (the ground-truth item the model is supposed to predict) \
             and the preceding interaction in the **test** or **validation** data. \
             We expect that the distribution of these time deltas should be similar \
             to the overall distribution in the dataset. Therefore, we compare it \
             with the distribution of time deltas between all interactions \
             in a **reference** subset (**preprocessed**, **raw**, or **train**). \
             A significant deviation between these distributions indicates \
             a potential distribution shift, suggesting that the data split may not be \
             fully representative of the properties of the dataset.")
    plot_pictures()

    target_subset, base_subset, selected_splits = select_test_and_reference_subsets()

    df_base = load_data(base_subset)
    dfs_inputs = {split: load_data_split(split, f"{target_subset}_input")
                  for split in selected_splits}
    dfs_targets = {split: load_data_split(split, f"{target_subset}_target")
                   for split in selected_splits}

    df_combined, base_deltas, deltas = compute_deltas(df_base, dfs_inputs, dfs_targets, base_subset)

    plot_histogram(df_combined)

    ks_stats = {}
    for split in dfs_inputs.keys():
        ks_stats[split], _ = ks_2samp(deltas[split]['delta'].dropna().values,
                                      base_deltas['delta'].dropna().values)
    st.write('To quantify the difference between the compared splits and the reference subset, \
             the Kolmogorov–Smirnov (KS) statistic is computed (smaller KS indicates closer distributions).')
    col, _ = st.columns([1, 2])
    with col:
        st.write(pd.Series(ks_stats, name='Kolmogorov-Smirnov statistic'))

    st.subheader("Descriptive statistics of time deltas")
    percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    stats = pd.DataFrame({base_subset: base_deltas['delta'].describe(percentiles=percentiles)})
    for split in selected_splits:
        stats[split] = deltas[split]['delta'].describe(percentiles=percentiles)
    st.write(stats.T)


def plot_pictures():

    with st.expander("Pictures illustrating time deltas"):

        st.write('The **time deltas** between all consecutive interactions in the reference subset.')
        col, _ = st.columns([1, 1])
        with col:
            st.image("streamlit_ui/pictures/time_deltas_interactions.png", width='stretch')

        st.write('The **time deltas** between target interaction and the preceding interaction.')
        col, _ = st.columns([1, 1])
        with col:
            st.image("streamlit_ui/pictures/time_deltas_targets.png", width='stretch')


def compute_deltas(df_base, dfs_inputs, dfs_targets, base_subset):

    base_deltas = df_base.sort_values(['user_id', 'timestamp'])
    base_deltas['delta'] = base_deltas.groupby('user_id')['timestamp'].diff()

    deltas = {}
    for split in dfs_inputs.keys():
        deltas[split] = deltas_between_subsets(dfs_inputs[split], dfs_targets[split])

    df_combined = base_deltas.assign(source=base_subset)[['source', 'delta']]
    for split in dfs_inputs.keys():
        df_combined = pd.concat(
            [df_combined, deltas[split].assign(source=split)[['source', 'delta']]])

    return df_combined, base_deltas, deltas


def deltas_between_subsets(df_base, df_new):

    last_input = df_base.sort_values(
        ['user_id', 'timestamp']).groupby('user_id').last()['timestamp']
    first_target = df_new.sort_values(
        ['user_id', 'timestamp']).groupby('user_id').first()['timestamp']

    deltas = pd.merge(last_input, first_target, on='user_id',
                      suffixes=('_last_input', '_first_target'))
    deltas['delta'] = deltas['timestamp_first_target'] - deltas['timestamp_last_input']

    return deltas


def plot_histogram(df_combined):

    plot_col1, plot_col2 = st.columns([0.2, 0.8])

    with plot_col1:
        st.header('\n')
        log_scale_y = st.checkbox('Logarithmic Y-axis', key='log_scale_y')
        log_scale_x = st.checkbox('Logarithmic X-axis', key='log_scale_x', value=True)
        percent_checkbox = st.checkbox('Percent', key='percent', value=True)
        overlay_checkbox = st.checkbox('Overlay barmode', key='overlay', value=False)
        n_bins = st.slider('Number of bins', min_value=5, max_value=100,
                           value=30, key='n_bins')

        max_delta = df_combined.delta.max()
        if log_scale_x:
            max_delta = np.log(1 + max_delta)
        max_value = st.slider('Maximum delta', min_value=0.0,
                              max_value=max_delta, value=max_delta, key='max_value')
        if log_scale_x:
            max_value = np.exp(max_value) - 1

    with plot_col2:
        df_plot = df_combined[df_combined.delta <= max_value]
        if log_scale_x:
            df_plot['delta'] = np.log(1 + df_plot['delta'])
        fig = px.histogram(
            df_plot,
            x='delta',
            color='source',
            nbins=n_bins,
            barmode='overlay' if overlay_checkbox else 'group',
            color_discrete_sequence=px.colors.qualitative.Pastel1,
            opacity=0.7,
            histnorm='percent' if percent_checkbox else None,
            title='Time deltas',
            log_y=log_scale_y
        )
        st.plotly_chart(fig, config={"width": "stretch"})


if __name__ == "__main__":

    main()
