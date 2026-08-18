"""
Time deltas page.
"""

import warnings

import pandas as pd
import plotly.express as px
import streamlit as st

from scipy.stats import ks_2samp
from src.stats.utils import combine_and_cumcount, cumcount_by_user
from streamlit_ui.utils import load_data, load_data_split, page_config, select_test_and_reference_subsets


warnings.filterwarnings('ignore')
page_config()
pd.set_option('display.max_columns', None)

if not st.session_state.data_loaded:
    st.warning("Please load data to access this page")
    st.stop()


def main():

    st.title(f"Comparison of Different Splits: Distribution of item positions")
    st.markdown(f'**Dataset:** {st.session_state.selected_dataset}')

    st.write("This page analyzes the distribution of interaction positions \
              within user sequences across different splits.")
    st.write("The goal is to examine the position (i.e., the ordinal index) \
             of each target interaction (the ground-truth item the model is supposed to predict) \
             within its corresponding sequence in the **test** or **validation** data. \
             It is compared with the distribution of all interaction positions \
             in a **reference** subset (e.g., **preprocessed**, **raw**, or **train**).")

    target_subset, base_subset, selected_splits = select_test_and_reference_subsets()

    df_base = load_data(base_subset)
    dfs_inputs = {split: load_data_split(split, f"{target_subset}_input")
                  for split in selected_splits}
    dfs_targets = {split: load_data_split(split, f"{target_subset}_target")
                   for split in selected_splits}

    positions_base = cumcount_by_user(df_base)
    positions_target = {}
    for split in dfs_inputs.keys():
        _, positions_target[split] = combine_and_cumcount(dfs_inputs[split], dfs_targets[split])

    df_combined = positions_base.assign(source=base_subset)[['source', 'cumcount']]
    for split in dfs_inputs.keys():
        df_combined = pd.concat(
            [df_combined, positions_target[split].assign(source=split)[['source', 'cumcount']]])

    plot_histogram(df_combined)

    ks_stats = {}
    for split in dfs_inputs.keys():
        ks_stats[split], _ = ks_2samp(positions_target[split]['cumcount'].dropna().values,
                                      positions_base['cumcount'].dropna().values)
    st.write('To quantify the difference between the compared splits and the reference subset, \
             the Kolmogorov–Smirnov (KS) statistic is computed (smaller KS indicates closer distributions).')
    col, _ = st.columns([1, 2])
    with col:
        st.write(pd.Series(ks_stats, name='Kolmogorov-Smirnov statistic'))

    st.subheader("Descriptive statistics of item positions")
    percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    stats = pd.DataFrame({
        base_subset: positions_base['cumcount'].describe(percentiles=percentiles)})
    for split in selected_splits:
        stats[split] = positions_target[split]['cumcount'].describe(percentiles=percentiles)
    st.write(stats.T)


def plot_histogram(df_combined):

    plot_col1, plot_col2 = st.columns([0.2, 0.8])

    with plot_col1:
        st.header('\n')
        log_scale_y = st.checkbox('Logarithmic Y-axis', key='log_scale_y')
        percent_checkbox = st.checkbox('Percent', key='percent', value=True)
        overlay_checkbox = st.checkbox('Overlay barmode', key='overlay', value=False)
        n_bins = st.slider('Number of bins', min_value=5, max_value=100,
                           value=30, key='n_bins')

        max_position = df_combined['cumcount'].max()
        min_value = 0.0 if isinstance(max_position, float) else 0
        max_value = st.slider('Maximum position', min_value=min_value,
                              max_value=max_position, value=max_position, key='max_value')

    with plot_col2:
        df_plot = df_combined[df_combined['cumcount'] <= max_value]
        fig = px.histogram(
            df_plot,
            x='cumcount',
            color='source',
            nbins=n_bins,
            barmode='overlay' if overlay_checkbox else 'group',
            color_discrete_sequence=px.colors.qualitative.Pastel1,
            opacity=0.7,
            histnorm='percent' if percent_checkbox else None,
            title='Item positions',
            log_y=log_scale_y,
            labels={'cumcount': 'position'}
        )
        st.plotly_chart(fig, config={"width": "stretch"})


if __name__ == "__main__":

    main()
