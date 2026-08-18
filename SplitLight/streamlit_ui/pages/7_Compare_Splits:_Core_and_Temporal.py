"""
Compare splits page.
"""

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from src.stats.base import group_subsets
from streamlit_ui.utils import convert_time, load_data_split, page_config, prepare_to_print


warnings.filterwarnings('ignore')
page_config()
pd.set_option('display.max_columns', None)

if not st.session_state.data_loaded:
    st.warning("Please load data to access this page")
    st.stop()

if st.session_state.selected_split == '':
    st.warning("No splits found")
    st.stop()


def main():
    st.title("Comparison of Different Splits: Core and Temporal Statistics")
    st.markdown(f'**Dataset:** {st.session_state.selected_dataset}')

    st.write("Use this page to compare different splits for selected dataset in terms of core statistics\
         to assess the splitting results and choose the best splitting strategy."                                                                                                                                                                                                                                                   )
    st.markdown("If you want to go into details of a specific split, use the link below:")
    st.page_link(page="streamlit_ui/pages/2_Core_and_Temporal_Statistics.py", label="Core and Temporal Statistics", icon="📊")

    st.markdown("#### Select Analyzed Subsets")
    available_subsets = list(st.session_state.df_paths.keys())
    available_subsets_target = [s for s in available_subsets if s not in ("raw", "preprocessed")]
    target_subsets = st.multiselect(
        "Select Analyzed Subsets",
        available_subsets_target,
        default=available_subsets_target[0],
        key="target",
        label_visibility="collapsed"
        )

    if not target_subsets:
        st.warning("Please select at least one subset to analyze.")
        st.stop()

    comp_col1, comp_col2 = st.columns([0.125, 0.875])

    with comp_col1:
        to_compare = st.checkbox('Compare with: ')

    with comp_col2:
        base_subset = st.selectbox(
            "Reference Subset",
            available_subsets,
            index=available_subsets.index(st.session_state.default_subset),
            key="base",
            disabled=not to_compare,
            label_visibility="collapsed",
        )

    dataset_path = os.path.join(st.session_state.data_path, st.session_state.selected_dataset)

    split_options = [f for f in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, f))]

    st.markdown("#### Select Splits to Compare")
    selected_splits = st.multiselect(
        "Select Splits to Compare",
        split_options,
        default=st.session_state.selected_split,
        label_visibility="collapsed",
    )

    with st.spinner(text="Calculating statistics...", show_time=False, width="content"):
        base_subsets = {split: load_data_split(split, base_subset) for split in selected_splits}
        combined_stats_base = group_subsets(base_subsets, extended=True)
        base_to_print = prepare_to_print(combined_stats_base)

        default_cols = base_to_print.columns[:6]

    st.markdown("#### Choose the statistics to display")
    st.write("The statistics are calculated for each analyzed subset and compared with the reference subset.")
    stat_cols = st.columns([0.9, 0.1])
    with stat_cols[0]:
        st.markdown("##### Select Statistics")
        selected_cols = st.multiselect(
            "Select statistics:",
            base_to_print.columns,
            default=default_cols,
            label_visibility="collapsed",
        )
    with stat_cols[1]:
        st.markdown("##### Time unit")
        time_unit = st.selectbox("Time unit:",
                ['seconds', 'minutes', 'hours', 'days', 'years'],
                index=0,
                key='time_unit',
                label_visibility="collapsed")



    for subset2 in target_subsets:
        st.header(f"Analyzed Subset: {subset2}")

        with st.spinner(text="Calculating statistics...", show_time=False, width="content"):
            new_subsets = {split: load_data_split(split, subset2) for split in selected_splits}
            combined_stats_new = group_subsets(new_subsets, extended=True)
            combined_df = prepare_to_print(combined_stats_new)

        if to_compare:
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns

            pct_share = (combined_df[numeric_cols] / base_to_print[numeric_cols] * 100).round(2)

            combined_df = pd.concat(
                [combined_df, pct_share],
                axis=1,
                keys=['Abs. value', '%']
            )
            combined_df = combined_df.swaplevel(axis=1)

        combined_df = convert_time(combined_df, time_unit)
        st.dataframe(combined_df[selected_cols])

    if to_compare:
        st.header(f"Reference Subset: {base_subset}")
        if base_subset == 'raw' or base_subset == 'preprocessed':
            base_to_print = convert_time(base_to_print, time_unit)
            st.dataframe(base_to_print.iloc[[0]][selected_cols].reset_index(drop=True).rename(index={0: base_subset}))
        else:
            st.dataframe(base_to_print[selected_cols])


if __name__ == "__main__":
    main()
