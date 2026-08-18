"""
Core statistics page.
"""

import warnings

import pandas as pd
import plotly.express as px
import streamlit as st

from src.stats.base import compare_subsets, calc_lifetime, get_deltas
from src.stats.plots import plot_hist_px
from streamlit_ui.utils import convert_time, load_data, page_config, prepare_to_print


warnings.filterwarnings('ignore')
page_config()
pd.set_option('display.max_columns', None)

if not st.session_state.data_loaded:
    st.warning("Please load data to access this page")
    st.stop()


@st.cache_data
def calculate_metrics(df, group_by, metric, time_unit):
    def _process_df(df):
        # Convert time unit to seconds
        if time_unit == 'minutes':
            time_factor = 60
        elif time_unit == 'hours':
            time_factor = 60 * 60
        elif time_unit == 'days':
            time_factor = 60 * 60 * 24
        elif time_unit == 'years':
            time_factor = 60 * 60 * 24 * 365
        else:
            time_factor = 1

        if metric == 'Number of interactions':
            data = df[group_by].value_counts()
            x_title = "Number of Interactions"
        
        elif metric == 'Lifetime':
            data = calc_lifetime(df, col=group_by)['lifetime'] / time_factor
            x_title = f"Lifetime ({time_unit})"
        
        elif metric == 'Time between interactions':
            data = get_deltas(df, col=group_by)['delta'] / time_factor
            x_title = f"Time between interactions ({time_unit})"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return data.rename('count').to_frame(), x_title

    if isinstance(df, dict):
        result = {}
        for key, df in df.items():
            result[key], x_title = _process_df(df)
        return result, x_title
    else:
        return _process_df(df)


def create_figure(data, x_title, metric, group_by, log_scale, num_bins):
    fig = px.histogram(
        data,
        x=data.name,
        nbins=num_bins,
        title=f"{metric} by {group_by.replace('_', ' ').title()}",
        labels={'value': x_title},
        log_y=log_scale
    )
    fig.update_layout(
        bargap=0.1,
        xaxis_title=x_title,
        yaxis_title="Count",
        hovermode="x"
    )
    return fig


def main():

    st.title("Core and Temporal Statistics")
    st.markdown(f'**Dataset:** {st.session_state.selected_dataset}' + (f' | **Split:** {st.session_state.selected_split}' if st.session_state.selected_split else ''))

    st.write("Core statistics are the most important statistics that are used to describe the dataset. \
        The statistics are calculated for the analyzed subsets and compared with the reference subset if selected (e.g. `test_target` as an analyzed subset and `train` as a reference subset).\
            Mean (Avg.) and Median are available for distribution-based statistics at the table, and histogram and quantiles are available below the table. ")
    st.write("**Core statistics** include the number of interactions, users, items, average sequence length, density (interactions / (users × items)), mean and median item occurrence and user activity.  \n \
            **Temporal statistics** include the between-interaction time, user and item lifetimes, the maximum and minimum timestamps, dataset timeframe and amount of timestamp duplicates.")

    with st.expander("More details on Temporal Statistics"):
        st.markdown('''
        - The **dataset timeframe** is the time between the first and last interaction in the dataset.
        - The between-interaction time (**time delta between interactions**) is the time between two consecutive interactions of the same user.
        ''')
        col, _ = st.columns([1, 1.2])
        with col:
            st.image("streamlit_ui/pictures/time_deltas_interactions.png", width='stretch')

        st.markdown('''
        - The amount of **timestamp collisions** is the number of times the same timestamp appears in a history of a user. \
            Timestamp collisions (%) is the timestamp collisions as a percentage of the number of interactions. 

            Presence of timestamp collisions is a sign of data quality issues and make the temporal splitting results non-deterministic.
        - The **user lifetime** is the time between the first and last interaction of the user. \
            User lifetime (%) is the user lifetime as a percentage of the dataset timeframe.
        - The **item lifetime** is the time between the first and last interaction with the item. \
            Item lifetime (%) is the item lifetime as a percentage of the dataset timeframe.
        ''')
        col, _ = st.columns([1, 1.2])
        with col:
            st.image("streamlit_ui/pictures/user_lifetime.png", width='stretch')
    
    st.markdown("If you want to compare different splits in terms of core statistics, use the link below:")
    st.page_link(page="streamlit_ui/pages/7_Compare_Splits:_Core_and_Temporal.py", label="Compare Splits", icon="🔀")

    st.markdown("##### Select Analyzed Subsets")
    available_subsets = list(st.session_state.df_paths.keys())
    selected_subsets = st.multiselect(
        "Select analyzed subsets:",
        available_subsets,
        default=st.session_state.default_subset,
        key="target",
        label_visibility="collapsed",
        )
    
    if not selected_subsets:
        st.warning("Please select at least one subset to analyze.")
        st.stop()

    selected_df = {subset: load_data(subset) for subset in selected_subsets}

    comp_col1, comp_col2 = st.columns([0.125, 0.875])

    with comp_col1:
        to_compare = st.checkbox('Compare with: ')

    with comp_col2:
        subset2 = st.selectbox(
            "Reference subset",
            available_subsets,
            index=available_subsets.index(st.session_state.default_subset),
            key="base",
            disabled=not to_compare,
            label_visibility="collapsed"
        )

    with st.spinner(text="Calculating statistics...", show_time=False, width="content"):
        if to_compare:
            df_base = load_data(subset2)
            ref_stats, stats = compare_subsets(selected_df, df_base, extended=True, return_ref_stats=True)
        else:
            df_base = None
            stats = compare_subsets(selected_df, df_base, extended=True)

    stats_to_print = prepare_to_print(stats)

    columns_to_select = stats_to_print.columns.get_level_values(0).unique()
    default_cols = columns_to_select[:6]

    stat_cols = st.columns([0.9, 0.1])
    with stat_cols[0]:
        st.markdown("##### Select Statistics")
        selected_cols = stat_cols[0].multiselect(
            "Select statistics:",
            columns_to_select,
            default=default_cols,
            label_visibility="collapsed"
        )
    with stat_cols[1]:
        st.markdown("##### Time unit")
        time_unit = st.selectbox("Time unit:",
                ['seconds', 'minutes', 'hours', 'days', 'years'],
                index=3,
                key='time_unit',
                label_visibility="collapsed")

    st.subheader("Core Statistics of Analyzed Subsets")
    stats_final = convert_time(stats_to_print, time_unit)
    st.dataframe(stats_final[selected_cols])

    if to_compare:
        st.subheader("Core Statistics of Reference Subset")
        ref_stats = prepare_to_print(ref_stats)
        ref_stats_final = convert_time(ref_stats, time_unit)
        st.dataframe(ref_stats_final[selected_cols].rename(index={0: subset2}))

    # Histograms

    st.subheader('Statistics distribution')
    st.markdown("Aggregate the analyzed subsets ether by user or by item, select one of the available statistics, get the quantiles, and visualize it in a histogram.  \n \
        Adjust the number of bins, time unit and scaling of the y-axis for your data.")
    
    # st.markdown("**Available statistics:**  \n* Number of interactions  \n* Lifetime  \n* Time between interactions")

    plot_col1, plot_col2 = st.columns([0.2, 0.8])

    def create_plot():
        data_target, x_title = calculate_metrics(selected_df,
                                    st.session_state.group_by,
                                    st.session_state.metric,
                                    st.session_state.time_unit)

        fig_input = data_target

        if to_compare:
            data_base, _ = calculate_metrics(df_base,
                                    st.session_state.group_by,
                                    st.session_state.metric,
                                    st.session_state.time_unit)

            fig_input.update({subset2: data_base})

        fig, metrics_to_display = plot_hist_px(
            fig_input,
            col='count',
            nbins=st.session_state.n_bins,
            barmode='overlay' if st.session_state.overlay else 'group',  # group, overlay, stack
            histnorm='percent' if st.session_state.percent else None,
            color_discrete_sequence=px.colors.qualitative.Pastel1,
            title=f"{st.session_state.metric} by {st.session_state.group_by.replace('_', ' ').title()}",
            labels={'Value': x_title},
            log_y=st.session_state.log_scale,
        )

        fig.update_layout(
            bargroupgap=0.1,
            xaxis_title=x_title,
            hovermode="x"
        )
        return fig, metrics_to_display

    with plot_col1:
        st.header('\n')
        group_by = st.radio("Group by:", ['item_id', 'user_id'],
                           key='group_by',
                           on_change=create_plot)

        metric = st.radio("Select metric:",
                         ['Number of interactions', 'Lifetime', 'Time between interactions'],
                         key='metric',
                         on_change=create_plot)

        log_scale = st.checkbox('Logarithmic Y-axis',
                              key='log_scale',
                              on_change=create_plot)
        percent_checkbox = st.checkbox('Percent', key='percent', value=True)
        overlay_checkbox = st.checkbox('Overlay barmode', key='overlay', value=False)

        n_bins = st.slider('Number of bins:',
                      min_value=5,
                      max_value=100,
                      value=30,
                      key='n_bins',
                      on_change=create_plot)

    with plot_col2:
        with st.spinner(text="Calculating statistics...", show_time=False, width="content"):
            fig, metrics_to_display = create_plot()
            st.plotly_chart(fig, config={"width":"stretch"})

    st.markdown(f"##### Descriptive statistics for {st.session_state.metric} by {st.session_state.group_by.replace('_', ' ').title()}")
    st.dataframe(
        metrics_to_display.reset_index().rename(columns={"index": "Subset"}),
        hide_index=True
        )


if __name__ == "__main__":
    main()
