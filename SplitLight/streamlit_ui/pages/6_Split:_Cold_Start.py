"""
Cold start page.
"""

import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.stats.cold import cold_counts, cold_stats
from streamlit_ui.utils import load_data, page_config, subset_selector


warnings.filterwarnings('ignore')
page_config()
pd.set_option('display.max_columns', None)

if not st.session_state.data_loaded:
    st.warning("Please load data to access this page")
    st.stop()


def main():

    st.title(f'Cold Start Interactions')
    st.markdown(f'**Dataset:** {st.session_state.selected_dataset}' + (f' | **Split:** {st.session_state.selected_split}' if st.session_state.selected_split else ''))


    st.markdown("An interaction from the **analyzed subset** is considered **cold** if it involves a new entity \
    (a `user_id` or `item_id`) that **did not appear** in the reference data (typically the training subset)."                                                                                                                                                                                                                                                                                                                                          )
    with st.expander("Cold Start visualization"):
        col, _ = st.columns([1, 1.2])
        with col:
            st.image("streamlit_ui/pictures/cold_start.png", width='stretch')
    st.markdown("##### Select Analyzed and Reference Subsets")

    subset_choice = ["train", "test_target"] if "train" in st.session_state.df_paths.keys() else [st.session_state.default_subset]*2
    warm_subset, test_subset = subset_selector(subset_choice)

    df_warm = load_data(warm_subset)
    df_test = load_data(test_subset)

    st.markdown("##### Select Date Range and Time Granularity")

    min_date = pd.to_datetime(df_test['timestamp'].min(), unit='s').date()
    max_date = pd.to_datetime(df_test['timestamp'].max(), unit='s').date()

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

    granularity = st.selectbox(
        "Choose the time unit for aggregation:",
        ('Hour', 'Day', 'Week', 'Month'),
        index=3,
    )

    frequency_map = {
        'Hour': 'H',
        'Day': 'D',
        'Week': 'W',
        'Month': 'M'
    }

    selected_frequency = frequency_map[granularity]

    # Plot
    plot_col1, plot_col2 = st.columns([0.2, 0.8])
    with plot_col1:
        st.subheader("Select Entity")
        entity = st.radio("Select Entity",
                ['Cold Users', 'Cold Items'],
                index=0,
                key='cold_col',
                label_visibility='collapsed')

        cold_col = 'item_id' if entity == 'Cold Items' else 'user_id'

    with plot_col2:
        st.subheader("Cold Start Statistics for Analyzed Subset")
        summary_stats = cold_counts(df_test, df_warm, cold_col)
        entity_stats = cold_stats(df_test, df_warm)

        col4, col5, col6 = st.columns(3)
        col4.metric(f"{entity}", f"{entity_stats['Number'][entity]:,}")
        col5.metric(f"Total {entity.split(' ')[1]}", f"{df_test[cold_col].nunique():,}")
        col6.metric(f"Share of {entity}", f"{entity_stats.loc[entity, 'Share (by count)']:.4f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Cold Interactions", f"{summary_stats['cold_interactions']:,}")
        col2.metric("Total Interactions", f"{summary_stats['total_interactions']:,}")
        col3.metric(f"Share of {entity}' interactions", f"{summary_stats['cold_share']:.4f}")


    test_stats = pd.DataFrame(
        cold_counts(df_test, df_warm, cold_col, selected_frequency)
        ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=test_stats['timestamp'],
            y=test_stats['cold_share'],
            name='Cold Share',
            mode='lines+markers',
            line=dict(color='#17becf', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Bar(
            x=test_stats['timestamp'],
            y=test_stats['total_interactions'] - test_stats['cold_interactions'],
            name='Warm Interactions',
            marker_color='salmon',
            opacity=0.5
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(
            x=test_stats['timestamp'],
            y=test_stats['cold_interactions'],
            name='Cold Interactions',
            marker_color='lightblue',
            opacity=0.7
        ),
        secondary_y=False
    )

    # Set layout
    fig.update_layout(
        title_text=f"{entity}' Interactions",
        xaxis_title='Date',
        legend_title='Metric',
        barmode='stack',
        hovermode='x unified'
    )

    fig.update_yaxes(
        title_text="Number of Interactions",
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Cold Share (ratio)",
        secondary_y=True,
        range=[0, 1.05],
        tickformat=".0%"
    )

    st.plotly_chart(fig, config={"width":"stretch"})


if __name__ == "__main__":
    main()
