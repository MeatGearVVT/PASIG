"""
Temporal leakage page.
"""

import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.stats.leaks import temporal_overlap, leak_counts, find_shared_interactions
from streamlit_ui.summary_utils import format_time
from streamlit_ui.utils import filter_by_date, load_data, page_config, subset_selector


warnings.filterwarnings('ignore')
page_config()
pd.set_option('display.max_columns', None)

if not st.session_state.data_loaded:
    st.warning("Please load data to access this page")
    st.stop()


def main():

    st.title('Temporal Data Leakage')
    st.markdown(f'**Dataset:** {st.session_state.selected_dataset}' + (f' | **Split:** {st.session_state.selected_split}' if st.session_state.selected_split else ''))


    st.markdown("""This page offers the following data leakage checks (from the lest strict to the most strict):""")
    st.markdown("""
    - **Shared interactions:** Analyzed subset does not share any common interactions with the reference subset (typically training subset).
    - **Temporal leakage from future:** Reference subset does not include interactions with analyzed subset items from future \
        (with respect to the time of occurrence in the analyzed subset).
    - **Temporal overlap:** Analyzed subset does not overlap in time with the reference subset.
    """)

    with st.expander("More about Temporal leakage from future and Temporal overlap"):
        st.markdown("""An interaction from the analyzed subset is considered as **leaked** if it occurred **on \
            or before** the latest timestamp for the same `item_id` in the reference data (typically training subset). \
            """)
        col, _ = st.columns([1, 1.2])
        with col:
            st.image("streamlit_ui/pictures/data_leakage.png", width='stretch')
        
        st.markdown("""**Temporal overlap** of analyzed subset with the training data:
            """)
        col, _ = st.columns([1, 1.2])
        with col:
            st.image("streamlit_ui/pictures/duration_and_overlap.png", width='stretch')

    st.markdown("##### Select Analyzed and Reference Subsets")

    subset_choice = ["train", "test_target"] if "train" in st.session_state.df_paths.keys() else [st.session_state.default_subset]*2
    base_subset, new_subset = subset_selector(subset_choice)

    df_base = load_data(base_subset)
    df_new = load_data(new_subset)

    st.markdown("##### Select Date Range and Time Granularity")

    min_date = pd.to_datetime(df_new['timestamp'].min(), unit='s').date()
    max_date = pd.to_datetime(df_new['timestamp'].max(), unit='s').date()

    st.write(f"**Data available** from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

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
        ('Hour', 'Day', 'Week', 'Month', 'Year'),
        index=3,
    )

    frequency_map = {
        'Hour': 'H',
        'Day': 'D',
        'Week': 'W',
        'Month': 'M',
        'Year': 'Y',
    }

    selected_frequency = frequency_map[granularity]

    with st.spinner(text="Calculating statistics...", show_time=False, width="content"):
        df_new = filter_by_date(df_new, start_date, end_date)

        leak_stats = pd.DataFrame(
            leak_counts(df_new, df_base, selected_frequency)
        ).reset_index()

    st.subheader("Data Leakage Statistics for Analyzed Subset")
    col1, col2, col3 = st.columns(3)

    with st.spinner(text="Calculating statistics...", show_time=False, width="content"):
        summary_stats = leak_counts(
            df_new,
            df_base
            )

    col1.metric("Total Interactions", f"{summary_stats['total_interactions']:,}")
    col2.metric("Leaked Interactions", f"{summary_stats['leak_interactions']:,}")
    col3.metric("Share of Leaked Interactions", f"{summary_stats['leak_share']:.4f}")

    # Plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=leak_stats['timestamp'],
            y=leak_stats['leak_share'],
            name='Leaked Interactions Share',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Bar(
            x=leak_stats['timestamp'],
            y=leak_stats['total_interactions'] - leak_stats['leak_interactions'],
            name='Non-leaked Interactions',
            marker_color='lightblue',
            opacity=0.5
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(
            x=leak_stats['timestamp'],
            y=leak_stats['leak_interactions'],
            name='Leaked Interactions',
            marker_color='salmon',
            opacity=0.7
        ),
        secondary_y=False
    )

    # Set layout
    fig.update_layout(
        title_text='Interactions Distribution in Analyzed Subset: Data Leakage',
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
        title_text="Leaked Interactions Share (ratio)",
        secondary_y=True,
        range=[0, 1.05],
        tickformat=".0%"
    )

    st.plotly_chart(fig, config={"width":"stretch"})

    ## Interactions Overlap
    df_base['timestamp'] = pd.to_datetime(df_base['timestamp'], unit='s')
    shared_interactions = find_shared_interactions(df_new, df_base)

    if not shared_interactions.empty:
        st.warning(f"Warning: subsets have **{len(shared_interactions)}** shared interactions!")
    else:
        st.success("Sanity Check: **No shared interactions between subsets detected.**")  

    ## Temporal Overlap
    overlap_df = temporal_overlap(df_new, df_base)
    row = overlap_df.iloc[0]

    if pd.notna(row.overlap_start):
        overlap_duration = format_time(row.overlap_duration_sec)

    if row.overlap_duration_sec > 0:
        st.warning(
            f""" 
            **Temporal overlap detected!**  
            Overlap period: **{row.overlap_start.date()} → {row.overlap_end.date()}**   
            Duration: **{overlap_duration}**  
            Share: **{row.overlap_share_reference:.1%}** of reference subset, **{row.overlap_share_target:.1%}** of analyzed subset
            """
        )

        timeline_data = {
            "Subset": [base_subset, new_subset],
            "Start": [overlap_df["reference_start"][0], overlap_df["target_start"][0]],
            "End": [overlap_df["reference_end"][0], overlap_df["target_end"][0]],
            "Color": ["salmon", "lightblue"],
        }
        import plotly.express as px

        fig = px.timeline(
            timeline_data,
            x_start="Start",
            x_end="End",
            y="Subset",
            color="Subset",
            color_discrete_map={base_subset: "salmon", new_subset: "lightblue"}
        )

        fig.update_yaxes(autorange="reversed", title_text="")
        fig.update_traces(width=0.5)
        fig.update_layout(
            height=150,
            margin=dict(l=20, r=20, t=10, b=10),
        )

        st.plotly_chart(fig)
    else:
        st.success(
            f"""
            **No temporal overlap detected.**  
            Reference subset ends: **{row.reference_end.date()}**  
            Analyze subset starts: **{row.target_start.date()}**
            """,
        )


if __name__ == "__main__":
    main()
