import streamlit as st


pages = {
    "": [ 
        st.Page("streamlit_ui/pages/0_SplitLight_load.py", title="SplitLight", icon="🏠"),
        st.Page("streamlit_ui/pages/1_Summary.py",   title="Summary",    icon="🔍"),
    ],
    "Dataset and Subsets": [
        st.Page("streamlit_ui/pages/2_Core_and_Temporal_Statistics.py", title="Core and Temporal Statistics", icon="📊"),
        st.Page("streamlit_ui/pages/3_Interactions_Distribution_in_Time.py", title="Interactions Over Time", icon="📈"),
        st.Page("streamlit_ui/pages/4_Repeated_Consumption.py", title="Repeat Consumption", icon="🔁"),
    ],
    "Split Analysis": [
        st.Page("streamlit_ui/pages/5_Split:_Temporal_Leakage.py", title="Temporal Leakage", icon="💧"),
        st.Page("streamlit_ui/pages/6_Split:_Cold_Start.py", title="Cold Start", icon="❄️"),
    ],
    "Compare Splits": [
        st.Page("streamlit_ui/pages/7_Compare_Splits:_Core_and_Temporal.py", title="Core and Temporal Statistics", icon="📊"),
        st.Page("streamlit_ui/pages/8_Compare_Splits:_Time_Deltas.py", title="Time Deltas", icon="⏱️"),
        st.Page("streamlit_ui/pages/9_Compare_Splits:_item_positions.py", title="Item Positions", icon="📍"),
    ],
}

selected_page = st.navigation(pages)
if selected_page is not None:
    selected_page.run()
    st.stop()

st.switch_page("streamlit_ui/pages/0_SplitLight.py")
