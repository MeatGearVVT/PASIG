"""
Data load page.
"""

import os
import streamlit as st


st.set_page_config(layout="centered", page_icon=":streamlit:")

curr_dir = os.getcwd()
st.title("SplitLight")

st.image("streamlit_ui/pictures/teaser.png")
st.write("**SplitLight** is a lightweight toolkit that audits recommender datasets and splitting strategies. Automated health checks, drift analysis, leakage detection, and side-by-side split comparison.")

st.write("Easy to use -- upload your data, run the checks, explore the insights, and **pick the split you can trust**.")

st.subheader("To Start -- Load Your Data")

with st.expander("Data layout and example"):
    st.markdown("""
        **SplitLight expects each dataset under** `data/<DatasetName>/` with either a `raw.csv` (original schema) or `preprocessed.csv` (standard schema).
        - `raw.csv` (optional): original column names are defined in `runs/configs/dataset/<DatasetName>.yaml`
        - `preprocessed.csv` with `user_id`, `item_id`, `timestamp` (seconds)
        - After splitting, a per-split subfolder contains: `train.csv`, `validation_input.csv`, `validation_target.csv`, `test_input.csv`, `test_target.csv`

        Example:
    """)
    st.code("""
            data/
                Beauty/
                    raw.csv
                    preprocessed.csv
                    leave-one-out/
                        train.csv
                        validation_input.csv
                        validation_target.csv
                        test_input.csv
                        test_target.csv
                Diginetica/
                    preprocessed.csv
                    GTS-q09-val_by_time-target_last/
                        train.csv
                        validation_input.csv
                        validation_target.csv
                        test_input.csv
                        test_target.csv""")

def save_paths(dataset_path, split_path, file_format='csv'):
    st.session_state.df_paths = {}
    
    # Determine file extensions based on selected format
    if file_format == '.csv':
        extensions = ('.csv',)
    elif file_format == '.parquet':
        extensions = ('.parquet', '.pq')
    else: 
        extensions = ('.csv', '.parquet', '.pq')
    
    # Search in dataset path
    for f in os.listdir(dataset_path):
        if f.lower().endswith(extensions):
            key = os.path.splitext(f)[0]
            st.session_state.df_paths[key] = os.path.join(dataset_path, f)
    
    # Search in split path
    if split_path and os.path.exists(split_path):
        for f in os.listdir(split_path):
            if f.lower().endswith(extensions):
                key = os.path.splitext(f)[0]
                st.session_state.df_paths[key] = os.path.join(split_path, f)
    
    if not st.session_state.df_paths:
        st.warning(f"No {file_format.upper()} files found in dataset or split folders")
    
    available_subsets = list(st.session_state.df_paths.keys())
    st.session_state.default_subset = "raw" if "raw" in available_subsets else available_subsets[0]

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'data_path' not in st.session_state:
    st.session_state.data_path = os.path.join(curr_dir, 'data')

if 'selected_format' not in st.session_state:
    st.session_state.selected_format = '.csv'

new_data_path = st.text_input(
    'Enter Root Data Directory Path',
    value=st.session_state.data_path,
    help="Path to the directory containing your datasets relative to where you run the script."
)

if new_data_path != st.session_state.data_path:
    st.session_state.data_path = new_data_path
    st.success(f"Data path updated to: {st.session_state.data_path}")

# File format selector
file_format = st.selectbox(
    'Select File Format',
    options=['.csv', '.parquet', 'both'],
    index=['.csv', '.parquet', 'both'].index(st.session_state.selected_format),
    help="Choose which file format(s) to load"
)

if file_format != st.session_state.selected_format:
    st.session_state.selected_format = file_format
    st.session_state.data_loaded = False

dataset_folders = [f for f in os.listdir(st.session_state.data_path) 
                  if os.path.isdir(os.path.join(st.session_state.data_path, f))]

if not dataset_folders:
    st.warning("No dataset folders found in the root directory")
else:
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = dataset_folders[0]
    
    selected_dataset = st.selectbox(
        'Select Dataset',
        options=sorted(dataset_folders),
        index=0,
        key='dataset_selector',
        help="SplitLight expects each dataset under data/<DatasetName>/ with either a raw.csv (original schema) or preprocessed.csv (standard schema)"
    )

    if selected_dataset != st.session_state.selected_dataset:
        st.session_state.selected_dataset = selected_dataset
        st.session_state.data_loaded = False

    dataset_path = os.path.join(st.session_state.data_path, selected_dataset)

    # Split options selection (direct subfolders of dataset)
    split_options = [f for f in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, f))]
    
    if split_options:
        if 'selected_split' not in st.session_state:
            st.session_state.selected_split = split_options[0]
        
        selected_split = st.selectbox(
            'Select Split',
            options=split_options,
            index=split_options.index(st.session_state.selected_split)
            if st.session_state.selected_split in split_options else 0,
            key='split_selector',
        )



        if selected_split != st.session_state.selected_split:
            st.session_state.selected_split = selected_split
            st.session_state.data_loaded = False

        st.session_state.split_path = os.path.join(dataset_path, selected_split)
    else:
        st.warning(f"No split options found in {selected_dataset}")
        st.session_state.split_path = dataset_path
        st.session_state.selected_split = ''


# Load data button
if st.button('Load Data', type='primary'):
    try:
        with st.spinner('Loading dataset...'):
            st.cache_data.clear()
            save_paths(dataset_path, st.session_state.split_path, st.session_state.selected_format)
            
            st.session_state.data_loaded = True
            st.success(f"Data loaded successfully from: {st.session_state.split_path} \n \n **You can now explore the data and run the checks.**")
            
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.session_state.data_loaded = False
