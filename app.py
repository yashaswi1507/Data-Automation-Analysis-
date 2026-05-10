import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import DataPreprocessor
from eda import show_summary
from query_engine import parse_query, execute_query, generate_query_insight
from ml_engine import train_prediction_model

st.set_page_config(layout="wide")
st.title("Data Analysis Automation Tool")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
   
    # ================= SIDEBAR =================
    st.sidebar.title("Controls")

    st.sidebar.markdown("Data Cleaning Options")
    
    outlier_option = st.sidebar.selectbox(
        "Handle Outliers",
        ["No Action", "Remove Outliers", "Cap Outliers"]
    )
    
    missing_option = st.sidebar.selectbox(
    "Handle Missing Values",
    ["Mean", "Median", "Mode", "Drop Rows"]
    )   
    
    # PREPROCESSING
    processor = DataPreprocessor(df, outlier_option, missing_option)
    clean_df, report = processor.process()

    # SIDEBAR FILTER
    st.sidebar.header("Filters")

    filtered_df = clean_df.copy()
    for col in clean_df.select_dtypes(include='object').columns:
        selected = st.sidebar.selectbox(f"Filter by {col}", ["All"] + list(clean_df[col].unique()))
        if selected != "All":
            filtered_df = filtered_df[filtered_df[col] == selected]

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "EDA", "Query", "ML"])

    # ================= DASHBOARD =================
    with tab1:
        st.subheader("Dashboard Overview")

        num_cols = filtered_df.select_dtypes(include='number').columns

        if len(num_cols) > 0:
            col1, col2, col3 = st.columns(3)

            col1.metric("Total Rows", len(filtered_df))
            col2.metric("Average Value", round(filtered_df[num_cols[0]].mean(), 2))
            col3.metric("Max Value", filtered_df[num_cols[0]].max())

            # Plotly Chart
            fig = px.histogram(filtered_df, x=num_cols[0])
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("Raw Data Preview")

        rows_to_show = st.slider("Select number of rows", 5, 50, 10)

        st.dataframe(df.head(rows_to_show), use_container_width=True)

        st.subheader("Data Cleaning Report")
        for r in report:
            st.success(r)
            st.write("✔", r)
            
        st.markdown("Cleaned Data Preview")
        st.dataframe(clean_df.head(rows_to_show), use_container_width=True)
        
        st.markdown("Download Cleaned Data")

        csv = clean_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download Cleaned CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    # ================= EDA =================
    with tab2:
        st.subheader("Exploratory Data Analysis")
        show_summary(filtered_df, st)

        for col in filtered_df.select_dtypes(include='number').columns:
            fig = px.box(filtered_df, y=col)
            st.plotly_chart(fig, use_container_width=True)

    # ================= QUERY =================
    with tab3:
        st.subheader("Ask Questions")

        st.markdown("""
        **Examples:**
        - average math score  
        - max reading score by gender  
        """)

        query = st.text_input("Enter query")

        if query:
            op, target, group = parse_query(query, filtered_df)
            result = execute_query(filtered_df, op, target, group)

            st.write(result)

            insight = generate_query_insight(result, target, group)
            if insight:
                st.write("🧠", insight)

    # ================= ML =================
    with tab4:
        st.subheader("Prediction")

        target = st.selectbox("Select target column", filtered_df.select_dtypes(include='number').columns)

        if st.button("Train Model"):
            model, score = train_prediction_model(filtered_df, target)

            if model:
                st.write(f"Accuracy (R²): {score:.2f}")

                input_data = {}
                for col in filtered_df.select_dtypes(include='number').columns:
                    if col != target:
                        input_data[col] = st.number_input(f"Enter {col}", value=0.0)

                if st.button("Predict"):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)
                    st.success(f"Predicted {target}: {prediction[0]:.2f}")