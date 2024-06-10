from pandasai.llm.local_llm import LocalLLM  # Importing LocalLLM for local Meta Llama 3 model
import streamlit as st
import pandas as pd  # Pandas for data manipulation
from pandasai import SmartDataframe  # SmartDataframe for interacting with data using LLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to chat with CSV data and format the output
def chat_with_csv(df, query):
    try:
        # Initialize LocalLLM with Meta Llama 3 model
        llm = LocalLLM(
            api_base="http://localhost:11434/v1",
            model="llama3"
        )
        # Initialize SmartDataframe with DataFrame and LLM configuration
        pandas_ai = SmartDataframe(df, config={"llm": llm})
        
        # Chat with the DataFrame using the provided query
        raw_result = pandas_ai.chat(query)
        logging.info(f"Raw result from LLM: {raw_result}")

        if isinstance(raw_result, pd.DataFrame):
            # Convert DataFrame to a readable string format
            formatted_result = "The analysis result based on your query '{}' is:\n".format(query)
            formatted_result += raw_result.to_string(index=False)
        else:
            # If the result is not a DataFrame, just return the result as-is
            formatted_result = raw_result

        # Split the results into pointwise format
        pointwise_result = "\n".join(f"- {line.strip()}" for line in formatted_result.split("\n") if line.strip())

        return pointwise_result
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return f"An error occurred: {e}"

# Set layout configuration for the Streamlit page
st.set_page_config(layout='wide')
# Set title for the Streamlit application
st.title("Multiple-CSV ChatApp powered by LLM")

# Upload multiple CSV files
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# Check if CSV files are uploaded
if input_csvs:
    # Select a CSV file from the uploaded files using a dropdown menu
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)

    # Load and display the selected CSV file
    st.info("CSV uploaded successfully")
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data.head(3), use_container_width=True)

    # Enter the query for analysis
    st.info("Chat Below")
    input_text = st.text_area("Enter the query")

    # Perform analysis
    if input_text:
        if st.button("Chat with CSV"):
            with st.spinner('Processing your query...'):
                st.info("Your Query: " + input_text)
                result = chat_with_csv(data, input_text)
                if "An error occurred" in result:
                    st.error(result)
                else:
                    st.success(result)
                    # Optionally, add visualization code herev