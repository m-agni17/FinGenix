from pandasai.llm.local_llm import LocalLLM  # Importing LocalLLM for local Meta Llama 3 model
from langchain_community.llms import Ollama
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe  # SmartDataframe for interacting with data using LLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)



# Function to refine the query using LLM and dataset context
def refine_query_with_llm(llm, query_, df):
    try:
        logging.info("Starting query refinement with LLM...")
        # Get column names and types from the dataset
        column_info = {col: str(df[col].dtype) for col in df.columns}
        
        # Constructing the message
        p1 = (
            "Now you are a Python Expert"
            f"The user entered prompt for the given data is '{query_}' "
            f"and the column names in the dataset and their types are {column_info}. "
            "By using this column details and user prompt, "
            "convert the user prompt into a python code to give a correct answer for the user entered prompt.Give only the code no need any explanations. the code should be in 1 line with print statement in that. no need any import or comments and the dataset is stored in dataframe 'df'"
            "And also the code should not contain '\' also where ever any conditional words are given in the prompt, use lower() to reduce the error occuring due to case sensitivity"
        )  

        refined_query = llm.invoke(p1)

        # Refine the prompt using LLM
        logging.info("Query refinement with LLM completed successfully.")
        return refined_query
    
    except Exception as e:
        logging.error(f"Error in refining query with LLM: {e}")
        raise

# Function to chat with CSV data and format the output
def chat_with_csv(df, query_):
    try:
        
        # Initialize LocalLLM with Meta Llama 3 model
        llm = Ollama(model="llama3")
        pandas_ai = SmartDataframe(df, config={"llm": llm})
        
        # Refine the query for better precision using LocalLLM
        pointwise_result = refine_query_with_llm(llm, query_, df)
        refined_query = (f"Execute the given code in the dataframe. the code is'{pointwise_result}' ")
        raw_result = pandas_ai.chat(refined_query)
        
        logging.info("Starting chat with CSV data using refined query...")
        
        logging.info(f"Raw result from LLM: {raw_result}")

        logging.info("Chat with CSV data completed successfully.")
        return raw_result
    
    except Exception as e:
        logging.error(f"An error occurred during chat with CSV: {e}")
        raise

# Example usage within Streamlit application
def main():
    try:
        # Set layout configuration for the Streamlit page
        st.set_page_config(layout='wide')
        # Set title for the Streamlit application
        st.title("DataLlama Insights 🦙")
        st.subheader('Harnessing Data for Informed Decisions')

        # Upload multiple CSV files
        input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

        # Initialize a session state variable to store conversation history
        if "conversation" not in st.session_state:
            st.session_state.conversation = []

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
                        st.session_state.conversation.append({"user": input_text})
                        result = chat_with_csv(data, input_text)
                        st.session_state.conversation.append({"response": result})

            # Display the conversation history
            st.markdown("""
            <style>
            .user-message { 
                text-align: right; 
                padding: 10px; 
                margin: 10px;
                color: #a83c32; /* Gold color for YOU */
            }
            .bot-message { 
                text-align: left; 
                padding: 10px; 
                margin: 10px;
                color: #00BFFF; /* DeepSkyBlue color for BOT */
            }
            .message-label {
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)
            
            for entry in st.session_state.conversation:
                if "user" in entry:
                    st.markdown(f"<div class='user-message'><span class='message-label'>😎:</span> {entry['user']}</div>", unsafe_allow_html=True)
                if "response" in entry:
                    st.markdown(f"<div class='bot-message'><span class='message-label'>🦙:</span> {entry['response']}</div>", unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"An error occurred in the main application: {e}")
        raise

if __name__ == "__main__":
    main()


