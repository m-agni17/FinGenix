import logging
import pandas as pd
import streamlit as st
import mysql.connector
from langchain_community.llms import Ollama
import base64
import os


# Configure logging
logging.basicConfig(level=logging.INFO)
llm = Ollama(model="llama3")

def get_db_structure(cursor):
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    db_structure = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()
        column_info = [(column[0], column[1]) for column in columns]
        db_structure[table_name] = column_info
    return db_structure



def render_homepage():
    # Read the image file
    current_directory = os.path.dirname(__file__)

    # Construct the relative path to the image file
    image_path = os.path.join(current_directory, 'images', 'bg image.png')
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Create the HTML with the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: right;
            background-repeat: no-repeat;
        }}
        .homepage-title {{
            text-align: center;
            color: white;
            margin-top: 50px;
        }}
        .homepage-subtitle {{
            text-align: center;
            color: white;
        }}
        .homepage-description {{
            text-align: center;
            color: white;
            max-width: 800px;
            margin: auto;
            padding: 20px;
            background: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="homepage-title">Welcome to FinGenix 🧞</h1>', unsafe_allow_html=True)
    st.markdown('<h4 class="homepage-subtitle">A Financially Specialized Bot for Informed Decision-Making with Data</h2>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="homepage-description">
            FinGenix is your companion for querying both CSV data and MySQL databases using natural language.
            \nPlease select your data source from the sidebar to get started!
        </div>
        """,
        unsafe_allow_html=True
    )




def execute_mysql_query(host, user, password, database, user_prompt, llm):
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()

        # Get database schema
        db_structure = get_db_structure(cursor)

        # Create the prompt for the LLM
        prompt = (
            f"Now you are a MySQL Expert. The user entered prompt for the given data is '{user_prompt}' "
            f"and the schema details along with the table name is given as dictionary. The schema details are {db_structure}. "
            "By using the table schema details and user prompt, convert the user prompt into a MySQL query to give a correct answer for the user entered prompt. "
            "Give only the query, no need for any explanations. The query should be in 1 line with SELECT statement in that. "
            "No need for any comments. Also, where any conditional words are given in the prompt, use lower() to reduce errors due to case sensitivity."
        )

        # Refine the query using LLM
        refined_query = llm.invoke(prompt)
        logging.info(f"Refined SQL query: {refined_query}")

        # Execute the refined query
        cursor.execute(refined_query)
        result = cursor.fetchall()

        # Close cursor and connection
        cursor.close()
        conn.close()

        # Beautify the result using LLM
        beautified_result = beautify_result_with_llm(llm, user_prompt, result)
        return beautified_result

    except Exception as e:
        logging.error(f"Error executing MySQL query: {e}")
        raise

def refine_query_with_llm(llm, query_, df):
    try:
        logging.info("Starting query refinement with LLM...")
        # Get column names and types from the dataset
        column_info = {col: str(df[col].dtype) for col in df.columns}

        # Constructing the message
        prompt = (
            f"Now you are a Python Expert. The user entered prompt for the given data is '{query_}' "
            f"and the column names in the dataset and their types are {column_info}. "
            "By using this column details and user prompt, convert the user prompt into a python code to give a correct answer for the user entered prompt. "
            "Give only the code, no need for any explanations. The code should be in 1 line with print statement in that. "
            "No need for any import or comments and the dataset is stored in dataframe 'df'. "
            "Also, where any conditional words are given in the prompt, use lower() to reduce errors due to case sensitivity."
        )

        refined_query = llm.invoke(prompt)
        logging.info("Query refinement with LLM completed successfully.")
        return refined_query

    except Exception as e:
        logging.error(f"Error in refining query with LLM: {e}")
        raise

def beautify_result_with_llm(llm, user_prompt, result):
    try:
        logging.info("Starting result beautification with LLM...")
        prompt = (
            f"Now you are a data analyst and your name is FinGenix.you will be given prompts regarding financial sector. The user prompt was '{user_prompt}' and the raw result obtained is {result}. "
            "Convert this raw result into a human-readable format with descriptions if necessary. "
            "The output should be short and in a structured way."
        )
        
        beautified_result = llm.invoke(prompt)
        logging.info("Result beautification with LLM completed successfully.")
        return beautified_result

    except Exception as e:
        logging.error(f"Error in beautifying result with LLM: {e}")
        raise

def chat_with_csv(df, query_):
    try:
        # Refine the query for better precision using LLM
        refined_query = refine_query_with_llm(llm, query_, df)
        logging.info(f"Refined query from LLM: {refined_query}")

        # Define a safe execution environment
        local_vars = {'df': df}

        # Modify the refined query to assign to 'result'
        if 'print(' in refined_query:
            start = refined_query.index('print(') + len('print(')
            end = refined_query.rindex(')')
            refined_query = f"result = {refined_query[start:end]}"

        # Execute the generated code
        exec(refined_query, globals(), local_vars)
        result = local_vars['result'] if 'result' in local_vars else "No result variable found in the executed code."

        # Beautify the result using LLM
        beautified_result = beautify_result_with_llm(llm, query_, result)
        
        logging.info("Chat with CSV data completed successfully.")
        return beautified_result

    except Exception as e:
        logging.error(f"An error occurred during chat with CSV: {e}")
        raise

def main():
    try:
        # Set layout configuration for the Streamlit page
        st.set_page_config(layout='wide')
        

        # Initialize session state
        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        # Select data source: CSV upload or MySQL database connection
        data_source = st.sidebar.selectbox("Select Data Source", ["Home","Connect to MySQL", "Upload CSV"])
        if data_source == "Home":
            render_homepage()
        elif data_source == "Connect to MySQL":
            
            st.title("DataLlama Insights 🦙")
            st.subheader('Harnessing MySQL Data for Informed Decisions')
            st.sidebar.info("Enter MySQL connection details:")
            host = st.sidebar.text_input("Host")
            user = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            database = st.sidebar.text_input("Database")

            if st.sidebar.button("Connect to MySQL"):
                try:
                    # Test MySQL connection
                    conn = mysql.connector.connect(
                        host=host,
                        user=user,
                        password=password,
                        database=database
                    )
                    conn.close()
                    st.success("Connected to MySQL database successfully")
                except mysql.connector.Error as err:
                    st.error(f"Error: {err}")
                    return

            if host and user and password and database:
                input_text = st.text_area("Enter the query")
                if input_text and st.button("Chat with MySQL"):
                    with st.spinner('Processing your query...'):
                        st.session_state.conversation.append({"user": input_text})
                        result = execute_mysql_query(host, user, password, database, input_text, llm)
                        st.session_state.conversation.append({"response": result})

            # Display conversation history
            st.markdown("""
            <style>
            
            .user-message { 
                text-align: right; 
                padding: 10px; 
                margin: 10px;
                color: #a83c32; 
            }
            .bot-message { 
                text-align: left; 
                padding: 10px; 
                margin: 10px;
                color: #1E90FF; 
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
                    st.markdown(f"<div class='bot-message'><span class='message-label'>🧞:</span> {entry['response']}</div>", unsafe_allow_html=True)

        elif data_source == "Upload CSV":
            st.title("DataLlama Insights 🦙")
            st.subheader('Harnessing CSV Data for Informed Decisions')
            input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)
            if input_csvs:
                selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
                selected_index = [file.name for file in input_csvs].index(selected_file)
                data = pd.read_csv(input_csvs[selected_index])
                st.dataframe(data.head(3), use_container_width=True)
                input_text = st.text_area("Enter the query")
                if input_text:
                    if st.button("Chat with CSV"):
                        with st.spinner('Processing your query...'):
                            st.session_state.conversation.append({"user": input_text})
                            result = chat_with_csv(data, input_text)
                            st.session_state.conversation.append({"response": result})

            # Display conversation history
            st.markdown("""
            <style>
            .user-message { 
                text-align: right; 
                padding: 10px; 
                margin: 10px;
                color: #a83c32; 
            }
            .bot-message { 
                text-align: left; 
                padding: 10px; 
                margin: 10px;
                color: #00BFFF; 
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
