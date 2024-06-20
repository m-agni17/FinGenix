# FinGenix 🧞

## Overview

*FinGenix* is a powerful tool designed to simplify and enhance decision-making in the financial sector using data. It allows users to interact with both CSV data and MySQL databases using natural language queries. Leveraging the capabilities of the Llama 3 language model, FinGenix transforms your natural language prompts into precise SQL or Python queries, providing accurate and insightful results.

## Features

- *Natural Language Processing:* Type your query in plain English, and FinGenix will generate the corresponding SQL or Python code.
- *Versatile Data Sources:* Connect to MySQL databases or upload CSV files to start analyzing your data.
- *Intuitive Interface:* A user-friendly Streamlit interface to make data querying accessible to everyone.
- *Result Beautification:* Get your query results in a clear and structured format, making it easy to interpret and act upon.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/m-agni17/FinGenix.git
    cd FinGenix
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application:**

    ```bash
    streamlit run app.py```

## Usage

### Home Page

The homepage provides an introduction and an inviting interface to start your data analysis journey. From here, you can select your data source and proceed.

![WhatsApp Image 2024-06-20 at 2 53 06 PM](https://github.com/m-agni17/LlamaAnalytics/assets/113231945/51aba7d3-d781-4115-983a-b7c59efd46a5)


### Connect to MySQL

1. *Enter MySQL connection details* in the sidebar: Host, Username, Password, and Database.
2. *Connect to the database* by clicking the "Connect to MySQL" button.
3. *Enter your query* in the text area and click "Chat with MySQL" to get results.

![WhatsApp Image 2024-06-20 at 3 05 51 PM](https://github.com/m-agni17/LlamaAnalytics/assets/113231945/224252db-f947-430e-9af5-2ef90c35e499)
![WhatsApp Image 2024-06-20 at 3 11 03 PM](https://github.com/m-agni17/LlamaAnalytics/assets/113231945/f1df71f9-d55c-4c37-941d-1e77b7c60617)


### Upload CSV

1. *Upload your CSV files* using the file uploader in the sidebar.
2. *Select a CSV file* from the dropdown list.
3. *Enter your query* in the text area and click "Chat with CSV" to get results.

![WhatsApp Image 2024-06-20 at 2 56 25 PM](https://github.com/m-agni17/LlamaAnalytics/assets/113231945/87e68b73-a1cf-41eb-8295-e52a3cc6bcb3)
![WhatsApp Image 2024-06-20 at 2 54 17 PM](https://github.com/m-agni17/LlamaAnalytics/assets/113231945/0e76d1b8-00c0-46f6-9b76-e132982711a7)

## How It Works

1. *Schema Extraction:* For MySQL, the schema of the database is extracted to help formulate accurate queries.
2. *Query Refinement:* The user’s natural language query is refined using the fine tuned Llama 3 language model to generate precise SQL or Python code.
3. *Query Execution:* The refined query is executed on the connected data source (MySQL or CSV).
4. *Result Beautification:* The raw results are transformed into a human-readable format using LLM.

## Fine-Tuned Model

To ensure the highest accuracy and relevance in query results, the Llama 3 model used in FinGenix has been fine-tuned specifically for financial data analysis. This fine-tuning process involved training the model on a comprehensive dataset of financial queries and databases, allowing it to understand and process financial terminology and concepts with greater precision.
![WhatsApp Image 2024-06-20 at 3 13 22 PM](https://github.com/m-agni17/LlamaAnalytics/assets/113231945/633d1bf3-88de-49df-8ecc-a3a1d4aa479e)

| Model Type         | Average Perplexity | Average BLEU | Average ROUGE-1 | Average ROUGE-L |
|:-------------------|-------------------:|-------------:|----------------:|----------------:|
| Base LLaMA-3       |               2.80 |         0.51 |            0.67 |            0.67 |
| Fine-tuned LLaMA-3 |               2.66 |         0.63 |            0.75 |            0.75 |

These metrics demonstrate the improvements achieved through fine-tuning, showing enhanced performance across various evaluation criteria. For further details on the fine-tuning methodology and dataset used, please refer to the documentation provided in the repository.

## Example

### MySQL

 ```bash
host = "your_host"
user = "your_user"
password = "your_password"
database = "your_database"
user_prompt = "Show me the top 10 customers by sales."

result = execute_mysql_query(host, user, password, database, user_prompt, llm)
print(result)
```

### CSV

```bash
import pandas as pd

df = pd.read_csv("your_file.csv")
user_prompt = "What is the average sales amount?"

result = chat_with_csv(df, user_prompt)
print(result)
```

## Contributing

We welcome contributions to FinGenix! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin feature/your-feature)
5. Create a new Pull Request
