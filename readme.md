# 📌 DeGate Project Overview
This project automates the process of fetching, processing, and updating trading data from the DeGate DEX. The data is collected daily and stored in a structured format to power a comprehensive analytics dashboard.

# 🎯 Goal
The primary objective of this project is to track and analyze DeGate protocol trading activity by:

- Fetching historical and real-time trading data from DeGate API.
- Checking and updating available trading pairs daily.
- Storing the processed data for visualization in Dune Analytics.
- Providing a clear, automated workflow to keep the dataset up to date.

# 🚨 Problem Statement
**Why is this needed?**
- No direct access to structured DeGate trading data for analysis.
- Pairs can change over time, so we need daily updates to track new assets.
- Manually fetching data is inefficient—automation is required.

**Solution:** 
- Automate data fetching, processing, and updating with Python scripts.
- Schedule daily updates via cron to ensure data accuracy.
- Store results in CSV format, ready for ingestion into Dune Analytics.

# 🔄 Process Workflow

## Prerequisites
Before running the script, ensure you have the following installed:

- Dune Analytics account;
- Flipside account;
- Python, pip, Jupyter Notebook (any other working environment);
- Required Python packages (install using `pip install -r requirements.txt`):
    -   `pandas`
    -   `numpy`
    -   `requests`
    -   `python-dotenv`
    -   `flipsidecrypto`
- Environment variables:
    -   `DUNE_API_KEY`: Your Dune Analytics API key.
    -   `FLIPSIDE_API_KEY`: Your Flipside Crypto API key.

Create a `.env` file in the root directory of the project with the following content:
DUNE_API_KEY=your_dune_api_key
FLIPSIDE_API_KEY=your_flipside_api_key

## Instruction:
- Create and activate a virtualenv
- Clone the notebook
  
### Setting up the `.env` file

To securely manage your API keys for accessing Flipside and Dune Analytics, you need to create a `.env` file in the project root directory. Storing API keys in a `.env` file helps prevent accidental exposure of sensitive information and simplifies configuration. Follow the steps below to set it up:

1. Create a file named `.env` in the root directory of the project.
2. Add your API keys to the file in the following format:

   ```env
   FLIPSIDE_API_KEY=your_flipside_api_key_here
   DUNE_API_KEY=your_dune_api_key_here
   ```

3. Save the `.env` file.
4. Ensure the `.env` file is included in your `.gitignore` file to avoid pushing it to version control.

   Example `.gitignore` entry:
   ```gitignore
   .env
   ```

5. The project uses the `dotenv` library to load these keys automatically. Ensure the library is installed in your environment using:

   ```bash
   pip install python-dotenv
   ```

## Working notebook
Before running the data-fetching scripts, you must create a table in Dune Analytics to store the trading data.

The script performs the following main tasks:           
   0. **Prepare the Table on Dune Analytics**Before running the data-fetching scripts, you must create a table in Dune Analytics to store the trading 
    data.                    
   Go to Dune Analytics -> Upload a dataset -> upload data programmatically via API (https://docs.dune.com/api-reference/tables/endpoint/create)       
   Create a new table named `your_table_name` with the following schema:

    ```
    schema = [
    {"name": "pair", "type": "varchar"},
    {"name": "date", "type": "timestamp", "nullable": False},
    {"name": "open", "type": "double", "nullable": False},
    {"name": "high", "type": "double", "nullable": False}, 
    {"name": "low", "type": "double", "nullable": False}, 
    {"name": "close", "type": "double", "nullable": False}, 
    {"name": "volume", "type": "double", "nullable": False}, 
    {"name": "close_time", "type": "timestamp", "nullable": False},
    {"name": "quote_volume", "type": "double", "nullable": False},
    {"name": "trades", "type": "double", "nullable": False}
    ]
    ```
1.  **Fetch Token Metadata:** Retrieves a list of token registered on DeGate platform using the Flipside API.
2.  **Identify Valid Trading Pairs:** Once we have the tokens, we need to determine which trading pairs exist on DeGate.
3.  **Fetch Trading Data:** Fetches kline data for valid trading pairs from the DeGate API.
4.  **Clean and Format Data:** Cleans and formats the fetched trading data for upload.
5.  **Upload Data to Dune Analytics:** Uploads the cleaned data to Dune Analytics.
6.  **Process New Tokens:** Detects and processes new tokens, updating metadata and fetching trading data accordingly.
7.  **Automate the Process with Cron Jobs:** To ensure daily updates, we use cron to run the scripts automatically. 

Note: Automating process is not included in the notebook. 
To schedule the scripts, you will need to convert your notebook to a `.py` file and then set up a cron job or similar scheduling tool to run the script daily.  
Open the crontab editor:
```
crontab -e

```
Add this line to schedule it to run every day at the required time with output/error logging:
```
0 0 * * * /usr/bin/python3 /path/to/your/script.py >> /path/to/logfile.log 2>&1
```
You can check logs anytime:
```
cat /path/to/logfile.log

```



By completing this project, we will achieve:         
✅ A fully automated pipeline to fetch, process, and store DeGate trading data.           
✅ A clean dataset that can be easily analyzed in Dune Analytics.            
✅ A real-time trading dashboard displaying:                 
    - Total trading volume per asset.              
    - Most traded pairs on DeGate.               
    - Historical price movements and trends.                   


**This project is still a work in progress, and I am actively seeking feedback and suggestions to improve it.**



