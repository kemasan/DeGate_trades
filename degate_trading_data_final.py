#!/usr/bin/env python
# coding: utf-8

import time
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import requests
from requests import Request
import json
import csv
import dotenv
import os
from dotenv import load_dotenv, find_dotenv
from flipside import Flipside
import logging
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


load_dotenv()
dotenv.load_dotenv('../../.env')
api_key = os.environ["DUNE_API_KEY"]
headers = {"X-Dune-API-Key": api_key}
flipside_key = os.environ["FLIPSIDE_API_KEY"]
output_dir = "data" 


# ==========================
#  TOKEN METADATA
# ==========================

def fetch_token_metadata():
    """
    Fetches the list of tokens registered on the platform from Flipside API.
    
    :param flipside_key: API key for FlipsideCrypto
    :param contract_address: The contract address to filter registered tokens
    :return: List of registered tokens
    """
    try:
        flipside = Flipside(flipside_key, "https://api-v2.flipsidecrypto.xyz")
    
        sql = """with tokens as (
        SELECT 
        DECODED_LOG: token as token
        , DECODED_LOG: tokenId as tokenId
        , *
        FROM ethereum.core.ez_decoded_event_logs

        where CONTRACT_ADDRESS = '0x9c07a72177c5a05410ca338823e790876e79d73b'
        AND EVENT_NAME = 'TokenRegistered'
        )

        SELECT 
        DISTINCT token
        , tokenId
        , case 
            when token = '0x0000000000000000000000000000000000000000' then 'ETH'
            when token = '0xf2d4900994b24adac8b396500f24e0557e2bf84d' then 'RATS'
            when token = '0x32e4a492068bee178a42382699dbbe8eef078800' then 'BIRDPAW'
            when token = '0x4877d83faac234dbd38d0f17bbc27a80604d3aed' then 'Illiterate' --DeflationaryToken
            when token = '0x3f1ee2f15da3eaf3539006b651144ec87755876d' then 'BRETT' --deGate -- Brdg Token
            when token = '0x4b63ce7e179d1db5ddac5d9d54e48356cf3e8b7d' then 'AIB' -- on ERC20 V1 (AIB)
            when token = '0xa1cfa45c8313c2da73b93454adcb1dab24f0993a' then 'Elon' --Eloncoin 
            when token = '0xe97fb0268474ef444d732f4ce5cfa6d8772b97c6' then 'W' --Wormhole token 
            when token = '0x9aea32b459e96c8ef5010f69130bf95fd129ac05' then 'WKLAY' --Wrapped Klay  Wormhole -BT
            when token = '0x754e822b92ba9be681c461347e0d2d38abc0cf16' then 'LBS' --Lbscoin
            when token = '0xe31cfcd36fed044ae4cf9405b577fe875762194f' then 'ALGO'
            when token = '0xbf8c53c59fad2aff7ffd925db435ea10c5ea4b6c' then 'SIC' --Sicuro 
            when token = '0x061f60d153beeb3f78ed5f38bb326c5b20a65503' then 'ZPG' --zk pet dog 
            when token = '0x8cdf7af57e4c8b930e1b23c477c22f076530585e' then 'APT' --Aptos Coin (
            when token = '0x1c88d38d04acd3edd9051ec587c67abff04bf30d' then 'NEAR' --Bridget Token contract (BT)
            when token = '0x215cff9fa9f3466b07bc6b5a5f30b925fb71163b' then 'MR' -- Martin 
            when token = '0xccf27d3fff920d999cc7e8a3fc847a96bca44ccd' then 'SVR' -- SVR Token 
            when token = '0x366863c4d67f87cc1238d1008c7053f96e53e559' then 'Sailor'
            when token = '0x99613a517d20944246ab6ef124a735227a8c1af3' then 'Olda' --Oldacoin
            when token = '0xce0cd513a069e8ec9cb625fcdf6d5f29aa912dbc' then 'MMS'
            when token = '0x3635ffc4f860055a7f64365d39a27de3d84eb78b' then 'MAGA' --Magacoin (
            when token = '0x06b089cbf0403ac2c5f452584f8a18978019b858' then 'TBC' --Trade Bot Coin (
            when token = '0xb821b75b42da3c9f38383e457fa33c4e4b85a314' then 'GREN' -- Grencoin (
            when token = '0x6215a0fc6ba68cbb0f99a9e1d3a5adf1321b6eb7' then '$MUSK' --MUSK COIN 
            when token = '0xefc0ced4b3d536103e76a1c4c74f0385c8f4bdd3' then 'PYTH'--Pyth Network 
            when token = '0x84074ea631dec7a4edcd5303d164d5dea4c653d6' then 'SUI' --Bridget Token contract
            when token = '0xf91e605af079384cc7077b3914a4a36019a89ee8' then 'SEI' --Bridget Token contract
            when token = '0xf4feec8cf825cd5b23f8abb3075c01c22abd4352' then 'DEGEN' -- BT
            when token = '0x8a00bf67a9bb032204da83408c4d1cd5421b40b8' then 'DEGEN' --DEGEN Donkey 
            when token = '0x57fbf85655c3a08bffe37a4f32f8adbd369508df' then 'OLD' --Oldtown Coin (
            when token = '0x8f5affe2443ea12c575ad5b13bf8fd235ed184c9' then 'TEST' 
            when token = '0x4b63ce7e179d1db5ddac5d9d54e48356cf3e8b7d' then 'AIB' 
            when token = '0x256a63a4900bddcf7703b601ac0b70aa2d7f9318' then 'SORA'
            when token = '0xf852ffa34a20113cd741f3cd9406a1a86b70c8ab' then 'UNI-V2' -- SAGE/ETH pair
            when token = '0xf02123509a08632339102ee5fdd41b638592e495' then 'VEN' --DUCATO 
            when token = '0xf2fdd72bd1581b9bca7b4391975dbacca7ec37e8' then 'SHIB' ---SHIBcoin 
            when token = '0x2ed58b1fa208e9a08fdaac2a839b8539abe558e8' then 'WIF' --BT - dogwifhat 
            when token = '0x5e9b0c790707b95457d56fc8c6411662f61d4d98' then 'LYV' -- LYV Finance (
            when token = '0xe32e3851e0a4216581342defac353d1efdfb36d9' then 'SMFC' --Social Media Finance Coin (
            when token = '0x285308b5fc68cc0f737c77cb60042b1fb5633e81' then 'ZEUS' --Zeus Vip Coin 
            when token = '0x8ce949b02edc782c04f9c618396a9f8c0e2b9274' then 'ZBC' -- Zeus Basis Coin 
            when token = '0x1df721d242e0783f8fcab4a9ffe4f35bdf329909' then 'OP' -- BT Optimism 
            when token = '0x8687a10bca6f139b25eb31020fcabb5782214764' then 'JUP' --Jupiter 
            when token = '0x0a866a8256832aaf048f274b9b538e795b64137f' then 'NAP' --Napcoin 
            else symbol end as symbol
        FROM tokens t
        left join ethereum.price.ez_asset_metadata p ON t.token = p.token_address
        """

        query_result_set = flipside.query(sql)

        if not query_result_set or not hasattr(query_result_set, 'query_id'):
                logging.error("Failed to get valid query result set from Flipside")
                return None
        
        logging.info(f"Query submitted successfully with ID: {query_result_set.query_id}")
            

        all_records = auto_paginate_result(query_result_set)
        if not all_records:
                logging.warning("No records returned from query")
                return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        if df.empty:
                logging.warning("DataFrame is empty after conversion")
                return None
        
        # Generate timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create output directory if it doesn't exist
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        

    # Save DataFrame to CSV
        filename = os.path.join(output_dir, f"token_metadata_{timestamp}.csv")
        df.to_csv(filename, index=False)
        
        logging.info(f"Token metadata saved to {filename} with {len(df)} records")
        return filename
        
    except Exception as e:
        logging.error(f"Error in fetch_token_metadata: {e}")

def auto_paginate_result(query_result_set, page_size=100):
    """
    Automatically paginates through the API results and returns all records.
    """
    flipside = Flipside(flipside_key, "https://api-v2.flipsidecrypto.xyz")
    current_page_number = 1
    total_pages = 2  # Initial assumption until we get actual total pages
    all_records = []
    
    while current_page_number <= total_pages:
        try:
            results = flipside.get_query_results(
                query_result_set.query_id,
                page_number=current_page_number,
                page_size=page_size
            )

            # Update total pages from the response
            total_pages = results.page.totalPages
            
            # Append records if available
            if results.records:
                all_records.extend(results.records)
                logging.info(f"Fetched page {current_page_number}/{total_pages}")
                current_page_number += 1
        except Exception as e:
            print(f"Error fetching page {current_page_number}: {e}")
            break
    
    return all_records


# ==========================
#  VALID TRADING PAIRS
# ==========================

QUOTE_CURRENCY_PAIRS = {
    "USDT": "all",
    "USDC": "all",
    "WBTC": ["ETH", "SOL"],
    "ETH": "all",
    "USDM": ["ETH", "DG", "MAP", "wsETH", "WBTC", "GRT"],
    "LUSD": ["ETH"]
}
BASE_URL = "https://v1-mainnet-backend.degate.com/order-book-ws-api/ticker/bookTicker"

# Session for API requests
session = requests.Session()

def fetch_ticker(base_symbol, base_id, quote_symbol, quote_id, token_dict, QUOTE_CURRENCY_PAIRS):
    """
    Fetch ticker data for a given base-quote pair and return structured data if valid.
    """
    pair_name = f"{base_symbol}/{quote_symbol}"
    params = {"base_token_id": base_id, "quote_token_id": quote_id}

    try:
        response = session.get(BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        result = response.json()

        # Validate API response
        if not isinstance(result, dict) or "code" not in result:
            logging.error(f"Unexpected response format for {pair_name}: {result}")
            return None

        if result.get("code") == 0:
            logging.info(f"Valid trading pair found: {pair_name}")
            return {
                "pair": pair_name,
                "base_symbol": base_symbol,
                "base_id": base_id,
                "quote_symbol": quote_symbol,
                "quote_id": quote_id
            }
    except requests.RequestException as e:
        logging.error(f"Error fetching data for {pair_name}: {e}")
    
    return None


def get_valid_pairs(new_tokens=None):
    """
    Main function to fetch and validate trading pairs.
    If `new_tokens` is passed, only those tokens are processed.
    """
    valid_trading_pairs = []
     # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load token list and filter out test tokens
    token_metadata_path = os.path.join(output_dir, "token_metadata.csv")
    
    # Load token list and filter out test tokens
    try: 
        df = pd.read_csv(token_metadata_path).rename(columns={'tokenid': 'id'}).drop(columns=['__row_index'], errors='ignore')
        logging.info(f"Token metadata loaded from {token_metadata_path}")
    except FileNotFoundError:
        logging.error(f"File '{token_metadata_path}' not found. Exiting.")
        exit(1)
    
    # Create a dictionary for token lookup (symbol -> id)
    token_dict = df.set_index("symbol")["id"].to_dict()

    # Filter out test tokens early to avoid unnecessary processing
    filtered_tokens = {symbol: token_id for symbol, token_id in token_dict.items() if "TEST" not in symbol}

     # If new_tokens are provided, only process those
    if new_tokens:
        filtered_tokens = {symbol: token_id for symbol, token_id in filtered_tokens.items() if symbol in new_tokens}
    
   
    # Prepare a list of valid quote tokens to avoid unnecessary iterations
    valid_quote_tokens = {q for q in QUOTE_CURRENCY_PAIRS.keys() if q in token_dict}


    # ThreadPoolExecutor with dynamic task submission
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        for base_symbol, base_id in filtered_tokens.items():
            for quote_symbol in valid_quote_tokens:
                valid_bases = QUOTE_CURRENCY_PAIRS[quote_symbol]
                
                # Skip if quote symbol doesn't allow trading with base
                if valid_bases != "all" and base_symbol not in valid_bases:
                    continue

                quote_id = token_dict[quote_symbol]
                futures.append(executor.submit(fetch_ticker, base_symbol, base_id, quote_symbol, quote_id, token_dict, QUOTE_CURRENCY_PAIRS))

                # Prevent excessive task queuing
                if len(futures) >= 500:  # Process tasks in batches (500 at a time) to prevent memory overload 
                    for future in as_completed(futures): # as_completed returns futures in order of completion
                        result = future.result()
                        if result:
                            valid_trading_pairs.append(result)
                    futures.clear() # free up memory before the next batch

        # Process remaining tasks
        for future in as_completed(futures):
            result = future.result()
            if result:
                valid_trading_pairs.append(result)


    # Save valid pairs to CSV with 5 columns
    if valid_trading_pairs:
        output_path = os.path.join(output_dir, "valid_trading_pairs.csv")
        df_valid_pairs = pd.DataFrame(valid_trading_pairs)
        df_valid_pairs.to_csv(output_path, index=False)
        logging.info(f"Valid trading pairs saved successfully to {output_path}.")
    else:
        logging.warning("No valid trading pairs found.")



# ==========================
#  FETCH TRADING DATA
# ==========================

# takes the output file as input from dg3_fetch_klines main() and returns the latest close time from the file
def get_latest_close_time():
    try:
        files = sorted(glob.glob(os.path.join(output_dir, 'trading_data_*.csv')))
        if not files:
            logging.warning("No trading data files found. Assuming first run.")
            return None
        
        last_trading_data_file = files[-1]
        df = pd.read_csv(last_trading_data_file)
        
        max_close_time = df.iloc[:, 7].max()  # Assuming the 8th column contains close_time
        if pd.isna(max_close_time):
            logging.warning(f"No valid 'close_time' found in {last_trading_data_file}. Assuming first run.")
            return None
        
        logging.info(f"Latest close_time found: {max_close_time}")
        return int(max_close_time)
    except Exception as e:
        logging.error(f"Error reading trading data file: {e}")
        return None
    

# generates the startend time for the klines fetch
def get_end_time_calculation():
    today_utc = datetime.now()  # Ensure UTC time
    return int(today_utc.timestamp()) * 1000


BASE_URL_KLINES = "https://v1-mainnet-backend.degate.com/order-book-ws-api/klines"
GRANULARITY = 86400  
session = requests.Session()
    # pairs_df = pd.read_csv("valid_trading_pairs.csv")
def fetch_trading_data_by_pair(pair, base_id, quote_id, start_time, end_time):
    """
    Fetch trading data for a given pair using session.
    """
    params = {
        "base_token_id": base_id,
        "quote_token_id": quote_id,
        "granularity": GRANULARITY,
        "start": start_time, #START_TIME,
        "end": end_time #END_TIME
    }

    try:
        logging.info(f"Fetching data for {pair} from {start_time} to {end_time}")
        response = session.get(BASE_URL_KLINES, params=params, timeout=3)
        response.raise_for_status()
        result = response.json()
        
        # Log API response for debugging
        if "data" not in result:
            logging.warning(f"No 'data' key in API response for {pair}: {result}")
            return None

        data = result.get("data", [])

        if not data:
            logging.info(f"No trading data for {pair} in the requested period.")
            return None

        df = pd.DataFrame(data)
        df.insert(0, "pair", pair)
        return df

    except requests.RequestException as e:
        logging.error(f"Error fetching {pair}: {e}")
        return None

def fetch_all_trading_data(start_time=None, end_time=None, output_file=None):    
    all_data = []
    
    os.makedirs(output_dir, exist_ok=True)

    # Load token list and filter out test tokens
    valid_pairs_path = os.path.join(output_dir, "valid_trading_pairs.csv")
    
    # Load token list and filter out test tokens
    try: 
        pairs_df = pd.read_csv(valid_pairs_path)
        logging.info(f"Valid pairs loaded from {valid_pairs_path}")
    except FileNotFoundError:
        logging.error(f"File '{valid_pairs_path}' not found. Exiting.")
        exit(1)

    # allows multiple API requests to run concurrently, improving performance
    with ThreadPoolExecutor(max_workers=min(10, len(pairs_df))) as executor:
        
        # stores async results as a key and tradig pair as a value
        futures = { 
            # submits API call for each pair
            executor.submit(fetch_trading_data_by_pair, row["pair"], row["base_id"], row["quote_id"], start_time, end_time): row["pair"]
            for _, row in pairs_df.iterrows()
            }
        # collecting results form API calls
        for future in as_completed(futures):
            # gets the returned data and add it to all_data if not None
            if (result := future.result()) is not None:
                all_data.append(result)
   
   # Saving the data to a CSV file
    if all_data:
        # Generate a unique filename if none is provided
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")  # Example: 20250228_165430
            output_file = f"trading_data_{timestamp}.csv"

        # Combine output_dir and output_file
        output_file_path = os.path.join(output_dir, output_file)

        pd.concat(all_data, ignore_index=True).to_csv(output_file_path, index=False) #test_raw_trading_data.csv
        print(f"INFO: Saved trading data to {output_file_path}.")

        # Pass the output file to the next function for further processing
        clean_prepare_upload_data(output_file_path)
    
        return output_file_path
    
    else:
        logging.warning("No trading data retrieved.")
        return None
    

# ==========================
#  FORMATTING & UPLOADING
# ==========================

import time
def clean_prepare_upload_data(input_file):
    """
    Clean and prepare the trading data from the input file.

    """
    try:
        df = pd.read_csv(input_file)
        logging.info(f"File {input_file} loaded successfully with shape {df.shape}")

        base = df.copy()
        # logging.info("Dataframe copied for transformation.")
        base.rename(columns={'0':'date','1':'open','2':'high','3':'low','4':'close','5':'volume','6':'close_time',
                            '7':'quote_volume','8':'trades','9':'taker_buy_base_vol','10':'taker_buy_quote_vol',
                            '11':'ignore','12':'_avg_price'}, inplace=True)
        logging.info(f"Columns renamed successfully. New columns: {list(base.columns)}")

        base = base.drop(['taker_buy_base_vol','taker_buy_quote_vol','ignore','_avg_price'], axis=1)
        logging.info(f"Columns dropped successfully. New columns: {list(base.columns)}")

        # Prepare df for further uploads
        base['date'] = pd.to_datetime(base['date'], unit='ms')
        base['close_time'] = pd.to_datetime(base['close_time'], unit='ms')
        logging.info(f"Converted 'date' and 'close_time' to datetime format.")

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the final cleaned data to the "data" directory with timestamp
        cleaned_file_path = os.path.join(output_dir, f'degate_updates_{time.strftime("%Y%m%d_%H%M%S")}.csv')

        base.to_csv(cleaned_file_path, index=False)
        logging.info(f"Data cleaned and saved to {cleaned_file_path}")

        # Upload to Dune with the correct path
        upload_data_to_dune(cleaned_file_path)
        
        return cleaned_file_path
    
    except Exception as e:
        logging.error(f"Error during data cleaning process: {e}", exc_info=True)
        return None
    

def upload_data_to_dune(file_path):
    """
    Upload the data to Dune.

    """
    try:
        url = "https://api.dune.com/api/v1/table/kemasan/degate_trades/insert"
        headers = {
            "X-DUNE-API-KEY": api_key,
            "Content-Type": "text/csv"
        }
        # last_trading_data_file = sorted(glob.glob('degate_updates__*.csv'))[-1]
        with open(file_path, "rb") as data:
            response = requests.request("POST", url, data=data, headers=headers)
        logging.info(f"Data uploaded to Dune successfully with response: {response.text}")
        return response.text

    except Exception as e:
        logging.error(f"Error during data upload process: {e}", exc_info=True)
    return None


# In[9]:


# ==========================
#  PROCESS NEW TOKENS
# ==========================

def load_token_list(file_path):
    """ Load token list from a given CSV file. """
    try:
        df = pd.read_csv(file_path)
        return set(df["symbol"])
    except FileNotFoundError:
        logging.warning(f"File {file_path} not found. Returning empty token list.")
        return set()
    
def update_token_metadata(new_tokens, existing_token_list, new_token_data, file_path):
    """ 
    Update token metadata with new tokens and save it to the CSV. 
    """
    if new_tokens:
        logging.info(f"New tokens detected: {new_tokens}")
        
        # Filter new tokens from the incoming data
        new_tokens_data = new_token_data[new_token_data["symbol"].isin(new_tokens)]
        
        # Append new data to old data and save
        updated_data = pd.concat([existing_token_list, new_tokens_data], ignore_index=True)
        updated_data.to_csv(file_path, index=False)
        
        return True
    
    return False

output_dir = "data"
def process_new_tokens():
    """ 
    Process and handle new tokens. 
    """
    # Load existing tokens from token_metadata.csv in output_dir
    token_metadata_path = os.path.join(output_dir, "token_metadata.csv")
    existing_token_list = load_token_list(token_metadata_path)

    # Fetch the latest token list from Flipside API (simulated here)
    try:
        latest_token_file = sorted(glob.glob(os.path.join(output_dir,"token_metadata_*.csv")))[-1] # from fetch_token_metadata()
        new_token_data = pd.read_csv(latest_token_file)
        new_token_list = set(new_token_data["symbol"])
    except IndexError:
        logging.error("No new token data found.")
        return
    
    new_tokens = new_token_list - existing_token_list
    # new_tokens.to_csv(f"new_tokens_{time.strftime('%Y%m%d_%H%M%S')}.csv", index=False) # pass file to get_valid_pairs()
    # If there are new tokens, update the metadata and fetch valid pairs
    if new_tokens:
        logging.info(f"New tokens detected: {new_tokens}")

        if update_token_metadata(new_tokens, existing_token_list, new_token_data, "token_metadata.csv"):
            logging.info("Fetching valid pairs for new tokens...")
            
            # call the function to get the valid pairs
            get_valid_pairs(new_tokens) # pass here a new tokens to avoid the processing of all tokens
            logging.info("Fetching trading data for new tokens...")
            
            # call the function to fetch trading data
            fetch_all_trading_data(get_latest_close_time() + 1, get_end_time_calculation())
    else:
        logging.info("No new tokens detected. Proceeding with trading data update.")
        fetch_all_trading_data(get_latest_close_time() + 1, get_end_time_calculation())


# ==========================
#  MAIN EXECUTION
# ==========================
def main():
    """ Main execution entry point. """
    try:
        process_new_tokens()
    except Exception as e:
        logging.error(f"Error occurred: {e}")


main()





