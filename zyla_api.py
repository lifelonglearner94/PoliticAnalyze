# https://zylalabs.com/api-marketplace/tools/fact+checking+api/2753
# 7 days free trial

# https://www.factiverse.ai/products
# könnte nicht geeignet sein für mein Projekt, weil sieht eher aus wie fertige software
import requests
import json
from time import sleep
def claim_buster_check_fact(input_claim, api_key):

    # Define the endpoint (url) with the claim formatted as part of it, api-key (api-key is sent as an extra header)
    api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/query/knowledge_bases/{input_claim}"
    request_headers = {"x-api-key": api_key}

    # Send the GET request to the API and store the api response
    api_response = requests.get(url=api_endpoint, headers=request_headers)

    return api_response.json()

def claim_buster_claim_spotter(input_text, api_key):

    # Define the endpoint (url) with the claim formatted as part of it, api-key (api-key is sent as an extra header)
    api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/sentences/{input_text}"
    request_headers = {"x-api-key": api_key}

    # Send the GET request to the API and store the api response
    api_response = requests.get(url=api_endpoint, headers=request_headers)

    return api_response.json()

def zyla_check_fact_api(user_content, api_key, max_retries=3, retry_delay=5):
    """
    Sends a GET request to the Fact Checking API to verify a fact with retry logic.

    Args:
        user_content (str): The content to be fact-checked.
        api_key (str): Your API key for authentication.
        max_retries (int): Maximum number of retry attempts.
        retry_delay (int): Delay between retries in seconds.

    Returns:
        dict or None: The response from the API, parsed as JSON, or None if all retries fail.
    """
    url = "https://zylalabs.com/api/2753/fact+checking+api/2860/check+facts"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    params = {
        "user_content": user_content
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} of {max_retries} for API call...")
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 429:
                print("Rate Limit Reset:", response.headers.get("X-Zyla-RateLimit-Reset", "seconds"))
                print("Rate Limit:", response.headers.get("X-Zyla-RateLimit-Limit"))
            response.raise_for_status()  # Raise an exception for HTTP errors
            response_json = response.json()
            return json.loads(response_json[0])  # Parse and return JSON response
        except requests.exceptions.RequestException as e:
            print(f"An error occurred on attempt {attempt}: {e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
            else:
                print("Max retries reached. Returning None.")
                return None


def dummy_zyla_check_fact_api(claim, api_key):
    """
    Dummy function that simulates a response from the Zyla Labs fact-checking API.

    Args:
    claim (str): A claim or concatenated claims to be fact-checked.
    api_key (str): An API key (not used in this dummy function).

    Returns:
    dict: Simulated fact-checking results.
    """
    return {
        'statement': claim,
        'fact_check': {
            'Klimawandel verursacht durch menschliche Aktivitäten': True,
            'Situation in Deutschland': False
        }
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    import requests
    import json

    load_dotenv()
    #claim_buster_api_key = os.getenv("CLAIM_BUSTER_API_KEY")
    zyla_api_key = os.getenv("ZYLA_API_KEY")

    print(zyla_check_fact_api("Der Klimawandel wird durch menschliche Aktivitäten verursacht. Deutschland geht den Bach herunter.", zyla_api_key))
