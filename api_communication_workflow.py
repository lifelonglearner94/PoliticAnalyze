from Facticity_api import FacticityAPI
from dotenv import load_dotenv
import os
from zyla_api import zyla_check_fact_api, dummy_zyla_check_fact_api

def extract_claims(list_of_sentences, batch_size=20):
    """
    Extract claims from text using Facticity API and save them to a JSON file.

    Parameters:
        results_dict (dict): A dictionary containing a "sentences" key with a list of sentences to process.
        base_folder (str): Directory where the output JSON file will be saved.
        json_filename (str): Name of the JSON file to save the extracted claims.

    Returns:
        None
    """
    # Load API key from environment variables
    load_dotenv()
    api_key = os.getenv("FACT_API_KEY")
    if not api_key:
        raise ValueError("FACT_API_KEY not found in environment variables.")

    # Initialize Facticity API client
    facticity = FacticityAPI(api_key)

    extracted_claims = []

    # Process sentences in batches of 20
    for i in range(0, len(list_of_sentences), batch_size):
        print(f"Processing sentences {i} to {i+batch_size}...")
        input_text = " ".join(list_of_sentences[i:i+batch_size])
        try:
            claims = facticity.extract_claim(text=input_text)
            extracted_claims.extend(claims.get("claims", []))
            print("Extracted Claims:", claims.get("claims", []))
        except Exception as e:
            print(f"Error processing sentences {i} to {i+batch_size}: {e}")
    return extracted_claims


def fact_checking_facticity(claims_list):
    """
    Performs fact-checking on a list of claims using the Facticity API.

    Args:
    claims_list (list): A list of claims to be fact-checked.

    Returns:
    list: A list of tuples, where each tuple contains (claim, classification).
            Classification will be the API result or "Error" if processing failed.
    """
    load_dotenv()
    api_key = os.getenv("FACT_API_KEY")

    facticity = FacticityAPI(api_key)

    # Initialize the list to store claim-classification pairs
    claim_classification_list = []

    for idx, claim in enumerate(claims_list):
        try:
            # Fact-check the claim using the API
            result = facticity.fact_check(query=claim)

            # Extract the classification result
            classification = result['Classification']

            # Append the claim and its classification as a tuple to the list
            claim_classification_list.append((claim, classification))
            print(idx, "Successfully fact-checked claim:", claim, classification)
        except Exception as e:
            print(f"Error processing claim: {e}")
            print(result)
            # In case of an error, mark the claim with "Error" in the list
            claim_classification_list.append((claim, "Error"))

    return claim_classification_list


def fact_checking_zyla(claims_list):
    """
    Performs fact-checking on a list of claims using the Zyla Labs API.

    Args:
    claims_list (list): A list of claims to be fact-checked.

    Returns:
    list: A list of tuples, where each tuple contains (claim, classification).
            Classification will be the API result or "Error" if processing failed.
    """
    from itertools import zip_longest
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv("ZYLA_API_KEY")

    # Initialize the list to store claim-classification pairs
    claim_classification_list = []

    # Helper function to group claims into chunks of three
    def group_claims(iterable, n, fillvalue=""):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

    # Group claims into chunks of 3
    grouped_claims = group_claims(claims_list, 3)

    for idx, group in enumerate(grouped_claims):
        # Join up to 3 claims with spaces
        concatenated_claims = " ".join(filter(None, group))  # Filter to remove empty strings

        try:
            # Fact-check the concatenated claims using the API
            result = zyla_check_fact_api(concatenated_claims, api_key)
            #result = dummy_zyla_check_fact_api(concatenated_claims, api_key)

            list_of_current_claim_classi_pairs = list(result["fact_check"].items())

            # Append the claim and its classification as a tuple to the list
            claim_classification_list.extend(list_of_current_claim_classi_pairs)
            print(idx*3, "Successfully fact-checked claims:", list_of_current_claim_classi_pairs)
        except Exception as e:
            print(f"Error processing claims: {e}")
            print(result)

    return claim_classification_list


def fact_checking_zyla_RAW(claims_list):
    """
    Performs fact-checking on a list of claims using the Zyla Labs API.

    Args:
    claims_list (list): A list of claims to be fact-checked.

    Returns:
    list: A list of tuples, where each tuple contains (claim, classification).
            Classification will be the API result or "Error" if processing failed.
    """
    from itertools import zip_longest
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv("ZYLA_API_KEY")

    # Initialize the list to store claim-classification pairs
    claim_classification_list = []

    # Helper function to group claims into chunks of three
    def group_claims(iterable, n, fillvalue=""):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

    # Group claims into chunks of 3
    grouped_claims = group_claims(claims_list, 3)

    for idx, group in enumerate(grouped_claims):
        # Join up to 3 claims with spaces
        concatenated_claims = " ".join(filter(None, group))  # Filter to remove empty strings

        try:
            # Fact-check the concatenated claims using the API
            result = zyla_check_fact_api(concatenated_claims, api_key)

            # Append the claim and its classification as a tuple to the list
            claim_classification_list.append((concatenated_claims, result))


            print(idx*3, "Successfully fact-checked claims:", result)

        except Exception as e:
            print(f"Error processing claims: {e}")
            print(result)

    return claim_classification_list

if __name__ == "__main__":
    from decision_helper_lib import read_from_json#, translate_sentences
    # TESTING

    JSON_BASE_PATH = "saved_on_harddrive"
    json_path_full_results_dict = os.path.join(JSON_BASE_PATH, "full_results_dict.json")

    full_results_dict = read_from_json(json_path_full_results_dict)

    for key, value in full_results_dict.items():
        claims_list = value.get("extracted_claims", [])
        claim_classification_list = fact_checking_zyla(claims_list)
        break
    print(claim_classification_list)
