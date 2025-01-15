from Facticity_api import FacticityAPI
from dotenv import load_dotenv
import os

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


def fact_checking(claims_list):
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




if __name__ == "__main__":
    claims_list = [
        "Die Grünen setzen sich für den Umweltschutz ein."
    ]

    # Perform fact-checking on the list of claims
    claim_classification_list = fact_checking(claims_list)
    print("Fact-Check Result:", claim_classification_list)
