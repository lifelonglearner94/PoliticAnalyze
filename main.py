from decision_helper_lib import *
from api_communication_workflow import *
from finetune_bert_for_classi import *
import os


if __name__ == "__main__":
    DATA_PATH = "data"
    all_files_in_data = os.listdir(DATA_PATH)


    JSON_BASE_PATH = "saved_on_harddrive"
    os.makedirs(JSON_BASE_PATH, exist_ok=True)
    json_path_full_data_dict = os.path.join(JSON_BASE_PATH, "full_data_dict.json")

    def new_read():
        # Read pdfs
        all_raw_text = []
        filenames = []
        for document in all_files_in_data:
            if document.endswith(".pdf"):
                raw_text_from_document = read_pdf(os.path.join(DATA_PATH, document))
                all_raw_text.append(raw_text_from_document)
                filenames.append(document.removesuffix('.pdf').upper())
        # Build full preprocessed data dict
        full_data_dict = {}
        for idx, raw_text in enumerate(all_raw_text):
            nlp = create_german_nlp_pipeline()
            preprocessed_dict = preprocess_text(raw_text, nlp)
            full_data_dict[filenames[idx]] = preprocessed_dict
        # Save full preprocessed data dict
        write_to_json(json_path_full_data_dict, full_data_dict)

        print(full_data_dict.keys()) # TEST

        return full_data_dict

    bool_read_from_file = True # Adjust this to False if you want to re-read the data
    if os.path.exists(json_path_full_data_dict) and bool_read_from_file:
        full_data_dict = read_from_json(json_path_full_data_dict)
    else:
        full_data_dict = new_read()
    print("Data read successfully!")

    # Initialize the full results dict
    full_results_dict = {}
    json_path_full_results_dict = os.path.join(JSON_BASE_PATH, "full_results_dict.json")

    def new_analyze():
        # Analyze lemma frequencies
        for key, value in full_data_dict.items():
            lemma_freq_top_dict = calculate_lemma_frequencies(value)
            full_results_dict[key] = lemma_freq_top_dict
        print("Lemma frequencies analyzed successfully!")

        # Analyze german sentiments
        for key, value in full_data_dict.items():
            results, sentiment_percentages = analyze_german_sentiments(value["sentences"])
            full_results_dict[key]["sentiments"] = sentiment_percentages
        print("Sentiments analyzed successfully!")

        # Extract claims
        try:
            for key, value in full_data_dict.items():
                # extracted_claims = extract_claims(value["sentences"]) # using API
                extracted_claims = extract_claims_using_finetuned_bert(value["sentences"]) # using local finetuned model
                full_results_dict[key]["extracted_claims"] = extracted_claims
        except Exception as e:
            print(f"Error extracting claims: {e}")
        print("Claims extracted successfully!")
        write_to_json(json_path_full_results_dict, full_results_dict)

        return full_results_dict

    bool_read_from_file_analyze = True # Adjust this to False if you want to re-read the data
    if os.path.exists(json_path_full_results_dict) and bool_read_from_file_analyze:
        full_results_dict = read_from_json(json_path_full_results_dict)
    else:
        full_results_dict = new_analyze()

    print("Data analyzed successfully!")

    # look https://www.textrazor.com/demo if i have time some day


    json_path_FINAL_results_dict = os.path.join(JSON_BASE_PATH, "final_results_dict.json")

    json_path_sicherheitskopie = os.path.join(JSON_BASE_PATH, "sicherheitskopie.json")

    FINAL_results_dict = {}
    # Fact-checking
    def new_fact_checking():
        for key, value in full_results_dict.items():

            claims_list = value.get("extracted_claims", [])
            claim_classification_list = fact_checking_zyla_RAW(claims_list)

            full_results_dict[key]["fact_checks"] = claim_classification_list

            fact_checks_only = {
                    key: {"fact_checks": full_results_dict[key]["fact_checks"]}
                    for key in full_results_dict
                }
            # make safety copy
            write_to_json(json_path_sicherheitskopie, fact_checks_only)
        write_to_json(json_path_full_results_dict, full_results_dict)
        # write_to_json(json_path_FINAL_results_dict, FINAL_results_dict)

    bool_read_from_file_fact_checking = True # Adjust this to False if you want to re-read the data
    if os.path.exists(json_path_full_results_dict) and bool_read_from_file_fact_checking:
        full_results_dict = read_from_json(json_path_full_results_dict)
    else:
        new_fact_checking()

    print("Building final results")
    # Build final results
    for key in full_results_dict.keys():
        fact_checks_data = full_results_dict[key]["fact_checks"]
        full_results_dict[key]["fact_checks_percentage"] = calculate_classification_percentages_zyla_output(fact_checks_data)

    ## Build the final results dict ##
    # List of fields to copy
    fields_to_copy = ["lemmas_top_overall", "lemmas_top_per_category", "sentiments", "fact_checks_percentage"]

    for key in full_results_dict.keys():
        if key not in FINAL_results_dict:
            FINAL_results_dict[key] = {}
        for field in fields_to_copy:
            # Copy with default value as None if field is missing
            FINAL_results_dict[key][field] = full_results_dict[key].get(field, None)

    write_to_json(json_path_FINAL_results_dict, FINAL_results_dict)
