from decision_helper_lib import *
from api_communication_workflow import *
from finetune_bert_for_classi import *
import os

def process_data_flow(flow_name, extract_claims_fn, fact_checking_fn, calculate_percentages_fn):

    ###########################
    DATA_PATH = "data"
    ###########################

    all_files_in_data = os.listdir(DATA_PATH)

    JSON_BASE_PATH = f"saved_on_harddrive_{flow_name}"
    os.makedirs(JSON_BASE_PATH, exist_ok=True)

    json_path_full_data_dict = os.path.join(JSON_BASE_PATH, "full_data_dict.json")
    json_path_full_results_dict = os.path.join(JSON_BASE_PATH, "full_results_dict.json")
    json_path_FINAL_results_dict = os.path.join(JSON_BASE_PATH, "final_results_dict.json")
    json_path_sicherheitskopie = os.path.join(JSON_BASE_PATH, "sicherheitskopie.json")

    def read_data():
        all_raw_text = []
        filenames = []
        for document in all_files_in_data:
            if document.endswith(".pdf"):
                raw_text = read_pdf(os.path.join(DATA_PATH, document))
                all_raw_text.append(raw_text)
                filenames.append(document.removesuffix('.pdf').upper())

        full_data_dict = {
            filenames[idx]: preprocess_text(raw_text, create_german_nlp_pipeline())
            for idx, raw_text in enumerate(all_raw_text)
        }
        write_to_json(json_path_full_data_dict, full_data_dict)
        return full_data_dict

    def analyze_data(full_data_dict):
        full_results_dict = {}

        for key, value in full_data_dict.items():
            lemma_freq_top_dict = calculate_lemma_frequencies(value)
            sentiments = analyze_german_sentiments(value["sentences"])[1]

            full_results_dict[key] = {
                "lemmas_top_overall": lemma_freq_top_dict.get("lemmas_top_overall"),
                "lemmas_top_per_category": lemma_freq_top_dict.get("lemmas_top_per_category"),
                "sentiments": sentiments,
            }

            try:
                extracted_claims = extract_claims_fn(value["sentences"])
                full_results_dict[key]["extracted_claims"] = extracted_claims
            except Exception as e:
                print(f"Error extracting claims for {key}: {e}")

        write_to_json(json_path_full_results_dict, full_results_dict)
        return full_results_dict

    def perform_fact_checking(full_results_dict):
        for key, value in full_results_dict.items():
            claims_list = value.get("extracted_claims", [])
            fact_checks = fact_checking_fn(claims_list)
            full_results_dict[key]["fact_checks"] = fact_checks

            # Safely write backup without KeyError
            backup_data = {
            k: {"fact_checks": v.get("fact_checks")}
            for k, v in full_results_dict.items()
            }
            write_to_json(json_path_sicherheitskopie, backup_data)

        write_to_json(json_path_full_results_dict, full_results_dict)

    def build_final_results(full_results_dict):
        FINAL_results_dict = {}

        for key, value in full_results_dict.items():
            fact_checks_data = value.get("fact_checks", [])
            fact_checks_percentage = calculate_percentages_fn(fact_checks_data)

            FINAL_results_dict[key] = {
                "lemmas_top_overall": value.get("lemmas_top_overall"),
                "lemmas_top_per_category": value.get("lemmas_top_per_category"),
                "sentiments": value.get("sentiments"),
                "fact_checks_percentage": fact_checks_percentage,
            }

        write_to_json(json_path_FINAL_results_dict, FINAL_results_dict)

    # Main flow logic
    bool_read_from_file = True
    bool_read_from_file_analyze = True
    bool_read_from_file_fact_checking = True

    if os.path.exists(json_path_full_data_dict) and bool_read_from_file:
        full_data_dict = read_from_json(json_path_full_data_dict)
    else:
        full_data_dict = read_data()

    print("Data read successfully!")

    if os.path.exists(json_path_full_results_dict) and bool_read_from_file_analyze:
        full_results_dict = read_from_json(json_path_full_results_dict)
    else:
        full_results_dict = analyze_data(full_data_dict)

    print("Data analyzed successfully!")

    if not (os.path.exists(json_path_FINAL_results_dict) and bool_read_from_file_fact_checking):
        perform_fact_checking(full_results_dict)
        print("Fact-checking completed!")
        build_final_results(full_results_dict)
        print("Final results built successfully!")

# Define specific flows
if __name__ == "__main__":
    # process_data_flow(
    #     flow_name="test",
    #     extract_claims_fn=extract_claims,
    #     fact_checking_fn=fact_checking_facticity_RAW,
    #     calculate_percentages_fn=calculate_classification_percentages_facticity_output,
    # )

    # process_data_flow(
    #     flow_name="zyla",
    #     extract_claims_fn=extract_claims_using_finetuned_bert,
    #     fact_checking_fn=fact_checking_zyla_RAW,
    #     calculate_percentages_fn=calculate_classification_percentages_zyla_output,
    # )

    process_data_flow(
        flow_name="facticity_2",
        extract_claims_fn=extract_claims,
        fact_checking_fn=fact_checking_facticity_RAW,
        calculate_percentages_fn=calculate_classification_percentages_facticity_output,
    )
