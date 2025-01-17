from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertForSequenceClassification, pipeline
import numpy as np
from sklearn.metrics import f1_score


class ClaimClassifierTrainer:
    def __init__(self, model_name="google-bert/bert-base-german-cased", max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.trainer = None
        self.tokenized_dataset = None

    def prepare_dataset(self, data):
        """
        Prepares the dataset by tokenizing and splitting it.
        Args:
            data (dict): Dictionary with 'text' and 'label' keys.
        Returns:
            tokenized_dataset: Tokenized and split dataset.
        """
        dataset = Dataset.from_dict(data)

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_length)

        self.tokenized_dataset = dataset.map(tokenize_function, batched=True)
        self.tokenized_dataset = self.tokenized_dataset.train_test_split(test_size=0.1)


    def set_up_model_and_trainer(self, output_dir="./results", learning_rate=2e-5, batch_size=4, epochs=3):
        """
        Sets up the model and trainer.
        Args:
            output_dir (str): Directory to save results.
            learning_rate (float): Learning rate for training.
            batch_size (int): Batch size per device.
            epochs (int): Number of training epochs.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)



        # Set up TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
        )

        # Initialize Trainer with the custom compute_loss function
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        """Trains the model."""
        if not self.trainer:
            raise ValueError("Trainer has not been set up. Call set_up_model_and_trainer() first.")
        self.trainer.train()

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predictions, average='binary')
        return {'f1': f1}

    def evaluate(self):
        """Evaluates the model."""
        if not self.trainer:
            raise ValueError("Trainer has not been set up. Call set_up_model_and_trainer() first.")
        return self.trainer.evaluate()

    def predict(self, new_data=None):
        """
        Generates predictions on either the test set or new data.
        Args:
            new_data (list, optional): A list of strings (sentences) to predict on. Defaults to None.
        Returns:
            predictions: Model predictions for the given data.
        """
        if new_data:
            # Tokenize the new data
            tokenized_new_data = self.tokenizer(new_data, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            # Generate predictions
            outputs = self.model(**tokenized_new_data)
            predictions = outputs.logits.argmax(dim=-1).tolist()
        else:
            # Predict on the test set
            if not self.trainer:
                raise ValueError("Trainer has not been set up. Call set_up_model_and_trainer() first.")
            predictions = self.trainer.predict(self.tokenized_dataset["test"]).predictions.argmax(axis=-1).tolist()

        return predictions

    def save_model(self, model_dir="./claim_classifier"):
        """
        Saves the trained model and tokenizer.
        Args:
            model_dir (str): Directory to save the model and tokenizer.
        """
        if not self.trainer:
            raise ValueError("Trainer has not been set up. Call set_up_model_and_trainer() first.")
        self.trainer.save_model(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    def load_for_inference(self, model_dir="./claim_classifier"):
        """
        Loads a model and tokenizer for inference.
        Args:
            model_dir (str): Directory where the model and tokenizer are saved.
        Returns:
            model, tokenizer: Loaded model and tokenizer.
        """
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)


    @staticmethod
    def pipeline_inference(model_dir="./claim_classifier"):
        """
        Creates a pipeline for text classification using a saved model.
        Args:
            model_dir (str): Directory where the model and tokenizer are saved.
        Returns:
            pipeline: A Hugging Face pipeline for text classification.
        """
        return pipeline("text-classification", model=model_dir)


def extract_claims_using_finetuned_bert(data_to_predict_on, batch_size = 5):
    # Initialize the trainer and load it for inference
    trainer = ClaimClassifierTrainer()
    trainer.load_for_inference()

    # Function to process data in batches
    def batch_predict(data, batch_size):
        predictions = []
        for i in range(0, len(data), batch_size):
            print("Processing batch", i // batch_size + 1, "of ", len(data) // batch_size + 1)
            batch = data[i:i + batch_size]
            batch_predictions = trainer.predict(new_data=batch)
            predictions.extend(batch_predictions)
        return predictions

    # Perform batch prediction
    predictions = batch_predict(data_to_predict_on, batch_size)

    # Filter sentences classified as 1
    extracted_claims = [sentence for sentence, prediction in zip(data_to_predict_on, predictions) if prediction == 1]

    return extracted_claims


if __name__ == "__main__":
    import pandas as pd
    # # Load the dataset
    # first_test_df = pd.read_csv("labeled_sentences_first_621.csv")
    # first_test_df_subset = first_test_df[['sentence', 'label_numeric']]
    # first_test_df_subset_renamed = first_test_df_subset.rename(columns={'sentence': 'text', 'label_numeric': 'label'})
    # first_test_df_subset_as_dict = first_test_df_subset_renamed.to_dict(orient="list")

    # # Initialize the trainer
    # trainer = ClaimClassifierTrainer()

    # # Prepare the dataset
    # trainer.prepare_dataset(first_test_df_subset_as_dict)

    # # Set up model and trainer
    # trainer.set_up_model_and_trainer()

    # # Train the model
    # trainer.train()

    # # Evaluate the model
    # metrics = trainer.evaluate()
    # print("Evaluation metrics:", metrics)

    # # Predict on the test set
    # predictions = trainer.predict()
    # print("Predictions:", predictions)


    # # Save the model
    # trainer.save_model()
    # exit()

    # # Inference using the pipeline
    # text_pipeline = ClaimClassifierTrainer.pipeline_inference(model_dir="./claim_classifier")
    # result = text_pipeline("Das ist eine Behauptung.")
    # print("Pipeline prediction:", result)

    # Initialize the trainer
    trainer = ClaimClassifierTrainer()
    trainer.load_for_inference()
    predictions = trainer.predict(new_data=["Das ist eine Behauptung.", "Die Erde ist flach.", "Der Klimawandel wird durch menschliche Aktivitäten verursacht.", "Hallo, zusammen, wie geht es euch?"])
    print("Predictions:", predictions)
