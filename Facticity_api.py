import requests
from typing import Dict, Any

class FacticityAPI:
    def __init__(self, api_key: str, base_url: str = "https://api.facticity.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    def fact_check(self, query: str, version: str = "v3", timeout: int = 60, mode: str = "sync") -> Dict[str, Any]:
        """
        Initiates a fact-checking request.

        Args:
            query (str): The text to be fact-checked.
            version (str): API version to use ("v3" or "v2"). Defaults to "v3".
            timeout (int): Timeout in seconds for sync mode. Defaults to 60.
            mode (str): Mode of processing: "sync" or "async". Defaults to "sync".

        Returns:
            dict: The API response as a dictionary.
        """
        url = f"{self.base_url}/fact-check"
        payload = {
            "query": query,
            "timeout": timeout,
            "mode": mode,
            "version": version,
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            return {"error": f"HTTP error occurred: {http_err}", "status_code": response.status_code}
        except Exception as err:
            return {"error": f"An error occurred: {err}"}


    def extract_claim(self, text: str, content_type: str = "text") -> Dict[str, Any]:
        """
        Extracts claims from a given text.

        Args:
            text (str): The text from which claims are to be extracted.
            version (str): API version to use ("v3" or "v2"). Defaults to "v3".

        Returns:
            dict: The API response as a dictionary.
        """
        url = f"{self.base_url}/extract-claim"
        payload = {
            "input": text,
            "content_type": content_type,
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            return {"error": f"HTTP error occurred: {http_err}", "status_code": response.status_code}
        except Exception as err:
            return {"error": f"An error occurred: {err}"}


# Example Usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from decision_helper_lib import write_to_json, read_from_json
    load_dotenv()
    api_key = os.getenv("FACT_API_KEY")
    facticity = FacticityAPI(api_key)

    ######### Claim extraction example
    claims = facticity.extract_claim(text="Arbeits- und Fachkräfte gewinnen Der Arbeits- und Fachkräftemangel bremst unsere wirtschaftliche Entwicklung. Aus demo- grafischen Gründen verschärft er sich weiter und wird zu einem echten Standortrisiko. Ei- gentlich könnten die Unternehmen mehr produzieren, doch dafür fehlt das Personal. Mit einer Fachkräfteoffensive bekämpfen wir das Problem und sorgen für mehr Produktivität. Für ausländische Fachkräfte wollen wir ein attraktiver Standort sein und lebenswerte Hei- mat werden. • Hürden aus dem Weg räumen. Vor allem Frauen in Teilzeit sind eine Gruppe mit großem Potenzial für den Arbeitsmarkt. Es braucht bessere Rahmenbedingungen für Vollzeitar- beit oder vollzeitnahe Arbeit. • Haushaltsnahe Dienstleistungen stärken. Wir verbessern die steuerliche Absetzbarkeit haushaltsnaher Dienstleistungen. • Berufsabschluss nachholen. Menschen in Helfertätigkeiten fördern wir auf ihrem Weg zu einer qualifizierten Fachkraft und entwickeln Anreize zum Erwerb beruflicher Qualifikationen.")
    print(claims)
