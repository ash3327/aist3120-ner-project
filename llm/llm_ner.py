import sys
sys.path.append('.')

import json
import re
import os
import yaml
import time

import openai

from libs import NER
from eval import Eval
from llm.rate_limiter import RateLimiter

class llmNER(NER):
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, rate: float = 1.0, few_shot: bool = True):
        """
        Initialize the LLMNER class with the specified model name.

        Args:
            model_name (str): The name of the language model to use for NER.
            api_key (str, optional): The API key for the LLM service.
            base_url (str, optional): The base URL for the LLM service.
            rate (float, optional): Maximum number of API calls per second (default: 1.0)
            few_shot (bool, optional): Whether to use few-shot learning (default: True)
        """
        self.model_name = model_name
        self.few_shot = few_shot

        if api_key is None:
            # Load configuration from config.yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                api_key = config.get("openai", {}).get("api_key")
                base_url = config.get("openai", {}).get("base_url")
            except (FileNotFoundError, yaml.YAMLError) as e:
                print(f"Warning: Could not load config file, using environment variables: {e}")
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("OPENAI_API_BASE_URL")

        self.llm = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.ner_tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        self.rate_limiter = RateLimiter(rate)
        self.fail_count = 0

    def get_entities(self, tokens: list[str]) -> list:
        """
        Use the language model to extract entities from the given tokens.

        Args:
            tokens (list): List of tokens/words

        Returns:
            list: List of [label, text] tuples
        """
        # Prepare the prompt for the language model
        system_prompt = """
        You are a named entity recognition (NER) system. Your task is to identify and classify named entities in the text.
        The entities can be of the following types: PERSON, ORGANIZATION, LOCATION, and MISCELLANEOUS.
        You will receive a text input, and you need to return the entities in json format:
        {
            "entities": [
                {"label": "PERSON", "text": "entity_name"},
                {"label": "ORGANIZATION", "text": "entity_name"},
                {"label": "LOCATION", "text": "entity_name"},
                {"label": "MISCELLANEOUS", "text": "entity_name"}
                ...
            ]
        }

        """

        example_prompt = """
        Below are some more example inputs and outputs:
        text: "A Mujahideen Khalq statement said its leader Massoud Rajavi met in Baghdad the Secretary-General of the Kurdistan Democratic Party of Iran ( KDPI ) Hassan Rastegar on Wednesday and voiced his support to Iran 's rebel Kurds ."
        output: {
            "entities": [
                {"label": "ORGANIZATION", "text": "Mujahideen Khalq"},
                {"label": "PERSON", "text": "Massoud Rajavi"},
                {"label": "LOCATION", "text": "Baghdad"},
                {"label": "ORGANIZATION", "text": "Kurdistan Democratic Party of Iran"},
                {"label": "ORGANIZATION", "text": "KDPI"},
                {"label": "PERSON", "text": "Hassan Rastegar"},
                {"label": "LOCATION", "text": "Iran"},
                {"label": "MISCELLANEOUS", "text": "Kurds"}
            ]
        }

        text: "Ireland midfielder Roy Keane has signed a new four-year contract with English league and F.A. Cup champions Manchester United ."
        output: {
            "entities": [
                {"label": "LOCATION", "text": "Ireland"},
                {"label": "PERSON", "text": "Roy Keane"},
                {"label": "MISCELLANEOUS", "text": "English"},
                {"label": "MISCELLANEOUS", "text": "F.A. Cup"},
                {"label": "ORGANIZATION", "text": "Manchester United"}
            ]
        }

        text: "I have stayed out of the way and let them get on with the job ."
        output: {
            "entities": []
        } 

        text: "TENNIS - AUSTRALIANS ADVANCE AT CANADIAN OPEN ."
        output: {
            "entities": [
                {"label": "MISCELLANEOUS", "text": "AUSTRALIANS"},
                {"label": "MISCELLANEOUS", "text": "CANADIAN OPEN"}
            ]
        }   
        """ if self.few_shot else ""

        system_end_prompt = """
        Extract the entities from the text. Reply only with the specified JSON format. Do not include any additional text, code or explanations.

        text:
        """

        text = " ".join(tokens)

        # Apply rate limiting before making the API call
        self.rate_limiter.wait()

        try:
            # Check if we're using a Gemma model (which doesn't support system messages)
            if "gemma" in self.model_name.lower():
                # For Gemma models, combine system and user prompts into a single user message
                combined_prompt = f"{system_prompt}\n{example_prompt}\n{system_end_prompt}\n{text}"
                messages = [
                    {"role": "user", "content": combined_prompt}
                ]
            else:
                # For other models, use separate system and user messages
                messages = [
                    {"role": "system", "content": system_prompt + example_prompt + system_end_prompt},
                    {"role": "user", "content": text}
                ]

            # Send the prompt to the language model and get the response
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            
            # Extract the content from the response
            response_content = response.choices[0].message.content
            
            # Parse the JSON from the response
            try:
                # Extract JSON from potential text wrapping
                json_match = re.search(r'{.*}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    entity_data = json.loads(json_str)
                else:
                    entity_data = json.loads(response_content)
                    
                # Convert to the required format: list of [label, text] tuples
                entities = []
                for entity in entity_data.get("entities", []):
                    label = entity.get("label")
                    text = entity.get("text")
                    
                    # Map the label to the expected format (PER, ORG, etc.)
                    if label == "PERSON":
                        tag = "PER"
                    elif label == "ORGANIZATION":
                        tag = "ORG"
                    elif label == "LOCATION":
                        tag = "LOC"
                    elif label == "MISCELLANEOUS":
                        tag = "MISC"
                    else:
                        continue
                    
                    # Add the entity as a pair [tag, text]
                    entities.append((tag, text))
                
                return entities
                
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response: {response_content}")
                self.fail_count += 1
                return []
            except Exception as e:
                print(f"Error processing entities: {e}")
                self.fail_count += 1
                return []
                
        except Exception as e:
            print(f"API Error: {e}")
            self.fail_count += 1
            return []


def main(dataset="conll", split="test"):
    # Load configuration from config.yaml
    api_key = None
    base_url = None
    model_name = "gemma-3-27b-it"
    rate_limit = 0.4

    # Create NER comparison object with rate limiting
    ner_compare = llmNER(model_name=model_name, api_key=api_key, base_url=base_url, rate=rate_limit)

    Eval.evaluate_dataset(ner_compare, dataset, split)
    print("\n" + "=" * 50)
    print(f"Failed API calls: {ner_compare.fail_count}")

if __name__ == "__main__":
    main(dataset="conll", split="test")




