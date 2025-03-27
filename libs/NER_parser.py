from abc import ABC, abstractmethod

class NER(ABC):
    @abstractmethod
    def get_entities(self, tokens:list[str])->list:
        """
        Compare NER labels between dataset and spaCy.
        
        Args:
            tokens (list): List of tokens/words
            
        Returns:
            list: List of [label, text] tuples
        """
        pass