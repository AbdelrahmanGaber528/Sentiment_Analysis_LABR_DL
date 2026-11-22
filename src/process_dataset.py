import re
from re import Pattern
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download("stopwords", quiet=True)

ARABIC_STOPWORDS: set[str] = set[str](stopwords.words("arabic"))


# Patterns
URL_PATTERN: Pattern[str] = re.compile(r"https?://\S+|www\.\S+")
IP_PATTERN: Pattern[str] = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
EMOJI_PATTERN: Pattern[str] = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\u2600-\u26FF\u2700-\u27BF\u200D\uFE0F]")
PUNCT_PATTERN: Pattern[str] = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~\u060C\u061B\u061F\u0640\u00AB\u00BB]")
SYMBOL_PATTERN: Pattern[str] = re.compile(r"[\u00A2-\u00BF\u20A0-\u20CF\u2190-\u21FF\u2200-\u22FF\u064B-\u065F\u0670]")


TOKENIZER : RegexpTokenizer = RegexpTokenizer(pattern=r"\S+")


def process_comment(comment: str) -> list[str]:

    text : str = comment.lower()
    
    text = URL_PATTERN.sub(" رابط ", comment)
    text = IP_PATTERN.sub(" عنوان ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = PUNCT_PATTERN.sub(" ", text)
    text = SYMBOL_PATTERN.sub(" ", text)
    
    tokens: list[str] = TOKENIZER.tokenize(text)

    return [token for token in tokens if token not in ARABIC_STOPWORDS]



def build_vocab(tokenized_reviews : list[list[str]])-> dict[str , int]:
    
    vocab : dict[str , int] = {
        "PAD" : 0,
        "UNK" : 1
    }
    
    for tokens in tokenized_reviews :
        for token in tokens :
            if token not in vocab :
                vocab[token] = len(vocab)
    
    return vocab



# following methods for one rewview comment : list[str] tokens(words) , list[int] -> ids for one list

def tokens_to_ids(tokens : list[str], vocab : dict[str , int])-> list[int]: 
    return [vocab.get(token , vocab["UNK"]) for token in tokens]


def pad_or_truncate(ids : list[int], max_length : int, pad_id : int = 0 ) -> list[int]:
    
    if len(ids) > max_length:
        return ids[:max_length]

    return ids + [pad_id] * (max_length - len(ids))