!pip install transformers
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from tqdm import tqdm
import numpy as np

marbert_model_path = 'UBC-NLP/MARBERT'
tokenizer = AutoTokenizer.from_pretrained(marbert_model_path, from_tf=True)
marbert_model = TFAutoModel.from_pretrained(marbert_model_path, output_hidden_states=True)

remove_special_tokens=1  #change this to 0 if you want to keep the special token


def bert_tokenize(text: str) -> dict:
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=50)
    if remove_special_tokens == 1:
        # Modify the input IDs and attention mask as per your requirement
        modified_input_ids = [0 if token_id == 1 else 0 if token_id == 3 else 0 if token_id == 0 else 0 if token_id == 2 else 0 if token_id == 4 else token_id for token_id in tokens['input_ids']]
        modified_attention_mask = [0 if token_id in [1, 3, 0, 2, 4] else 1 for token_id in tokens['input_ids']]

        # Update the input IDs and attention mask in the tokens dictionary
        tokens['input_ids'] = modified_input_ids
        tokens['attention_mask'] = modified_attention_mask
    return tokens
def get_embeddings(tokens):
    ids = tf.expand_dims(tf.convert_to_tensor(tokens['input_ids']), 0)
    mask = tf.expand_dims(tf.convert_to_tensor(tokens['attention_mask']), 0)
    type_ids = tf.expand_dims(tf.convert_to_tensor(tokens['token_type_ids']), 0)
    hidden_states = marbert_model(input_ids=ids, attention_mask=mask, token_type_ids=type_ids)[0]
    return hidden_states.numpy()


sample_text = "This is a sample text for testing."

tokens = bert_tokenize(sample_text)
sample_embedding = get_embeddings(tokens)

print("Sample embedding:", sample_embedding)