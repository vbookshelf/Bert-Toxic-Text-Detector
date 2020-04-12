# *** NOTE: This entire file will get imported into __init__.py ***
# ------------------------------------------------------------------

#from the app folder import the app object
from app import app

from flask import request
from flask import jsonify

import numpy as np

import torch

import transformers
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer

from keras.preprocessing.sequence import pad_sequences


# Define the variables
# .....................

MODEL_TYPE = "bert-base-multilingual-uncased"
CASE_BOOL = True # do_lower_case=CASE_BOOL
MAX_LEN = 256


# Helper Functions
# .................

def preprocess_for_bert(sentences, MAX_LEN):
    
    """
    Preprocesses sentences to suit BERT.
    Input:
    sentences: numpy array
    
    Output:
    Tokenized sentences, padded and truncated.
    
    
    """
    
    print('Started pre-processing...')

    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
        
    
    # Pad the token matrix
    
    # **** Issue: If the length is greater than max_len then 
    # this part will cut off the [SEP] token (102), which is
    # at the end of the long sentence.
    # It's actually the index value (number) that's associated
    # with the SEP token that's at the end.


    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    padded_input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")
    
    
    # *** This fixes the issue above.
    # Check if the SEP token was cut off and if so put it back in.
    # Check if the last index is 102. 102 is the SEP token.
    # Correct the last token if needed.
    for sent in padded_input_ids: # go row by row through the numpy 2D array.
        length = len(sent)
        
        if (sent[length-1] != 0) and (sent[length-1] != 102): # 102 is the SEP token
            sent[length-1] = 102 # set the last value to be the SEP token i.e. 102
    
    
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in padded_input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
        
        
    print('Finished pre-processing...')
    return padded_input_ids, attention_masks


def make_prediction(sentence, model):
    
    
    # convert to a list
    sentence = [sentence]
    
    # pre-process
    padded_token_list, att_mask = preprocess_for_bert(sentence, MAX_LEN)
    
    print('Zero')
    
    # convert to torch tensors
    padded_token_list = torch.tensor(padded_token_list, dtype=torch.long)
    att_mask = torch.tensor(att_mask, dtype=torch.long)
    
    print('One')
    
    # make a prediction
    outputs = model(padded_token_list, 
                token_type_ids=None, 
                attention_mask=att_mask)
    
    print('Two')
    
    # get the preds
    preds = outputs[0]
    
    # convert to probabilities
    preds_proba = torch.sigmoid(preds)
    
    print('Three')
    
    # convert to numpy
    np_preds = preds_proba.detach().cpu().numpy()
    
    # get the first row
    np_preds = np_preds[0]
    
    print('Four')
    
    # extract the probailities for each class
    not_toxic_proba = np_preds[0]
    toxic_proba = np_preds[1]
    
    print('Finished prediction...')
    return not_toxic_proba, toxic_proba
	



# Define the device
# ..................

device = 'cpu'


# Instantiate the tokenizer
# ..........................

tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE, do_lower_case=CASE_BOOL)
print('Tokenizer loaded.')
	





# Load the Model
# ...............

print('Local model is initializing...')


# Load the architecture.

# This path is to a folder and not a file.
# We are loading a saved model architecture and not downloading it.
# https://github.com/huggingface/transformers/issues/136

folder_path = '../folder-bert-base-multilingual-uncased' 
model = BertForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels = 2,
                                                output_attentions = False, 
                                                output_hidden_states = False)
	
print('Model initialization complete.')

print('Model is loading...')
# Load the saved weights into the architecture.
# Note that this file (views.py) gets imported into the app.ini file therefore,
# place the model in the same folder as the app.ini file.
path = 'model.pt'
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

print('../Model loaded.')

# Send the model to the device.
model.to(device)
	
	





@app.route('/')
def index():
	return 'Hello world - flask app'
	
	

@app.route('/test')
def test():
	return 'Testing testing...'


	
	
# If we don't send data to the code then we use GET here.
# If we are sending data with the request then we must use POST here.

# The methods parameter specifies 
# what kinds of http request are allowed for this endpoint.
# With a POST request the expectation is that the client application 
# will send data to the web server along with the request.
@app.route('/predict', methods=['POST'])
def predict():
	
	# This is the data that is received from the client
	# get_json gives us the message from the client in json
	# force=True tells flask to always parse the json from the request even if
	# it is unsure of the data type.
	message = request.get_json(force=True)
	
	# references the name key from the json key/value pair
	sentence = message['name']
	
	print(sentence)
	
	
	# Process the text and make a prediction
	# .......................................
	
	not_toxic_proba, toxic_proba = make_prediction(sentence, model)
	
	print(toxic_proba)
	
	# Define the response
	# ....................
	
	# Convert to a percentage
	toxic_proba = toxic_proba * 100
	
	# Round to 2 decimal places
	toxic_proba = np.round(toxic_proba, 2) 
	
	# Convert to type string because json cannot work with numpy float32.
	toxic_proba = str(toxic_proba)
	
	response = {
	'greeting': toxic_proba + '%'
	}
	
	# convert the python dictionary into json
	# The predict function returns a response to the client web application.
	return jsonify(response)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	