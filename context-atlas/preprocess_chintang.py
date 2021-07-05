# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Preprocessing the data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers.modeling_utils import PreTrainedModel
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoModelForMaskedLM, AutoTokenizer, DistilBertTokenizer, DistilBertForMaskedLM, DistilBertModel
from transformers import DistilBertConfig
import sqlite3 as sql
import re
import numpy as np
import umap
import json
import nltk
import typer
from acqdiv_reader import ChintangUtterances
from tqdm import tqdm

MIN_SENTENCES = 20

DB_PATH = "./enwiki-20170820.db"
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")


def neighbors(word, lstSentenceData, tokenizer, model, device):
    """Get the info and (umap-projected) embeddings about a word."""

    lstSentences = [sentenceData["sentence"] for sentenceData in lstSentenceData]
    # Get embeddings.
    points = get_embeddings(word, lstSentences, tokenizer, model, device)

    # Use UMAP to project down to 3 dimnsions.
    points_transformed = project_umap(points)

    return {"labels": lstSentenceData, "data": points_transformed}


def project_umap(points):
    """Project the words (by layer) into 3 dimensions using umap."""
    points_transformed = []
    for layer in points:
        transformed = umap.UMAP().fit_transform(layer).tolist()
        points_transformed.append(transformed)
    return points_transformed


def get_embeddings(word, sentences, tokenizer, model: PreTrainedModel, device):
    """Get the embedding for a word in each sentence."""
    # Tokenized input
    layers = range(-6, 0)
    points = [[] for layer in layers]
    typer.secho(
        f"Getting embeddings for {len(sentences)} sentences", fg=typer.colors.MAGENTA
    )

    for sentence in sentences:
        sentence = "[CLS] " + sentence + " [SEP]"
        tokenized_text = tokenizer.tokenize(sentence)

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        # should give you something like [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(device)

        # Predict hidden states features for each layer
        with torch.no_grad():
            outputs = model(tokens_tensor, output_hidden_states=True)
            encoded_layers = outputs[1]
            # typer.secho(
            #     f"encoded_layers size: {encoded_layers.shape}", fg=typer.colors.MAGENTA
            # )
            # typer.secho(
            #     f"outputs[2] size: {len(outputs[2])}", fg=typer.colors.MAGENTA
            # )
            # typer.secho(
            #     f"outputs[2]: {outputs[2]}", fg=typer.colors.MAGENTA
            # )
            encoded_layers = [l.cpu() for l in encoded_layers]

        # We have a hidden states for each of the 12 layers in model bert-base-uncased
        encoded_layers = [l.numpy() for l in encoded_layers]
        word_idx = -1
        try:
            word_idx = tokenized_text.index(word)
        # If the word is made up of multiple tokens, just use the first one of the tokens that make it up.
        except:
            for i, token in enumerate(tokenized_text):
                if token == word[: len(token)]:
                    word_idx = i

        # Reconfigure to have an array of layer: embeddings
        for l in layers:
            # typer.secho(f"word_idx = {word_idx}", fg=typer.colors.MAGENTA)
            sentence_embedding = encoded_layers[l][0][word_idx]
            points[l].append(sentence_embedding)

    points = np.asarray(points)
    return points


def getSentenceDataForWord(strWord: str, chintangUtteranceIndex, nMaxCount: int = -1):

    lstUtterances = chintangUtteranceIndex.getUtterancesForWord(strWord)
    if lstUtterances == []:
        return None

    lstSentencesWithPOS = []
    for mapUtterance in lstUtterances:
        nWordIndex = mapUtterance["token_list"].index(strWord)
        strPOS = mapUtterance["pos_list"][nWordIndex]

        lstSentencesWithPOS.append(
            {
                "sentence": mapUtterance["utterance_raw"],
                "pos": strPOS,
            }
        )
        if nMaxCount > 0 and len(lstSentencesWithPOS) == nMaxCount:
            break

    return lstSentencesWithPOS


if __name__ == "__main__":

    strDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : ", strDevice)

    # Load pre-trained model tokenizer (vocabulary)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-multilingual-cased"
    )
    # Load pre-trained model (weights)
    # model = BertModel.from_pretrained('bert-base-uncased', config=config)
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")
    strAdapterName = model.load_adapter("models/adapted/mlm/")
    model.set_active_adapters(strAdapterName)

    model.eval()
    model = model.to(strDevice)

    # Get selection of sentences from wikipedia.
    with open("static/words.json") as f:
        words = json.load(f)

    chintangUtteranceIndex = ChintangUtterances()

    for strWord in tqdm(words):
        lstSentenceData = getSentenceDataForWord(strWord, chintangUtteranceIndex)

        # And don't show anything if there are less than 20 sentences.
        if lstSentenceData == None:
            typer.secho(f"no sentences for word : {strWord}", fg=typer.colors.BLUE)
        elif len(lstSentenceData) > MIN_SENTENCES:
            typer.secho(f"starting process for word : {strWord}", fg=typer.colors.BLUE)
            locs_and_data = neighbors(
                strWord, lstSentenceData, tokenizer, model, strDevice
            )
            with open(f"static/jsons/{strWord}.json", "w") as f_word:
                json.dump(locs_and_data, f_word)
        else:
            typer.secho(
                f"too few sentences ({len(lstSentenceData)} < {MIN_SENTENCES}) for word : {strWord}",
                fg=typer.colors.BLUE,
            )

    # Store an updated json with the filtered words.
    filtered_words = []
    for word in os.listdir("static/jsons"):
        word = word.split(".")[0]
        filtered_words.append(word)

    with open("static/filtered_words.json", "w") as outfile:
        json.dump(filtered_words, outfile)
    print(filtered_words)
