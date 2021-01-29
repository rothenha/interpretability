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
from subprocess import PIPE
from sys import stdout
from typing import List, Optional

import os
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers.modeling_utils import PreTrainedModel
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import re
import numpy as np

# import umap
import json
from tqdm import tqdm
import typer
import subprocess
import umap

MIN_SENTENCES = 20


class SentenceData(object):
    def __init__(self, lstWords, lstPOSs):
        self.lstWords = lstWords
        self.lstPOSs = lstPOSs

    def getPOSForFirstOccurrenceOfWord(self, strWord):
        nIndexWord = self.lstWords.index(strWord)
        if nIndexWord >= 0:
            return self.lstPOSs[nIndexWord]
        else:
            return nIndexWord


class VRTSentenceProvider:
    id_map = {}

    def __init__(self, f_Corpus, strSentenceTag, nMinSentenceLength, nMaxSentenceLength, lstVocab):
        self.f_corpus = f_Corpus
        self.strSentenceTag = strSentenceTag
        self.nMaxSentenceLength = nMaxSentenceLength
        self.nMinSentenceLength = nMinSentenceLength
        self.lstSentenceData = self.process_corpus()
        typer.secho(f"shuffling corpus sentences", fg=typer.colors.MAGENTA, err=True)
        np.random.shuffle(self.lstSentenceData)
        self.mapWordSentenceIndices = self.indexSentences(
            set(lstVocab), self.lstSentenceData
        )
        # typer.secho(f"index: {self.mapWordSentenceIndices}", fg=typer.colors.MAGENTA)

    def indexSentences(self, setVocab, lstSentenceData: List[SentenceData]):
        typer.secho(
            f"indexing corpus for the defined vocabulary",
            fg=typer.colors.MAGENTA,
            err=True,
        )
        mapWordSentenceIndices = {}
        for nSentenceIndex, sentenceData in enumerate(lstSentenceData):
            setWordsSeenInSentence = set()
            for nWordIndex, strWord in enumerate(sentenceData.lstWords):
                if strWord in setVocab:
                    if strWord not in mapWordSentenceIndices:
                        mapWordSentenceIndices[strWord] = [
                            (nSentenceIndex, sentenceData.lstPOSs[nWordIndex])
                        ]
                    else:
                        # only add first occurrence of a word
                        if strWord not in setWordsSeenInSentence:
                            mapWordSentenceIndices[strWord].append(
                                (nSentenceIndex, sentenceData.lstPOSs[nWordIndex])
                            )

        return mapWordSentenceIndices

    def getSentenceDataForWord(self, strWord: str, nMaxCount: int = -1):
        if strWord not in self.mapWordSentenceIndices:
            return None

        lstSentenceIndexAndPOS = self.mapWordSentenceIndices[strWord]

        lstSentencesWithPOS = []
        for nSentenceIndex, strPOS in lstSentenceIndexAndPOS:
            lstSentencesWithPOS.append(
                {
                    "sentence": " ".join(self.lstSentenceData[nSentenceIndex].lstWords),
                    "pos": strPOS,
                }
            )
            if nMaxCount > 0 and len(lstSentencesWithPOS) == nMaxCount:
                break

        return lstSentencesWithPOS

    def isTagLine(self, strLine):
        return re.match(r"^</?(\S+).*>$", strLine)

    def process_corpus(self):
        self.id_map = {}
        lstSentenceData = []
        strSentenceStartPattern = r"^<{}( .*)?>$".format(self.strSentenceTag)
        strSentenceEndPattern = r"^</{}>$".format(self.strSentenceTag)

        isInSentence = False
        lstTokens = []
        lstPOSs = []
        for strLine in self.f_corpus:
            # ignore everything outside sentence tags
            if not isInSentence:
                matchSentenceStartTag = re.match(strSentenceStartPattern, strLine)
                if matchSentenceStartTag:
                    lstTokens = []
                    lstPOSs = []
                    isInSentence = True
                # ignore everything outside sentence tags
                else:
                    continue
            else:
                strLine = strLine.strip()
                if self.isTagLine(strLine):
                    if re.match(strSentenceEndPattern, strLine):
                        isInSentence = False

                        nTokensInSentence = len(lstTokens)
                        if nTokensInSentence >= self.nMinSentenceLength and nTokensInSentence <= self.nMaxSentenceLength:
                            lstSentenceData.append(SentenceData(lstTokens, lstPOSs))
                else:
                    # typer.secho(f"strLine: {strLine}", fg=typer.colors.MAGENTA)
                    strWord, strPOS = strLine.split("\t")
                    lstTokens.append(strWord)
                    lstPOSs.append(strPOS)

        return lstSentenceData


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
    layers = range(-12, 0)
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
        sep_idxs = [-1] + [i for i, v in enumerate(tokenized_text) if v == "[SEP]"]
        segments_ids = []
        for i in range(len(sep_idxs) - 1):
            segments_ids += [i] * (sep_idxs[i + 1] - sep_idxs[i])

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        tokens_tensor = tokens_tensor.to(device)
        segments_tensors = segments_tensors.to(device)

        # Predict hidden states features for each layer
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            encoded_layers = outputs[2]
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


def get_vocab_from_cwb(
    strCorpus: str, nMaxVocabSize: int, nMinFrequency: int, strPositionalAttritbute: str
) -> List[str]:
    with subprocess.Popen(
        ["cwb-lexdecode", "-f", "-b", "-P", strPositionalAttritbute, strCorpus],
        stdout=PIPE,
    ) as procLexDecode:

        lstLexLines = []
        while True:
            strLine = procLexDecode.stdout.readline()
            if not strLine:
                break
            lstLexLines.append(strLine.decode().strip().split("\t"))

        lstLexLines.sort(key=lambda entry: int(entry[0]), reverse=True)
        typer.secho(
            f"overall vocabulary size: {len(lstLexLines)}", fg=typer.colors.BLUE
        )

        lstWords = []
        for lstWordInfo in lstLexLines:
            strWord = lstWordInfo[1]
            nFrequency = int(lstWordInfo[0])
            if len(lstWords) >= nMaxVocabSize or nFrequency < nMinFrequency:
                break

            if len(strWord) < 2 or re.match(r"^\W+$", strWord):
                # typer.secho(f"filtering out: {strWord}", fg=typer.colors.MAGENTA)
                continue

            lstWords.append(strWord)

        typer.secho(f"selected vocabulary size: {len(lstWords)}", fg=typer.colors.BLUE)

        return lstWords


def main(
    f_corpus: Optional[typer.FileText] = typer.Argument(
        None,
        help="decoded CWB corpus file (read from stdin if not given)",
        metavar="DECODED_CORPUS",
    ),
    f_vocab: typer.FileText = typer.Option(
        ...,
        "--lexicon",
        "-l",
        help="file in json format with list of words for which to extract sentences",
        metavar="JSON_FILE",
    ),
    # nMaxVocabSize: int = typer.Option(
    #     100000,
    #     "--max_vocab_size",
    #     "-m",
    #     help="maximum number of words for which to compute sense clusterings",
    # ),
    nMinFrequency: int = typer.Option(
        40,
        "--min_word_frequency",
        "-f",
        help="minimum number of occurrences of words for which to compute sense clusterings",
    ),
    strPositionalAttribute: str = typer.Option(
        "word",
        "--positional_attribute",
        "-P",
        help="the positional attribute (word, lemma) to use",
    ),
):
    """
    Compute sentence sense clusterings for words in CWB CORPUS read from decoded file or stdin.
    """

    strDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    typer.secho(f"device: {strDevice}", fg=typer.colors.BLUE)

    # Load pre-trained model tokenizer (vocabulary)
    config = BertConfig.from_pretrained(
        "bert-base-german-cased", output_hidden_states=True
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained("bert-base-german-cased", config=config)
    model.eval()
    model = model.to(strDevice)

    if f_corpus == None:
        f_corpus = open("/dev/stdin")

    lstWords = json.load(f_vocab)

    with open("static/words.json", "w") as f_words:
        json.dump(lstWords, f_words)

    vrtSentenceProvider = VRTSentenceProvider(f_corpus, "s", 10, 100, lstWords)

    for strWord in tqdm(lstWords):
        lstSentenceData = vrtSentenceProvider.getSentenceDataForWord(
            strWord, nMaxCount=1000
        )

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
    for strWord in os.listdir("static/jsons"):
        strWord = strWord.split(".")[0]
        filtered_words.append(strWord)

    with open("static/filtered_words.json", "w") as f_filtered_words:
        json.dump(filtered_words, f_filtered_words)


if __name__ == "__main__":
    typer.run(main)
