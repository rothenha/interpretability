import re
import numpy as np
import json
import typer
import tqdm
from typing import List, Optional

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

    def __init__(self, f_Corpus, strSentenceTag, nMaxSentenceLength, lstVocab):
        self.f_corpus = f_Corpus
        self.strSentenceTag = strSentenceTag
        self.nMaxSentenceLength = nMaxSentenceLength
        self.lstSentenceData = self.process_corpus()
        print(f"shuffling corpus sentences")
        np.random.shuffle(self.lstSentenceData)
        self.mapWordSentenceIndices = self.indexSentences(set(lstVocab), self.lstSentenceData)
        # typer.secho(f"index: {self.mapWordSentenceIndices}", fg=typer.colors.MAGENTA)

    def indexSentences(self, setVocab, lstSentenceData: List[SentenceData]):
        print(f"indexing corpus for the defined vocabulary")
        mapWordSentenceIndices = {}
        for nSentenceIndex, sentenceData in enumerate(lstSentenceData):
            setWordsSeenInSentence = set()
            for nWordIndex, strWord in enumerate(sentenceData.lstWords):
                if strWord in setVocab:
                    if strWord not in mapWordSentenceIndices:
                        mapWordSentenceIndices[strWord] = [ (nSentenceIndex, sentenceData.lstPOSs[nWordIndex]) ]
                    else:
                        # only add first occurrence of a word
                        if strWord not in setWordsSeenInSentence:
                            mapWordSentenceIndices[strWord].append( (nSentenceIndex, sentenceData.lstPOSs[nWordIndex]) )

        return mapWordSentenceIndices
    
    def getSentenceDataForWord(self, strWord: str, nMaxCount: int = -1):
        if strWord not in self.mapWordSentenceIndices:
            return None
        
        lstSentenceIndexAndPOS = self.mapWordSentenceIndices[strWord]

        lstSentencesWithPOS = []
        for nSentenceIndex, strPOS in lstSentenceIndexAndPOS:
            lstSentencesWithPOS.append( {
                "sentence": " ".join(self.lstSentenceData[nSentenceIndex].lstWords),
                "pos": strPOS} )
            if nMaxCount > 0 and len(lstSentencesWithPOS) == nMaxCount:
                break

        return lstSentencesWithPOS

    def isTagLine(self, strLine):
        return re.match(r'^</?(\S+).*>$', strLine)				
   
    def process_corpus(self):
        self.id_map = {}
        lstSentenceData = []
        strSentenceStartPattern = r'^<{}( .*)?>$'.format(self.strSentenceTag)
        strSentenceEndPattern = r'^</{}>$'.format(self.strSentenceTag)

        with open(self.f_corpus) as f_corpus:
            isInSentence = False
            lstTokens = []
            lstPOSs = []
            for strLine in f_corpus:
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

                            if len(lstTokens) <= self.nMaxSentenceLength:
                                lstSentenceData.append(SentenceData(lstTokens, lstPOSs))
                    else:
                        # typer.secho(f"strLine: {strLine}", fg=typer.colors.MAGENTA)
                        strWord, strPOS = strLine.split("\t")
                        lstTokens.append(strWord)
                        lstPOSs.append(strPOS)

        return lstSentenceData

def main(
    f_corpus: Optional[typer.FileText] = typer.Argument(None, help="decoded and CWB corpus file", metavar="DECODED_CORPUS"),
    f_vocab: typer.FileText = typer.Option(..., help="file in json format with list if words for which to extract sentences", metavar="JSON_FILE")
):
    """
    Get sentence data from CWB CORPUS.
    """

    if f_corpus == None:
        f_corpus = open("/dev/stdin")

    lstWords = json.load(f_vocab)
    
    vrtSentenceProvider = VRTSentenceProvider(f_corpus, "s", 40, lstWords)
    
    nLast = len(lstWords)-1
    print("[")
    for i in enumerate(tqdm(lstWords)):
        strWord = lstWords[i]
        lstSentenceData = vrtSentenceProvider.getSentenceDataForWord(
            strWord, nMaxCount=1000
        )
        if i < nLast:
            print(f"{json.dumps(lstSentenceData, indent=4)},")
        else:
            print(f"{json.dumps(lstSentenceData, indent=4)}")
    print("]")

if __name__ == "__main__":
    typer.run(main)