import re
import logging
import subprocess
from subprocess import PIPE
import numpy as np
import typer
from typing import List

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

    def __init__(self, strCorpus, strPositionalAttribute, strSentenceTag, nMaxSentenceLength, lstVocab):
        self.strCorpus = strCorpus
        self.strPositionalAttribute = strPositionalAttribute
        self.strSentenceTag = strSentenceTag
        self.nMaxSentenceLength = nMaxSentenceLength
        self.lstSentenceData = self.decode_and_process_corpus()
        typer.secho(f"shuffling corpus sentences", fg=typer.colors.BLUE)
        np.random.shuffle(self.lstSentenceData)
        self.mapWordSentenceIndices = self.indexSentences(set(lstVocab), self.lstSentenceData)
        # typer.secho(f"index: {self.mapWordSentenceIndices}", fg=typer.colors.MAGENTA)

    def indexSentences(self, setVocab, lstSentenceData: List[SentenceData]):
        typer.secho(f"indexing corpus for the defined vocabulary", fg=typer.colors.BLUE)
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

    def decode_and_process_corpus(self):
        self.id_map = {}
        lstSentenceData = []
        strSentenceStartPattern = r'^<{}( .*)?>$'.format(self.strSentenceTag)
        strSentenceEndPattern = r'^</{}>$'.format(self.strSentenceTag)

        strCommand = f"cwb-decode -Cx {self.strCorpus} -P {self.strPositionalAttribute} -P pos -S {self.strSentenceTag}"
        typer.secho(f"decoding corpus with the following command:\n{strCommand}", fg=typer.colors.BLUE)
        with subprocess.Popen(strCommand.split(), stdout=PIPE) as procCWBDecode:
            isInSentence = False
            lstTokens = []
            lstPOSs = []
            while True:
                outLine = procCWBDecode.stdout.readline()
                if not outLine:
                    break
                strLine = outLine.decode().strip()

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

