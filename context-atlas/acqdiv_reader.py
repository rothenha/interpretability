import sqlite3
import typer
import json
import pickle
from tqdm import tqdm
from typing import Any, List, Dict

class ChintangUtterances():
    conn_sqlite = sqlite3.connect("data/beta_acqdiv_corpus_2020-11-30.sqlite3")
    cursor_acqdiv_db = conn_sqlite.cursor()

    lstUtterances = None
    mapToken2Utterances = None
    setPOSValues = set()

    def __init__(self):
        lstUtterances = []
        n_total = 0
        for row in self.cursor_acqdiv_db.execute("SELECT utterances.id, session_id_fk, utterance_raw, gloss_raw, pos_raw, morpheme FROM utterances INNER JOIN sessions ON utterances.session_id_fk = sessions.id where sessions.corpus = 'Chintang'"):
            n_total += 1
            if row[2] != None and row[4] != None:
                lstTokens = row[2].split()
                lstPOSRaw = [ strPOS for strPOS in row[4].split() ]
                lstMorphemes = [ strMorpheme for strMorpheme in row[5].split() ]
                lstPOSs = []

                lstModifiedPOS = self.sanitize_hyphens(row, lstPOSRaw)
                    
                lstPOSs = [ strPOS for strPOS in lstModifiedPOS if not strPOS.startswith("-") and not strPOS.endswith("-") ]
                self.setPOSValues.update(lstPOSs)

                mapUtterance = { "id": f"{row[1]}-{row[0]}", "utterance_raw": f"{row[2]}", "token_list": lstTokens, "gloss_raw": f"{row[3]}", "pos_raw": f"{row[4]}", "pos_list": lstPOSs, "morpheme_list": lstMorphemes}

                # if len(lstMorphemes) == len(lstPOSRaw) and len(lstTokens) != len(lstPOSs):
                #     print(mapUtterance)
                if len(lstTokens) == len(lstPOSs):
                    lstUtterances.append(mapUtterance)
                else:
                    pass
            
        n_invalid = n_total-len(lstUtterances)
        self.lstUtterances = lstUtterances
        self.mapToken2Utterances = self.invertUtteranceList(self.lstUtterances)

        typer.secho(f"{n_invalid} utterances out of {n_total} had invalid pos tags corresponding to {100*n_invalid/n_total:.2f}%", fg=typer.colors.MAGENTA, err=True)
        self.write_pos_config_file()

    def invertUtteranceList(self, lstUtterances: List[Dict[str,Any]]):
        mapWordInfo = {}
        mapToken2Utterances = {}
        for i, mapUtterance in tqdm(enumerate(lstUtterances)):
            lstTokens = mapUtterance["token_list"]

            for nTokenIndex, strToken in enumerate(lstTokens):
                if strToken not in mapToken2Utterances:
                    mapToken2Utterances[strToken] = [ i ]
                    mapWordInfo[strToken] = {}
                    mapWordInfo[strToken]["n"] = 1
                    mapWordInfo[strToken]["pos_set"] = set()
                    mapWordInfo[strToken]["pos_set"].add(mapUtterance["pos_list"][nTokenIndex])
                else:
                    mapToken2Utterances[strToken].append(i)
                    mapWordInfo[strToken]["n"] += 1
                    mapWordInfo[strToken]["pos_set"].add(mapUtterance["pos_list"][nTokenIndex])

        self.lstTokenStats = []
        for strToken, lstIndices in mapToken2Utterances.items():
            self.lstTokenStats.append( (strToken, len(lstIndices)) )
        self.lstTokenStats.sort(key=lambda tupStat: -tupStat[1])

        typer.secho(f"Found {len(mapToken2Utterances)} distinct tokens with a frequency range of [{self.lstTokenStats[-1][1]} ({self.lstTokenStats[-1][0]}), {self.lstTokenStats[0][1]} ({self.lstTokenStats[0][0]})]", fg=typer.colors.BLUE, err=True)
        typer.secho(f"Saving vocabulary info pickle to 'chintang_vocab.p'", fg=typer.colors.BLUE, err=True)

        with open("chintang_vocab.p", "wb") as f:
            pickle.dump(mapWordInfo, f)

        return mapToken2Utterances

    def getAllUtterances(self) -> List[str]:
        lstUtterances = []

        for mapUtterance in self.lstUtterances:
            lstUtterances.append(mapUtterance["utterance_raw"])

    def getUtterancesForWord(self, strToken):
        lstUtterances = []
        if strToken in self.mapToken2Utterances:
            lstUtterances = [self.lstUtterances[i] for i in self.mapToken2Utterances[strToken]]

        return lstUtterances

    def sanitize_hyphens(self, row, lstPOSRaw):
        n_raw_pos = len(lstPOSRaw)
        lstHyphenIndices = [i for i, strPOS in enumerate(lstPOSRaw) if strPOS == "-"]
        lstModifiedPOS = list(lstPOSRaw)

        for nHyphenIndex in lstHyphenIndices[::-1]:
            if nHyphenIndex < n_raw_pos-2:
                if lstPOSRaw[nHyphenIndex+1] == "gm":
                    if lstPOSRaw[nHyphenIndex-1] != "gm":
                        lstModifiedPOS[nHyphenIndex] = "-gm"
                        del lstModifiedPOS[nHyphenIndex+1]
                    # else:
                    #     print(f"+++ utterance: {row[2]}, pos: {row[4]}, morpheme: {row[5]}")
                elif lstPOSRaw[nHyphenIndex-1] == "gm":
                    lstModifiedPOS[nHyphenIndex-1] = "gm-"
                    del lstModifiedPOS[nHyphenIndex]

        return lstModifiedPOS
    
    def write_pos_config_file(self):
        typer.secho(f"writing pos config file 'pos.out'", fg=typer.colors.BLUE, err=True)
        lstPOSValues = []
        lstSimplePOSValues = []
        for strPOSValue in self.setPOSValues:
            lstPOSValues.append({"tag": strPOSValue, "description": strPOSValue, "dispPos": strPOSValue})
            lstSimplePOSValues.append({"tag": strPOSValue, "description": strPOSValue})
    
        strPOSValues = json.dumps(lstPOSValues, indent=4)
        strSimplePOSValues = json.dumps(lstSimplePOSValues, indent=4)
        
        with open("pos.out", "w") as f:
            f.write("""
            export interface POSTag {
                tag: string;
                description: string;
                dispPos?: string;
            }

            export const POS: POSTag[] =
            """ + strPOSValues + \
            """

            export const SimplePOS: POSTag[] =
            """ + strSimplePOSValues)

if __name__ == "__main__":
    theApp = ChintangUtterances()

