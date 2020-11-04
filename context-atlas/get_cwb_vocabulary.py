import subprocess
import typer
import re
import json
from subprocess import PIPE
from typing import List

def get_vocab(
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
        lstLexLines = lstLexLines[100:]
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
    str_corpus: str = typer.Argument(..., help="CWB corpus name", metavar="CORPUS"),
    nMaxVocabSize: int = typer.Option(
        100000,
        "--max_vocab_size",
        "-m",
        help="maximum number of words for which to compute sense clusterings",
    ),
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
    Get vocabulary for CWB corpus to be used with for 'preprocess_cwb_corpus'. Outputs json on stdout.
    """


    # Get selection of sentences from corpus.
    lstWords = get_vocab(
        str_corpus, nMaxVocabSize, nMinFrequency, strPositionalAttribute
    )
    print(json.dumps(lstWords))
    # with open("static/words.json", "w") as f_vocab:
    #     json.dump(lstWords, f_vocab)

if __name__ == "__main__":
    typer.run(main)
