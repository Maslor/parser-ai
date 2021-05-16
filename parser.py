import string

import nltk

nltk.download('punkt')
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP Conj S | NP VP Conj VP | NP VP P NP | S NP | S Conj S | S Conj VP
NP -> N | Adj NP | Det NP | NP P NP | NP Adv
VP -> V | VP Adv | Adv VP | VP P
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    tokens = nltk.word_tokenize(sentence.lower().translate(str.maketrans('', '', string.punctuation)))
    for token in tokens:
        if not token.isalpha() or len(token) < 1:
            tokens.remove(token)
    return tokens


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_chunks = []
    path = []
    traverse(tree, np_chunks)
    return np_chunks


def traverse(tree, np_chunks):
    if tree.label() == "NP" and len(list(tree.subtrees(filter=lambda t: t.label() == "NP"))) == 1:
        if tree not in np_chunks:
            np_chunks.append(tree)
    else:
        for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
            if subtree == tree:
                continue
            traverse(subtree, np_chunks)


if __name__ == "__main__":
    main()
