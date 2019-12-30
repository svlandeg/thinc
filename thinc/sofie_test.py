from thinc.neural.ops import NumpyOps

import numpy as np

from numpy.testing import assert_allclose


if __name__ == "__main__":
    # x_list = [np.asarray([1, 2]), np.asarray([5, 6]), np.asarray([1])]
    # x = np.asarray(x_list)
    # print("x:", x)
    # flat = NumpyOps().flatten(x, pad=3)
    # print("flat:", flat)
    # unflat = NumpyOps().unflatten(flat, [2, 2, 1], pad=3)
    # print("unflat:", unflat)
    # print()

    import spacy

    spacy.require_gpu()
    print("This is with GPU.")

    nlp = spacy.load("en_core_web_sm")
    sentences = [
        "The decrease in 2008 primarily relates to the decrease in cash and cash equivalents 1.\n",
        "The Company's current liabilities of &euro;32.6 million primarily relate to deferred income from collaborative arrangements and trade payables.\n",
        "The increase in deferred income is related to new deals with partners."
        ]

    for s in sentences:
        print()
        doc = nlp(s)
        print([tok.text for tok in doc])
        print([tok.pos_ for tok in doc])
        print([tok.dep_ for tok in doc])

    # x_list = [np.asarray([]), np.asarray([]), np.asarray([])]
    # x = np.asarray(x_list)
    # print("x:", x)
    # flat = NumpyOps().flatten(x, pad=3)
    # print("flat:", flat)
    # unflat = NumpyOps().unflatten(flat, [2, 2, 0], pad=3)
    # print("unflat:", unflat)

    # assert_allclose(x, unflat)

    # print("test")
    # imdb_data, _ = imdb()
    # print(imdb_data[0:10])
