""" word2num.py: transform text to numeric representation.
"""

from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm as tqdm

def transform(df, tokenizer, insert_bos=True, insert_eos=False, insert_oov=False, vocab_size=60000, min_freq=2, pad_index=0, mask_index=None):
    """
    Return tuple: vocabulary, collection of tokens, collection of IDs where
                  vocabulary:           list of words / tokens with padding, mask and out-of-vocabulary tokens inserted
                  collection of tokens: numeric representation of words of sequences
                  collection of IDs:    list of sequences IDs
    Params
    :df: Data frame.
    """
    token_oov="_oov_"
    token_pad="_pad_"
    token_bos="_bos_"
    token_eos="_eos_"
    token_mask="_mask_"

    tokens = []
    ID = []
    for index, row in tqdm(df.iterrows()):
        item_tok = tokenizer(row['sequence'])
        if(insert_bos):
            item_tok=[token_bos] + item_tok
        if(insert_eos):
            item_tok=item_tok +[token_eos]
        tokens.append(item_tok)
        ID.append(index)
    counts = Counter( (t for seq in tokens for t in seq) )
    
    # Vocabulary and interger-to-token mapping
    vocab = [t for t, freq in counts.most_common(vocab_size) if freq > min_freq]
    if insert_oov:
        vocab.append(token_oov)
    if mask_index is None:
        vocab.insert(pad_index, token_pad)
    elif pad_index < mask_index:
        vocab.insert(pad_index, token_pad)
        vocab.insert(mask_index, token_mask)
    else:
        vocab.insert(mask_index, token_mask)
        vocab.insert(pad_index, token_pad)
    
    # Make numeric representation of tokens
    word2index = defaultdict(lambda: len(vocab) if insert_oov else pad_index,
                             { word: index for index, word in enumerate(vocab)})
    tokens_num = np.array([[word2index[w] for w in seq] for seq in tokens ])
    return vocab, tokens_num, ID



