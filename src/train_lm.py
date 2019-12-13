import fastai
import fastai.text

def train_lm(vocab, 
             train, 
             validation, 
             workdir,
             arch,
             config,
             batch_size=96, 
             bptt=70,
             dropout=0.5,
             clip=25,
             weight_decay=1e-7,
             lr=1e-3):
    """
    :vocab:      Collection of tokens across all data set's tokens.
    :train:      Train collection of tokenized sequences, numeric - token index - representation.
    :validation: Validation collection of tokenized sequences.
    :workdir:    Path to a working directory
    :arch:       Name of network architecture.
    :config:     Dictionary with network architecture parameters
    """
    fastai_vocab = fastai.text.Vocab({index: word for index, word in enumerate(vocab)})
    src = fastai.text.ItemLists(workdir,
                            fastai.text.TextList(items=train, 
                                                 vocab=fastai_vocab, 
                                                 path=workdir,
                                                 processor=[]),
                            fastai.text.TextList(items=validation,
                                                 vocab=fastai_vocab, 
                                                 path=workdir,
                                                 processor=[]))
    src = src.label_for_lm()
    data_lm = src.databunch(bs=batch_size, bptt=bptt)

    learner = fastai.text.language_model_learner(data_lm,
                                            get_arch(arch),
                                            config=make_configuration(arch, config),
                                            pretrained=False,
                                            drop_mult=dropout,
                                            clip=clip,
                                            wd=weight_decay)

def get_arch(arch):
    if arch == "AWD_LSTM":
        return fastai.text.AWD_LSTM
    else:
        raise ValueError("Unsupported architecture")


def make_configuration(arch, config):
    """
    """
    if arch == "AWD_LSTM":
        config_lm = fastai.text.awd_lstm_lm_config.copy()
        config_lm["emb_sz"]      = config["embed_size"]
        config_lm["n_hid"]       = config["hidden_units"]
        config_lm["n_layers"]    = config["num_layers"]
        config_lm["pad_token"]   = config["pad_index"]
        config_lm["tie_weights"] = config["tie_encoder"] # tie embedding and output for LMs
        return config_lm
    raise ValueError("Unknown architecture")


