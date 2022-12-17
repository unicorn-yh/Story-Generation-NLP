import os
import pickle
import torch
from transformers import PreTrainedTokenizer


def tokenize_longform_text(raw_text_paths: list,
                           tokenizer: PreTrainedTokenizer,
                           block_size: int,
                           drop_last=True,
                           overlap=True):
    """ Loads raw LONGFORM text from a list of paths to text files, tokenizes it, splits the tokenized
     text into training examples and returns the list. Requires passing in a HuggingFace Transformers
     pretrained tokenizer"""

    # TODO: Look into methods of text augmentation, put this in as a placeholder

    # find correct block size of the tokenizer
    block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

    # check that all the text file paths actually files
    for text_file in raw_text_paths:
        assert os.path.isfile(text_file), "{} is not a file".format(text_file)

    # make empty list to store all the examples
    examples = []

    # loop over all text files
    for text_file in raw_text_paths:

        with open(text_file, encoding="utf-8") as f:
            try:
                text = f.read()
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                print("{} successfully read and tokenized".format(text_file))
            except:
                print("Error reading or tokenizing file {}".format(text_file))

        # check that the tokenized file is at least one block size long
        len_tokens = len(tokenized_text)
        print(len_tokens)
        if len_tokens < block_size:
            print("File {} is too short for the block size".format(text_file))
            pass

        try:
            if overlap is False:
                for i in range(0, len_tokens - block_size + 1, block_size):  # don't overlap examples
                    examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size]))
            else:  # overlap examples
                for i in range(0, len_tokens - block_size + 1, int(block_size / 2)):
                    examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size]))

            if drop_last is False:
                examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[-block_size:]))

            print("Successfully split tokens from file {} into examples".format(text_file))
        except:
            print("Failed at splitting tokens from file {} into examples".format(text_file))

    print("{} examples total tokenized".format(len(examples)))

    return examples


def make_tokenized_examples(tokenizer: PreTrainedTokenizer,
                            block_size: int,
                            root_dir_path: str,
                            examples_file=None):
    """Tokenize a directory of text where the raw text is in a subdirectory '/raw_text' and the default
    is to put the tokenized text into a sub directory '/tokenized_examples' """

    # -------------------------------------------------------------------------------------------------------------
    # Do a bunch of error checking, finding text files and making output file names
    # -------------------------------------------------------------------------------------------------------------

    # check that root_dir_path is actually a path
    assert os.path.isdir(root_dir_path), "{} is not a directory".format(root_dir_path)
    # and check that we didn't accidentally put a / at the end of the dir path
    if not os.path.split(root_dir_path)[1]:
        root_dir_path = os.path.split(root_dir_path)[0]

    # check that there is a raw_text directory
    raw_dir_path = os.path.join(root_dir_path, "raw_text")
    assert os.path.isdir(raw_dir_path), "{} has no raw_text/ subdirectory".format(root_dir_path)

    # check that there are text files in there and if so get their names
    file_list = []
    for file in os.listdir(raw_dir_path):
        if file.endswith(".txt"):
            file_list.append(os.path.join(raw_dir_path, file))
    if len(file_list) == 0:
        raise RuntimeError("No text files found in {}".format(raw_dir_path))

    # now get the tokenized text file name, make the tokenized_text directory if necessary

    if examples_file is None:
        tokenized_dir = os.path.join(root_dir_path, "tokenized_examples")
        if not os.path.isdir(tokenized_dir):
            os.mkdir(tokenized_dir)

        print(root_dir_path)
        print(os.path.split(root_dir_path))

        author_name = os.path.split(root_dir_path)[1]
        examples_file = "examples_gpt2_blocksize_{}_{}.pkl".format(block_size, author_name)
        examples_file = os.path.join(tokenized_dir, examples_file)

    else:
        assert type(examples_file) is str, "tokenized_file_name must be a string or None"
        tokenized_dir = os.path.split(examples_file)[0]
        assert os.path.isdir(tokenized_dir), "{} is not a directory".format(tokenized_dir)

    # -------------------------------------------------------------------------------------------------------------
    # After all that make the examples and save them
    # -------------------------------------------------------------------------------------------------------------

    # tokenize all the files, split them into examples and concatenate them
    examples = tokenize_longform_text(file_list, tokenizer, block_size, drop_last=False, overlap=True)

    # save them as a pickle
    with open(examples_file, 'wb') as f:
        pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("{} examples created and saved in {}".format(len(examples), examples_file))

