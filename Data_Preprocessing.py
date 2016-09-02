from nltk.tokenize import word_tokenize
from Categories_Data import categories
import numpy as np
import codecs
import glob
import os
import re

class Data_Preprocessor:

    """
    This class contains utility methods in order to process the 20 NewsGroup DataSet
    """

    """ Takes the following parameters as an input:
        text : text to be tokenized (string)
        Returns:
        alpha : the tokenized text with non alphabetical values removed (list of strings)
    """
    @staticmethod
    def tokenize(text):
        tokens = word_tokenize(text)
        alpha = [t for t in tokens if unicode(t).isalpha()]
        return alpha

    """ Takes the following parameters as an input:
        text : text whose header may still be present (string)
        Returns:
        True or False depending on the presence of part of the header
    """
    @staticmethod
    def header_not_fully_removed(text):
        if ":" in text.splitlines()[0]:
            return len(text.splitlines()[0].split(":")[0].split()) == 1
        else:
            return False

    """ Takes the following parameters as an input:
        text : text with header (string)
        Returns:
        after : text without header
    """
    @staticmethod
    def strip_newsgroup_header(text):
        _before, _blankline, after = text.partition('\n\n')
        if len(after) > 0 and Data_Preprocessor.header_not_fully_removed(after):
            after = Data_Preprocessor.strip_newsgroup_header(after)
        return after

    """ Takes the following parameters as an input:
        text : text with quotes (string)
        Returns:
        text without quotes
    """
    @staticmethod
    def strip_newsgroup_quoting(text):
        _QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'r'|^In article|^Quoted from|^\||^>)')
        good_lines = [line for line in text.split('\n')
            if not _QUOTE_RE.search(line)]
        return '\n'.join(good_lines)

    """ Takes the following parameters as an input:
        text : text with footer (string)
        Returns:
        text without footer
    """
    @staticmethod
    def strip_newsgroup_footer(text):
        lines = text.strip().split('\n')
        for line_num in range(len(lines) - 1, -1, -1):
            line = lines[line_num]
            if line.strip().strip('-') == '':
                break
        if line_num > 0:
            return '\n'.join(lines[:line_num])
        else:
            return text

    """ Takes the following parameters as an input:
        path: path to the DataSet folder (string)
        to_be_stripped: specifies which elements to remove (list of strings)
        noise_threshold: specifies which document sizes to ignore (integer)
        Returns:
        train_data: samples in raw (text) format (numpy array of strings)
        label_data: labels' data (numpy of integers)
    """
    @staticmethod
    def raw_to_vector(path, to_be_stripped=["header", "footer", "quoting"], noise_threshold=-1):
        base_dir = os.getcwd()
        train_data = []
        label_data = []
        for category in categories:
            os.chdir(base_dir)
            os.chdir(path+"/"+category[0])
            for filename in glob.glob("*"):
                with codecs.open(filename, 'r', encoding='utf-8', errors='replace') as target:
                    data = target.read()
                    if "quoting" in to_be_stripped:
                        data = Data_Preprocessor.strip_newsgroup_quoting(data)
                    if "header" in to_be_stripped:
                        data = Data_Preprocessor.strip_newsgroup_header(data)
                    if "footer" in to_be_stripped:
                        data = Data_Preprocessor.strip_newsgroup_footer(data)
                    data = re.sub("[^a-zA-Z]", " ", data)
                    if len(data) > noise_threshold:
                        train_data.append(data)
                        label_data.append(category[1])
        os.chdir(base_dir)
        return np.array(train_data), np.array(label_data)

    """ Takes the following parameters as an input:
        path: path to the saved vector folder (string)
    """
    @staticmethod
    def clean_saved_vector(path):
        map(os.unlink, [os.path.join(path, f) for f in os.listdir(path)] )
