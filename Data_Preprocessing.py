from nltk.tokenize import word_tokenize
from Categories_Data import categories
import numpy as np
import codecs
import glob
import os
import re

class Data_Preprocessor:

    def tokenize(self, text):
        tokens = word_tokenize(text)
        alpha = [tk for tk in tokens if unicode(tk).isalpha()]
        return alpha

    def header_not_fully_removed(self, text):
        if ":" in text.splitlines()[0]:
            return len(text.splitlines()[0].split(":")[0].split()) == 1
        else:
            return False

    def strip_newsgroup_header(self, text):
        _before, _blankline, after = text.partition('\n\n')
        if len(after) > 0 and self.header_not_fully_removed(after):
            after = self.strip_newsgroup_header(after)
        return after

    def strip_newsgroup_quoting(self, text):
        _QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'r'|^In article|^Quoted from|^\||^>)')
        good_lines = [line for line in text.split('\n')
            if not _QUOTE_RE.search(line)]
        return '\n'.join(good_lines)

    def strip_newsgroup_footer(self, text):
        lines = text.strip().split('\n')
        for line_num in range(len(lines) - 1, -1, -1):
            line = lines[line_num]
            if line.strip().strip('-') == '':
                break
        if line_num > 0:
            return '\n'.join(lines[:line_num])
        else:
            return text

    def raw_to_vector(self, path, to_be_stripped=["header", "footer", "quoting"], noise_threshold=-1):
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
                        data = self.strip_newsgroup_quoting(data)
                    if "header" in to_be_stripped:
                        data = self.strip_newsgroup_header(data)
                    if "footer" in to_be_stripped:
                        data = self.strip_newsgroup_footer(data)
                    if len(data) > noise_threshold:
                        train_data.append(data)
                        label_data.append(category[1])
        os.chdir(base_dir)
        return np.array(train_data), np.array(label_data)