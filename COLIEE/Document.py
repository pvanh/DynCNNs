import sys

import re

def Split(text ='', delimiters=' |,'):
    words = re.split('(?:' + delimiters + ')*', text)
    return words
class Art: #Article
    def __init__(self, id='', sens=None):
        self.id = id
        if sens is None:
            self.sens=[]
        else:
            self.sens = sens

    def show(self, buf=sys.stdout):
        buf.write(self.id+': ')
        for s in self.sens:
            s.show(buf=buf)
class Sen: # sentence
    def __init__(self, text ='', parse =''):
        self.text = text
        self.parse = parse
    def show(self, buf=sys.stdout):
        buf.write(self.text)
    def getWords(self, delimiters=' |,|\n'):
        words = re.split('(?:' + delimiters + ')*', self.text)
        return words
class Ques:
    def __init__(self, id ='',sen=None, article_ids=None, label=''):
        self.id = id
        self.label = label
        if sen is None:
            self.sen =Sen()
        else:
            self.sen = sen
        if article_ids is None:
            self.article_ids=[]
        else:
            self.article_ids

    def show(self, buf=sys.stdout):
        buf.write(self.id+': ')
        self.sen.show(buf= buf)
        buf.write('\nrelevant articles: ')
        for art in self.article_ids:
            buf.write(art+', ')