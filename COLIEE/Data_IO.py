import json

from Document import Ques, Sen, Art


def getQuestions(filename =''):
    file = open(filename)
    jelements = json.loads(file.read())
    file.close()
    questions =[]
    for e in jelements:
        qes = Ques()
        qes.id = e["id"]
        qes.label = e["label"]
        qes.sen = Sen(text=e['question'].strip().lower(), parse=e['parse'])
        for a in e['relevant_articles']:
            qes.article_ids.append(a[0][8:]) # id of article

            questions.append(qes)

    return questions

def getArticles(filename =''):
    articles = {}
    file = open(filename)
    jelements = json.loads(file.read())
    file.close()
    for e in jelements:  # articles
        art = Art()
        art.id = e["article_id"]
        for s in e["sentences"]:
            art.sens.append(Sen(text=s["text"].strip().lower(), parse=s['parse']))

        articles[art.id] = art

    return articles