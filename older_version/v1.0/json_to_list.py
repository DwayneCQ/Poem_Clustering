from config import *

def json_to_list(file_name, poems_num):
    poems = []
    authors = []
    strains = []
    titles = []

    titleField = "title"
    contentField = "paragraphs"
    authorField = "author"
    strainField = "strains"

    path_list = ['./Datasets/' + file_name + '.{}.json'.format(i) for i in range(0, poems_num + 1000, 1000)]
    for path in path_list:
        if os.path.isfile(path) and re.match('(.*)(\.)(json)', path) != None:
            print("processing file: %s" % path)
            poems_json = json.load(open(path, "r"))
            for singlePoem in poems_json:
                poem = "".join(singlePoem[contentField])
                poems.append(poem)
                titles.append(singlePoem[titleField])
    return poems, titles

