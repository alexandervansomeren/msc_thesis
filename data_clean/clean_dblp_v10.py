import glob
import json
import pprint
import shutil

import os

try:
    os.remove('../data.tmp/dblp_v10/texts.txt')
except OSError:
    pass
try:
    os.remove('../data.tmp/dblp_v10/links.txt')
except OSError:
    pass

for jsf in glob.glob('../data.tmp/dblp_v10/dblp-ref/*.json'):
    with open(jsf) as f:
        for record in f:
            rec = json.loads(record)
            if 'references' in rec and 'abstract' in rec and len(rec['references']) >= 5:
                with open('../data.tmp/dblp_v10/texts.txt', 'a') as output_file:
                    output_file.write(rec['id'] + ' ' + rec['abstract'].replace('\n', ' ') + '\n')
                with open('../data.tmp/dblp_v10/links.txt', 'a') as output_file:
                    output_file.write(rec['id'] + ' ' + ' '.join(rec['references']) + '\n')
