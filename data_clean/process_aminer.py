import os
refs = []
has_refs = False
title = ''
file_index = ''
with open('./v1/outputacm_aminer.txt', 'r') as f:
    for line in f:
        line = str(line)
        if line[:2] == '#*':
            title = line[2:-1]
        elif line[:6] == '#index':
            file_index = line[6:-1]
        elif line[:2] == '#%':
            refs.append(line[2:-1])
            has_refs = True
        elif line[:2] == '#!':
            abstract = line[2:-1]
            if has_refs:
                with open('./v1/v1_abstracts.txt', 'a') as output_file:
                    output_file.write(file_index + ' ' + abstract + '\n')
                with open('./v1/v1_refs.txt', 'a') as output_file:
                    output_file.write(file_index + ' ' + ' '.join(refs) + '\n')
                has_refs = False
                refs = []
