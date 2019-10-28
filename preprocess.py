#!/usr/bin/env python
# coding=utf8

import sys


def preprocess(f_txt, f_out):
    '''
        make data ba ->ba5
        make data di1ng->ding1
    '''
    with open(f_out, 'w') as fout:
        with open(f_txt, 'r') as f:
            index = 0
            all_content = f.readlines()
            for sline in all_content:
                index += 1
                print('-----%d\n' % index)	
                cn, pronunciation = sline.split(':')
                words = pronunciation.strip().split(' ')
                swords = ''
                for w in words:
                    tone = '5'
                    temp_word = ''
                    for sw in w:
                        if sw in '0123456789':
                            tone = sw
                        else:
                            temp_word += sw
                    temp_word += tone        
                    swords += temp_word + ' '
                swords = swords[0:-1] # remove the last ' '
                fout.write(cn + '\t' + swords + '\n')


if __name__ == '__main__':
    preprocess(sys.argv[1], sys.argv[2])            

