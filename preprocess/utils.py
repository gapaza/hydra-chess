import config
import os
import re
import json
import ast
import pickle
import tensorflow as tf
import zipfile
import random
from tqdm import tqdm





def parse_ft_line(line):

    # 1. Extract: previous_moves, candidate_moves
    matches = re.findall(r'\[.*?\]', line)
    if len(matches) != 2:
        print('Error parsing line:', line)
        return None, None, None, None
    candidate_moves = json.loads(matches[0])
    previous_moves = ast.literal_eval(matches[1])

    # 2. Extract: cp_score, best_score
    try:
        cp_scores = []
        for move in candidate_moves:
            eval_str = move['eval']
            if 'Cp(' in eval_str:
                cp_score = int(re.search(r"Cp\((.+?)\)", eval_str).group(1))
            elif 'Mate(' in eval_str:
                mate_score = int(re.search(r"Mate\((.+?)\)", eval_str).group(1))
                # Assign a large score for checkmate evaluations.
                cp_score = 10000 if mate_score > 0 else -10000
            cp_scores.append(cp_score)
    except Exception as e:
        print('Error parsing line:', line, e)
        return None, None, None, None
    if 'WHITE' in candidate_moves[0]['eval']:
        best_score = max(cp_scores)
    else:
        best_score = min(cp_scores)

    return previous_moves, candidate_moves, cp_scores, best_score



def parse_ft_filter(line, mates=False, tactics=False):

    # 1. Extract: previous_moves, candidate_moves
    matches = re.findall(r'\[.*?\]', line)
    if len(matches) != 2:
        print('Error parsing line:', line)
        return None, None, None, None
    candidate_moves = json.loads(matches[0])
    previous_moves = ast.literal_eval(matches[1])

    # 2. Extract: cp_score, best_score
    try:
        cp_scores = []
        for move in candidate_moves:
            eval_str = move['eval']
            if 'Cp(' in eval_str:
                cp_score = int(re.search(r"Cp\((.+?)\)", eval_str).group(1))
            elif 'Mate(' in eval_str:
                mate_score = int(re.search(r"Mate\((.+?)\)", eval_str).group(1))
                # Assign a large score for checkmate evaluations.
                cp_score = 10000 if mate_score > 0 else -10000
            cp_scores.append(cp_score)
    except Exception as e:
        print('Error parsing line:', line, e)
        return None, None, None, None
    if 'WHITE' in candidate_moves[0]['eval']:
        best_score = max(cp_scores)
    else:
        best_score = min(cp_scores)

    # Parse Tactics / Mates
    line_segments = line.split('\t')
    if len(line_segments) < 5:
        print('Error parsing line, tactical segment missing', line)
        return None, None, None, None

    is_tactic = (line_segments[3] == '1')
    is_mate = (line_segments[4].strip() == '1')


    if mates is True and is_mate is True:
        return previous_moves, candidate_moves, cp_scores, best_score
    else:
        return None, None, None, None

    # if tactics is True and is_tactic is False and is_mate is False:
    #     return None, None, None, None
    #
    # return previous_moves, candidate_moves, cp_scores, best_score








