# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 08:52:11 2023

@author: 12631
"""

def chunk_list(numbers, medians):
    num_len = len(numbers)
    chunk_size = int(num_len / 4)
    medians = sorted(medians)
    chunked_lists = []
    start = 0
    for median in medians:
        index = bisect_left(numbers, median)
        if index == 0:
            chunked_lists.append(numbers[start:start+chunk_size])
            start += chunk_size
            continue
        if index == num_len:
            index -= 1
        if abs(numbers[index] - median) > abs(numbers[index-1] - median):
            index -= 1
        chunked_lists.append(numbers[start:index])
        start = index
    if start < num_len:
        chunked_lists.append(numbers[start:])
    while len(chunked_lists) < 4:
        chunked_lists.append([])
    for i in range(4):
        if len(chunked_lists[i]) > chunk_size:
            excess = len(chunked_lists[i]) - chunk_size
            chunked_lists[i+1] = chunked_lists[i][-excess:] + chunked_lists[i+1]
            chunked_lists[i] = chunked_lists[i][:-excess]
    return chunked_lists


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
medians = [5, 10, 15]
chunked_lists = chunk_list(numbers, medians)
print(chunked_lists)


import numpy as np

def chunk_list(input_list, num_groups, target_medians):
    # calculate the median of the input list
    median = np.median(input_list)
    
    # split the list into two halves
    left_half = [x for x in input_list if x <= median]
    right_half = [x for x in input_list if x > median]
    
    # calculate the median of each half
    left_median = np.median(left_half)
    right_median = np.median(right_half)
    
    # if either half has fewer than half the elements of the original list, discard its median and try again
    if len(left_half) < len(input_list) / 2:
        return chunk_list(input_list, num_groups, target_medians)
    if len(right_half) < len(input_list) / 2:
        return chunk_list(input_list, num_groups, target_medians)
    
    # divide each half into two subgroups based on its median
    left_left = [x for x in left_half if x <= left_median]
    left_right = [x for x in left_half if x > left_median]
    right_left = [x for x in right_half if x <= right_median]
    right_right = [x for x in right_half if x > right_median]
    
    # recursively divide each subgroup until the desired number of groups is obtained
    subgroups = [left_left, left_right, right_left, right_right]
    while len(subgroups) < num_groups:
        new_subgroups = []
        for subgroup in subgroups:
            subgroup_median = np.median(subgroup)
            subgroup_left = [x for x in subgroup if x <= subgroup_median]
            subgroup_right = [x for x in subgroup if x > subgroup_median]
            if len(subgroup_left) >= len(subgroup) / 2 and len(subgroup_right) >= len(subgroup) / 2:
                new_subgroups.append(subgroup_left)
                new_subgroups.append(subgroup_right)
            else:
                # if a subgroup has fewer than half the elements of its parent subgroup, discard its median and try again
                new_subgroups.extend(chunk_list(subgroup, 2, [np.median(subgroup)]))
        subgroups = new_subgroups
    
    # calculate the median of each subgroup and sort them in ascending order
    subgroup_medians = [np.median(subgroup) for subgroup in subgroups]
    subgroup_medians.sort()
    
    # determine which target medians each subgroup should be assigned to
    assignments = []
    for subgroup_median in subgroup_medians:
        distances = [abs(subgroup_median - target_median) for target_median in target_medians]
        assignments.append(distances.index(min(distances)))
    
    # construct the final list of subgroups
    final_subgroups = [[] for _ in range(num_groups)]
    for i, subgroup in enumerate(subgroups):
        final_subgroups[assignments[i]].extend(subgroup)
    
    return final_subgroups



def chunk_list(input_list, num_groups, target_medians, n):
    # if input list is too short to be split into num_groups or n is greater than the length of the input list
    if len(input_list) < num_groups * n:
        return None
    
    # initialize list of subgroups with at least n elements each
    subgroups = [input_list[i:i+n] for i in range(0, len(input_list), n)]
    num_subgroups = len(subgroups)
    
    # if the number of subgroups is less than num_groups, pad the subgroups with the last element
    if num_subgroups < num_groups:
        last_subgroup = subgroups.pop()
        for i in range(num_groups - num_subgroups):
            subgroups.append(last_subgroup.copy())
    
    # calculate the median of each subgroup and sort them in ascending order
    subgroup_medians = [np.median(subgroup) for subgroup in subgroups]
    subgroup_medians.sort()
    
    # determine which target medians each subgroup should be assigned to
    assignments = []
    for subgroup_median in subgroup_medians:
        distances = [abs(subgroup_median - target_median) for target_median in target_medians]
        assignments.append(distances.index(min(distances)))
    
    # construct the final list of subgroups
    final_subgroups = [[] for _ in range(num_groups)]
    for subgroup, assignment in zip(subgroups, assignments):
        final_subgroups[assignment].extend(subgroup)
    
    # if any subgroup has less than n elements, try to merge it with the adjacent subgroups
    for i in range(num_groups):
        while len(final_subgroups[i]) < n and num_subgroups > num_groups:
            if i == 0:
                # merge with right neighbor
                final_subgroups[i].extend(final_subgroups[i+1])
                del final_subgroups[i+1]
                num_subgroups -= 1
            elif i == num_groups-1:
                # merge with left neighbor
                final_subgroups[i].extend(final_subgroups[i-1])
                del final_subgroups[i-1]
                num_subgroups -= 1
                i -= 1
            elif len(final_subgroups[i-1]) < len(final_subgroups[i+1]):
                # merge with left neighbor
                final_subgroups[i].extend(final_subgroups[i-1])
                del final_subgroups[i-1]
                num_subgroups -= 1
                i -= 1
            else:
                # merge with right neighbor
                final_subgroups[i].extend(final_subgroups[i+1])
                del final_subgroups[i+1]
                num_subgroups -= 1
        
    # if the final list of subgroups still has less than num_groups elements, return None
    if len(final_subgroups) < num_groups:
        return None
    
    return final_subgroups

(input_list, num_groups, target_medians, n)
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
medians = [5, 10, 15]
chunked_lists = chunk_list(numbers, 3, medians,1)
print(chunked_lists)
