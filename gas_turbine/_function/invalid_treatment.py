# -*- coding: utf-8 -*-
"""
Deal with the invaild input 
"""
#def divide(a,b):
#    try:
#        return a/b
#    except ZeroDivisionError:
#        return None

def divide(a,b):
    try:
        return a/b
    except ZeroDivisionError as e:
        return ValueError('Invalid inputs')# from e

x = 0
y = 2
result = divide(x,y)
