#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 3
"""
import numpy as np

v = np.arange(0,12.2,1.1)
print('v = ', v)

vi = v[::-1]
print('\nvi = ', vi)

v1 = v[::2]
v2 = v[1::2]
print('\nv1 = ', v1)
print('v2 = ', v2)

v1 = v[::3]
v2 = v[1::3]
v3 = v[2::3]
print('\nv1 = ', v1)
print('v2 = ', v2)
print('v3 = ', v3)

v1 = v[::4]
v2 = v[1::4]
v3 = v[2::4]
v4 = v[3::4]
print('\nv1 = ', v1)
print('v2 = ', v2)
print('v3 = ', v3)
print('v4 = ', v4)
