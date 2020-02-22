# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:45:04 2020

@author: Eric Su
"""

class foo:
    def __init__(self):
        print('foo')
        
class bar:
    def __init__(self, Bar):
        print(Bar)
        
        
class foobar(foo, bar):
    def __init__(self, Bar='Bar'):
        super(foobar, self).__init__()
        super(foo, self).__init__(Bar)
        super(bar, self).__init__()
        
        print('foobar')
        
a = foobar('abc')

foobar.mro()

class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width


class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)
        
        
class Cube(Square):
    def surface_area(self):
        face_area = super().area()
        return face_area * 6

    def volume(self):
        face_area = super().area()
        return face_area * self.length
    

class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height

class RightPyramid(Triangle, Square):
    def __init__(self, base, slant_height):
        self.base = base
        self.slant_height = slant_height

    def area(self):
        base_area = super().area()
        perimeter = super().perimeter()
        return 0.5 * perimeter * self.slant_height + base_area
    

pyramid = RightPyramid(2, 4)
pyramid.area()


