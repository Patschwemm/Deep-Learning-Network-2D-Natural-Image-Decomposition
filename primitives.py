from typing import List

class Rectangle:
    def __init__(self, center: List[int], length:int, width: int):
        self.center = center
        self.length = length
        self.width = width


    @property
    def __center__(self):
        return self.center

    @property 
    def __height__(self):
        return self.length

    @property 
    def __width__(self):
        return self.width




    
