## 队列
```python
from queue import Queue
```

## 优先级队列
```python
from queue import PriorityQueue


class Prior:

    def __init__(self, _id, dist):
        self.id = _id
        self.dist = dist

    def __lt__(self, other):
        return self.dist < other.dist

    def __str__(self):
        return f'{self.id}: {self.dist}'


a = PriorityQueue()
a.put(Prior('4PBL01', 1234))
a.put(Prior('4PBL01', 452))
a.put(Prior('4PBL01', 8124))

while True:
    if a.empty():
        break
    print(a.get())
```