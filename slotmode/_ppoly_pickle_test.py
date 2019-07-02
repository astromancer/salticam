import pickle
from salticam.slotmode.ppoly import PPoly2D

olist = (1, 2, 3), (1, 4)
breaks = (0, 5, 10, 30), (0, 9, 12)
ppx = PPoly2D.from_orders(olist, breaks)

serialized = pickle.dumps(ppx)
clone = pickle.loads(serialized)

print(clone.y)
# print(vars(clone.y))

print(clone.x)
# print(vars(clone.x))
