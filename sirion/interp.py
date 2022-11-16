
## WUDS test

alfa  = 0.5

uW = 2
uP = 3
uE = 1

alfa_p = 0.5 + alfa
alfa_m = 0.5 - alfa

uw = (0.5 + alfa)*uW + (0.5 - alfa)*uP
fw = uw * (0.5 + alfa)

ue = (0.5 + alfa)*uP + (0.5 - alfa)*uE
fe = ue*(0.5 - alfa)

fp = ue*(0.5 + alfa) - uw*(0.5 - alfa)

print(f'fw : {fw} \n')
print(f'fe : {fe} \n')
print(f'ap : {fw-fe}')
print(f'fp : {fp} \n')